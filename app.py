from __future__ import annotations

import io
import json
import tempfile
import threading
import zipfile
from collections import Counter, deque
from pathlib import Path
from typing import Any

import av
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "my_model.keras"
ACTIONS = ("Hello", "Thanks", "I Love You")
SEQUENCE_LENGTH = 30
FEATURE_COUNT = 1662
CONFIDENCE_THRESHOLD = 0.50
PREDICTION_INTERVAL = 5
MODEL_LOCK = threading.Lock()


@st.cache_resource(show_spinner="Loading the trained LSTM model...")
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Keep my_model.keras in the repository root.")
    # Older .keras archives may serialize built-in initializers with
    # legacy metadata (module/registered_name/shared_object_id) that some
    # Keras 3 loaders do not accept through initializers.get(). Normalize
    # only the Orthogonal initializer in a temporary copy of config.json.
    # The stored model.weights.h5 file is copied byte-for-byte.
    with tempfile.TemporaryDirectory(prefix="sign_model_") as temp_dir:
        with zipfile.ZipFile(MODEL_PATH, "r") as source_zip:
            names = source_zip.namelist()
            if "config.json" not in names:
                raise ValueError("The .keras archive does not contain config.json.")
            config = json.loads(source_zip.read("config.json").decode("utf-8"))

            def normalize_orthogonal(value):
                if isinstance(value, dict):
                    if (
                        value.get("class_name") == "Orthogonal"
                        and str(value.get("module", "")).startswith("keras.initializers")
                    ):
                        # The saved recurrent weights replace this initializer
                        # during loading; the canonical string is sufficient.
                        return "orthogonal"

                    normalized = {}
                    for key, item in value.items():
                        # Keras 3 removed the legacy RNN time_major argument.
                        # The saved model uses batch-major input (batch, time,
                        # features), so removing time_major=False preserves the
                        # model's data layout and computation.
                        if (
                            key == "time_major"
                            and value.get("class_name") in {
                                "LSTM", "GRU", "SimpleRNN", "RNN"
                            }
                        ):
                            continue
                        normalized[key] = normalize_orthogonal(item)
                    return normalized
                if isinstance(value, list):
                    return [normalize_orthogonal(item) for item in value]
                return value

            repaired_config = normalize_orthogonal(config)
            repaired_model_path = Path(temp_dir) / "model_compatible.keras"

            with zipfile.ZipFile(
                repaired_model_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as repaired_zip:
                for item in source_zip.infolist():
                    contents = source_zip.read(item.filename)
                    if item.filename == "config.json":
                        contents = json.dumps(repaired_config).encode("utf-8")
                    repaired_zip.writestr(item, contents)

        loaded = tf.keras.models.load_model(
            repaired_model_path,
            compile=False,
        )
    shape = loaded.input_shape
    if len(shape) != 3 or tuple(shape[1:]) != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(
            f"Expected model input (None, {SEQUENCE_LENGTH}, {FEATURE_COUNT}), "
            f"but found {shape}."
        )
    if int(loaded.output_shape[-1]) != len(ACTIONS):
        raise ValueError("The model output count does not match the action labels.")
    return loaded


def create_holistic_model():
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def has_landmarks(results) -> bool:
    return any((
        results.pose_landmarks,
        results.face_landmarks,
        results.left_hand_landmarks,
        results.right_hand_landmarks,
    ))


def extract_keypoints(results) -> np.ndarray:
    if results.pose_landmarks:
        pose = np.array(
            [[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    if results.face_landmarks:
        face = np.array(
            [[p.x, p.y, p.z] for p in results.face_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        face = np.zeros(468 * 3, dtype=np.float32)

    if results.left_hand_landmarks:
        left_hand = np.array(
            [[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        left_hand = np.zeros(21 * 3, dtype=np.float32)

    if results.right_hand_landmarks:
        right_hand = np.array(
            [[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        right_hand = np.zeros(21 * 3, dtype=np.float32)

    keypoints = np.concatenate((pose, face, left_hand, right_hand))
    if keypoints.shape != (FEATURE_COUNT,):
        raise ValueError(f"Expected 1,662 features; extracted {keypoints.size}.")
    return keypoints


def predict_sequence(model: Any, sequence) -> np.ndarray:
    batch = np.asarray(sequence, dtype=np.float32)[np.newaxis, ...]
    with MODEL_LOCK:
        return model(batch, training=False).numpy()[0].astype(np.float32)


def overlay(rgb: np.ndarray, headline: str, label=None, confidence=0.0):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 58), fill=(15, 23, 42))
    draw.text((14, 8), headline, fill="white")
    if label:
        draw.text((14, 31), f"{label}  |  {confidence:.0%}", fill=(120, 230, 170))
    return np.asarray(image)


class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.holistic = create_holistic_model()
        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_history = deque(maxlen=5)
        self.probability_history = deque(maxlen=5)
        self.frame_count = 0
        self.lock = threading.Lock()
        self.status = {
            "label": None,
            "confidence": 0.0,
            "probabilities": np.zeros(len(ACTIONS), dtype=np.float32),
            "message": "Waiting for camera frames",
        }

    def get_status(self):
        with self.lock:
            return {**self.status, "probabilities": self.status["probabilities"].copy()}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        rgb = frame.to_ndarray(format="rgb24")
        results = self.holistic.process(rgb)
        self.frame_count += 1

        if not has_landmarks(results):
            self.sequence.clear()
            self.prediction_history.clear()
            self.probability_history.clear()
            with self.lock:
                self.status = {
                    "label": None,
                    "confidence": 0.0,
                    "probabilities": np.zeros(len(ACTIONS), dtype=np.float32),
                    "message": "No person detected — move into camera view",
                }
            return av.VideoFrame.from_ndarray(
                overlay(rgb, "Move into the camera view"), format="rgb24"
            )

        self.sequence.append(extract_keypoints(results))

        if (
            len(self.sequence) == SEQUENCE_LENGTH
            and self.frame_count % PREDICTION_INTERVAL == 0
        ):
            probs = predict_sequence(self.model, self.sequence)
            self.prediction_history.append(int(np.argmax(probs)))
            self.probability_history.append(probs)
            average = np.mean(np.stack(self.probability_history), axis=0)
            voted_index, votes = Counter(self.prediction_history).most_common(1)[0]
            confidence = float(average[voted_index])
            if votes >= 3 and confidence >= CONFIDENCE_THRESHOLD:
                label, message = ACTIONS[voted_index], "Recognized sign"
            else:
                label, message = None, "Analyzing — hold the sign steady"
            with self.lock:
                self.status = {
                    "label": label,
                    "confidence": confidence,
                    "probabilities": average,
                    "message": message,
                }

        current = self.get_status()
        headline = (
            f"Collecting frames: {len(self.sequence)}/{SEQUENCE_LENGTH}"
            if len(self.sequence) < SEQUENCE_LENGTH
            else current["message"]
        )
        return av.VideoFrame.from_ndarray(
            overlay(rgb, headline, current["label"], current["confidence"]),
            format="rgb24",
        )


def analyze_uploaded_video(video_bytes: bytes, model, progress):
    sequence = deque(maxlen=SEQUENCE_LENGTH)
    predictions = []
    detected_frames = 0
    frame_count = 0

    with av.open(io.BytesIO(video_bytes), mode="r") as container:
        if not container.streams.video:
            raise ValueError("The uploaded file does not contain a video stream.")
        stream = container.streams.video[0]
        total_frames = int(stream.frames or 0)

        with create_holistic_model() as holistic:
            for frame in container.decode(stream):
                rgb = frame.to_ndarray(format="rgb24")
                results = holistic.process(rgb)
                frame_count += 1
                if has_landmarks(results):
                    detected_frames += 1
                sequence.append(extract_keypoints(results))

                if len(sequence) == SEQUENCE_LENGTH and frame_count % PREDICTION_INTERVAL == 0:
                    predictions.append(predict_sequence(model, sequence))

                if total_frames and frame_count % max(1, total_frames // 100) == 0:
                    progress.progress(
                        min(frame_count / total_frames, 1.0),
                        text=f"Analyzing video: {frame_count}/{total_frames} frames",
                    )

    progress.empty()
    if frame_count < SEQUENCE_LENGTH or not predictions:
        raise ValueError(
            "This clip is too short. Upload at least 30 frames; a 2–5 second "
            "clip is recommended."
        )
    if detected_frames == 0:
        raise ValueError(
            "No person or hand landmarks were detected. Keep your upper body "
            "and hands visible in the video."
        )

    average = np.mean(np.stack(predictions), axis=0)
    best = int(np.argmax(average))
    return {
        "label": ACTIONS[best],
        "confidence": float(average[best]),
        "probabilities": average,
        "windows": len(predictions),
        "frames": frame_count,
        "detected_frames": detected_frames,
    }


st.set_page_config(page_title="Sign Language Detection", page_icon="🤟", layout="wide")
st.title("🤟 Sign Language Detection")
st.write(
    "Recognize three trained signs from a live webcam or a short video. "
    "MediaPipe extracts body, face, and hand landmarks; a trained LSTM "
    "classifies each 30-frame sequence."
)
st.caption("Available classes: Hello · Thanks · I Love You")

try:
    model = load_model()
except Exception as exc:
    st.error(f"The trained model could not be loaded: {exc}")
    st.stop()

live_tab, upload_tab, about_tab = st.tabs(
    ["📷 Live camera", "🎞️ Upload a video", "ℹ️ About"]
)

with live_tab:
    st.subheader("Live sign recognition")
    st.write(
        "Click START and allow camera access. Keep your face, upper body, "
        "and hands visible. Hold one sign steadily for about 1–2 seconds."
    )
    context = webrtc_streamer(
        key="sign-language-live-camera",
        video_processor_factory=lambda: SignLanguageProcessor(model),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    @st.fragment(run_every="1s")
    def show_live_status(webrtc_context):
        processor = webrtc_context.video_processor if webrtc_context else None
        if processor is None:
            st.info("Start the camera above. The prediction will appear here.")
            return
        status = processor.get_status()
        first, second = st.columns(2)
        first.metric("Recognized sign", status["label"] or "—")
        second.metric(
            "Model confidence",
            f'{status["confidence"]:.1%}' if status["label"] else "—",
        )
        st.caption(status["message"])
        chart_data = {
            ACTIONS[i]: float(status["probabilities"][i])
            for i in range(len(ACTIONS))
        }
        st.bar_chart(chart_data, y_label="Model probability")

    show_live_status(context)

with upload_tab:
    st.subheader("Analyze a recorded sign")
    st.write(
        "Upload a short clip containing one sign. A 2–5 second video with "
        "your face and both hands clearly visible is recommended."
    )
    uploaded_video = st.file_uploader(
        "Choose a video", type=["mp4", "mov", "avi", "mkv"]
    )
    if uploaded_video is not None:
        st.video(uploaded_video)

    run_analysis = st.button(
        "Analyze video",
        type="primary",
        disabled=uploaded_video is None,
        use_container_width=True,
    )
    if run_analysis and uploaded_video is not None:
        progress = st.progress(0, text="Preparing video...")
        try:
            result = analyze_uploaded_video(uploaded_video.getvalue(), model, progress)
            st.success("Video analysis complete.")
            a, b, c = st.columns(3)
            a.metric("Predicted sign", result["label"])
            b.metric("Average confidence", f'{result["confidence"]:.1%}')
            c.metric("Sequences analyzed", result["windows"])
            st.bar_chart(
                {
                    ACTIONS[i]: float(result["probabilities"][i])
                    for i in range(len(ACTIONS))
                },
                y_label="Average probability",
            )
            st.caption(
                f'Processed {result["frames"]} frames; landmarks were detected '
                f'in {result["detected_frames"]} frames.'
            )
        except Exception as exc:
            progress.empty()
            st.error(f"Could not analyze this video: {exc}")

with about_tab:
    st.subheader("How the model works")
    st.markdown(
        """
        1. **Landmark extraction:** MediaPipe Holistic detects pose, face, and hand landmarks.
        2. **Feature vector:** Every frame becomes 1,662 numerical values.
        3. **Temporal sequence:** The application groups 30 consecutive frames.
        4. **LSTM classification:** The saved Keras model predicts one of the three trained signs.
        """
    )
    st.markdown("**Input shape:** 30 × 1,662")
    st.markdown("**Architecture:** stacked LSTM layers followed by dense layers")
    st.warning(
        "This is a prototype for three selected signs, not a general-purpose "
        "sign-language translator. The original dataset has limited signer "
        "diversity, so accuracy may vary with different people, lighting, "
        "camera angles, and signing speed."
    )

st.divider()
st.caption("Built with Streamlit, TensorFlow/Keras, MediaPipe Holistic, and streamlit-webrtc.")
