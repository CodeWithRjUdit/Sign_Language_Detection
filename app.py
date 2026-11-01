from __future__ import annotations

import io
import shutil
import tempfile
import threading
import time
import zipfile
from collections import Counter, deque
from pathlib import Path
from typing import Any

import av
import h5py
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
MAX_INFERENCE_DIMENSION = 640
CAMERA_TARGET_FPS = 15
MODEL_LOCK = threading.Lock()


def build_trained_architecture():
    """Recreate the architecture exactly as defined in the training notebook."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_COUNT)),
            tf.keras.layers.LSTM(
                64, return_sequences=True, activation="relu", name="lstm_3"
            ),
            tf.keras.layers.LSTM(
                128, return_sequences=True, activation="relu", name="lstm_4"
            ),
            tf.keras.layers.LSTM(
                64, return_sequences=False, activation="relu", name="lstm_5"
            ),
            tf.keras.layers.Dense(64, activation="relu", name="dense_3"),
            tf.keras.layers.Dense(32, activation="relu", name="dense_4"),
            tf.keras.layers.Dense(
                len(ACTIONS), activation="softmax", name="dense_5"
            ),
        ],
        name="sequential_1",
    )
    # Build variables before loading the stored parameter tensors.
    model(tf.zeros((1, SEQUENCE_LENGTH, FEATURE_COUNT), dtype=tf.float32))
    return model


def load_weights_file(model, source_path: Path, temp_dir: str):
    """Load either modern Keras 3 weights or legacy full-model HDF5 weights."""
    if not h5py.is_hdf5(source_path):
        with open(source_path, "rb") as file_handle:
            header = file_handle.read(16)
        raise ValueError(
            f"{source_path.name} is not a readable HDF5 weights file. "
            f"File header: {header!r}"
        )

    # Keras 3 uses a different reader for its native .weights.h5 layout and
    # legacy HDF5 files produced by Keras 2. Detect the layout from HDF5 keys,
    # then choose the extension that selects the matching Keras loader.
    with h5py.File(source_path, "r") as h5_file:
        root_keys = set(h5_file.keys())
        is_keras3_weights = (
            "layers" in root_keys
            and "vars" in root_keys
            and "model_weights" not in root_keys
            and "layer_names" not in h5_file.attrs
        )

    filename = "restored.weights.h5" if is_keras3_weights else "restored_legacy.h5"
    destination = Path(temp_dir) / filename
    shutil.copyfile(source_path, destination)
    model.load_weights(destination)
    return model


@st.cache_resource(show_spinner="Loading the trained LSTM model...")
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Keep my_model.keras in the repository root.")

    errors = []
    with tempfile.TemporaryDirectory(prefix="sign_model_") as temp_dir:
        # The training notebook may have produced a legacy HDF5 model despite
        # the .keras suffix. Detect the real file format by its contents,
        # rather than trusting the filename extension.
        if h5py.is_hdf5(MODEL_PATH):
            model = build_trained_architecture()
            try:
                model = load_weights_file(model, MODEL_PATH, temp_dir)
            except Exception as exc:
                errors.append(f"my_model.keras HDF5 weights: {exc!r}")
                model = None
        elif zipfile.is_zipfile(MODEL_PATH):
            model = None
            try:
                with zipfile.ZipFile(MODEL_PATH, "r") as source_zip:
                    candidates = [
                        name for name in source_zip.namelist()
                        if name.endswith(".weights.h5") or name.endswith("model.weights.h5")
                    ]
                    if not candidates:
                        candidates = [
                            name for name in source_zip.namelist()
                            if name.endswith(".h5")
                        ]
                    if not candidates:
                        raise ValueError(
                            "The .keras archive contains no embedded HDF5 weights."
                        )
                    embedded = Path(temp_dir) / "embedded.weights.h5"
                    embedded.write_bytes(source_zip.read(candidates[0]))

                model = build_trained_architecture()
                model = load_weights_file(model, embedded, temp_dir)
            except Exception as exc:
                errors.append(f"my_model.keras archive weights: {exc!r}")
                model = None
        else:
            with open(MODEL_PATH, "rb") as file_handle:
                header = file_handle.read(32)
            errors.append(
                "my_model.keras is neither a valid .keras ZIP archive nor "
                f"an HDF5 file. Header: {header!r}"
            )
            model = None

        # The second repository artifact is also tried as a weights source.
        # Loading weights into the notebook-defined architecture avoids
        # deserializing its legacy Orthogonal initializer configuration.
        if model is None:
            legacy_path = ROOT / "actions.h5"
            if legacy_path.exists() and h5py.is_hdf5(legacy_path):
                try:
                    model = build_trained_architecture()
                    model = load_weights_file(model, legacy_path, temp_dir)
                except Exception as exc:
                    errors.append(f"actions.h5 weights: {exc!r}")
                    model = None

        if model is None:
            raise RuntimeError(
                "Unable to restore the trained parameters. "
                + " | ".join(errors)
            )

    shape = model.input_shape
    if len(shape) != 3 or tuple(shape[1:]) != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(
            f"Expected model input (None, {SEQUENCE_LENGTH}, {FEATURE_COUNT}), "
            f"but found {shape}."
        )
    if int(model.output_shape[-1]) != len(ACTIONS):
        raise ValueError(
            f"Expected three output classes, but found {model.output_shape[-1]}."
        )
    return model


def create_holistic_model():
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        # Complexity 0 is faster and more suitable for a CPU-hosted demo.
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def resize_for_inference(rgb: np.ndarray) -> np.ndarray:
    """Downscale video frames before MediaPipe to reduce CPU and memory use."""
    height, width = rgb.shape[:2]
    longest_side = max(height, width)
    if longest_side <= MAX_INFERENCE_DIMENSION:
        return rgb

    scale = MAX_INFERENCE_DIMENSION / float(longest_side)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return np.asarray(
        Image.fromarray(rgb).resize(new_size, Image.Resampling.BILINEAR)
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
        self.process_interval = 1.0 / CAMERA_TARGET_FPS
        self.last_process_time = 0.0
        self.last_output = None
        self.lock = threading.Lock()
        self.status = {
            "label": None,
            "confidence": 0.0,
            "probabilities": np.zeros(len(ACTIONS), dtype=np.float32),
            "frames_collected": 0,
            "message": "Waiting for camera frames",
        }

    def get_status(self):
        with self.lock:
            return {
                **self.status,
                "probabilities": self.status["probabilities"].copy(),
            }

    def _video_frame(self, rgb: np.ndarray) -> av.VideoFrame:
        self.last_output = rgb
        return av.VideoFrame.from_ndarray(rgb, format="rgb24")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        now = time.monotonic()

        # Drop excess incoming frames instead of building a processing backlog.
        # Re-send the latest small preview while waiting for the next inference slot.
        if (
            self.last_output is not None
            and now - self.last_process_time < self.process_interval
        ):
            return av.VideoFrame.from_ndarray(self.last_output, format="rgb24")

        self.last_process_time = now
        rgb = frame.to_ndarray(format="rgb24")
        rgb = resize_for_inference(rgb)
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
                    "frames_collected": 0,
                    "message": "No person detected — move into camera view",
                }
            return self._video_frame(overlay(rgb, "Move into the camera view"))

        self.sequence.append(extract_keypoints(results))
        with self.lock:
            self.status["frames_collected"] = len(self.sequence)
            if len(self.sequence) < SEQUENCE_LENGTH:
                self.status["message"] = (
                    f"Preparing sequence: {len(self.sequence)}/{SEQUENCE_LENGTH} frames"
                )

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
                label, message = ACTIONS[voted_index], "Sign recognized"
            else:
                label, message = None, "Analyzing — hold the sign steady"

            with self.lock:
                self.status = {
                    "label": label,
                    "confidence": confidence,
                    "probabilities": average,
                    "frames_collected": len(self.sequence),
                    "message": message,
                }

        current = self.get_status()
        if len(self.sequence) < SEQUENCE_LENGTH:
            headline = f"Preparing: {len(self.sequence)}/{SEQUENCE_LENGTH} frames"
        else:
            headline = current["message"]

        return self._video_frame(
            overlay(rgb, headline, current["label"], current["confidence"])
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
                rgb = resize_for_inference(rgb)
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
st.markdown(
    """<style>
    [data-testid="stVideo"], video { max-width: 100%; }
    div[data-testid="stMetric"] { padding: 0.25rem 0; }
    </style>""",
    unsafe_allow_html=True,
)

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
        "Start the camera, allow browser access, then keep your face, upper "
        "body, and hands visible. Hold one sign steadily for 2–3 seconds."
    )

    camera_col, result_col = st.columns([1.15, 0.85], gap="large")
    with camera_col:
        context = webrtc_streamer(
            key="sign-language-live-camera",
            video_processor_factory=lambda: SignLanguageProcessor(model),
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": CAMERA_TARGET_FPS, "max": 20},
                },
                "audio": False,
            },
            video_html_attrs={
                "style": {
                    "width": "100%",
                    "maxWidth": "520px",
                    "height": "auto",
                    "margin": "0 auto",
                    "borderRadius": "12px",
                    "objectFit": "contain",
                },
                "controls": False,
                "autoPlay": True,
                "muted": True,
                "playsInline": True,
            },
            async_processing=True,
        )

    @st.fragment(run_every="1s")
    def show_live_status(webrtc_context):
        processor = webrtc_context.video_processor if webrtc_context else None
        if processor is None:
            st.info("Camera is off. Press START in the preview to begin.")
            return

        status = processor.get_status()
        st.metric("Recognized sign", status["label"] or "Listening…")
        st.caption(status["message"])
        st.progress(
            min(status["frames_collected"] / SEQUENCE_LENGTH, 1.0),
            text=f'Frame sequence: {status["frames_collected"]}/{SEQUENCE_LENGTH}',
        )

        if status["label"]:
            st.success(f'{status["label"]} detected')
            st.metric("Model confidence", f'{status["confidence"]:.1%}')
        else:
            st.caption(
                "A result appears after the model has collected enough frames "
                "and found a stable prediction."
            )

        with st.expander("Show class probabilities"):
            chart_data = {
                ACTIONS[i]: float(status["probabilities"][i])
                for i in range(len(ACTIONS))
            }
            st.bar_chart(chart_data, y_label="Model probability")

    with result_col:
        st.subheader("Live result")
        show_live_status(context)
        st.markdown("**Tips for better recognition**")
        st.markdown(
            "- Face the camera with your upper body in view.\n"
            "- Keep both hands inside the frame.\n"
            "- Use steady lighting and a plain background.\n"
            "- Hold one sign for 2–3 seconds before changing it."
        )
        st.caption(
            "For smoother performance, the app uses a smaller camera frame "
            "and processes up to 15 frames per second."
        )

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
