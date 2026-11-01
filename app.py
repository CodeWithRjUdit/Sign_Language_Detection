from __future__ import annotations

import io
import os

# Keep numerical libraries from creating large thread pools on small cloud instances.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import shutil
import tempfile
import threading
import time
import zipfile
from collections import Counter, deque
from pathlib import Path
from typing import Any

import av
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "my_model.keras"
ACTIONS = ("Hello", "Thanks", "I Love You")
SEQUENCE_LENGTH = 30
FEATURE_COUNT = 1662
CONFIDENCE_THRESHOLD = 0.50
PREDICTION_INTERVAL = 5
MAX_INFERENCE_DIMENSION = 480
CAMERA_TARGET_FPS = 8
MODEL_LOCK = threading.Lock()


def build_trained_architecture():
    """Import TensorFlow only when the user starts recognition."""
    import tensorflow as tf

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        # TensorFlow may already have initialized its runtime.
        pass

    # Recreate the architecture exactly as defined in the training notebook.
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
    import h5py

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
    import h5py

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
    # MediaPipe is loaded only after the user explicitly starts a recognition mode.
    import mediapipe as mp

    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        # Complexity 1 uses the bundled full pose model and avoids
        # MediaPipe's runtime download of the lite model into site-packages.
        model_complexity=1,
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
            "candidate": None,
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
                    "candidate": None,
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
                label, candidate = ACTIONS[voted_index], ACTIONS[voted_index]
                message = "Sign recognized"
            else:
                label = None
                candidate_index = int(np.argmax(average))
                candidate = ACTIONS[candidate_index]
                confidence = float(average[candidate_index])
                message = "Possible match — hold the sign steady"

            with self.lock:
                self.status = {
                    "label": label,
                    "candidate": candidate,
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

        displayed_label = current["label"]
        if displayed_label is None and current.get("candidate"):
            displayed_label = f"Possible: {current['candidate']}"

        return self._video_frame(
            overlay(rgb, headline, displayed_label, current["confidence"])
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


st.set_page_config(
    page_title="Sign Language Studio",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#142033; --muted:#64748b; --line:#e7edf5; --blue:#3568f5; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp {
      background: radial-gradient(circle at 8% 0%,rgba(53,104,245,.08),transparent 27rem),
                  radial-gradient(circle at 94% 6%,rgba(22,184,166,.07),transparent 24rem),#f6f8fc;
      color:var(--ink);
    }
    .block-container {max-width:1320px;padding-top:1.55rem;padding-bottom:3rem;}
    [data-testid="stHeader"] {background:transparent;}
    [data-testid="stTabs"] {margin-top:.55rem;}
    [data-testid="stTab"] {font-weight:700;padding-top:.85rem;padding-bottom:.85rem;}
    [data-testid="stMetric"] {background:#fff;border:1px solid var(--line);border-radius:16px;padding:14px 16px;box-shadow:0 5px 20px rgba(20,32,51,.035);}
    [data-testid="stMetricLabel"] {color:#64748b;font-weight:600;}
    [data-testid="stMetricValue"] {color:#142033;font-family:'Manrope',sans-serif;}
    div[data-testid="stButton"] button {border-radius:12px;font-weight:700;min-height:2.8rem;}
    div[data-testid="stFileUploader"] {border:1px dashed #cbd5e1;border-radius:16px;padding:.7rem;background:#fbfdff;}
    .hero {position:relative;overflow:hidden;padding:2rem 2.1rem;border-radius:26px;color:#fff;background:linear-gradient(118deg,#101d38 0%,#183568 53%,#116b75 100%);box-shadow:0 18px 50px rgba(24,53,104,.18);margin-bottom:1.15rem;}
    .hero:after {content:"";position:absolute;width:330px;height:330px;right:-80px;top:-180px;border-radius:50%;border:1px solid rgba(255,255,255,.17);box-shadow:0 0 0 35px rgba(255,255,255,.035),0 0 0 75px rgba(255,255,255,.025);}
    .hero-kicker {display:inline-flex;align-items:center;gap:.45rem;border:1px solid rgba(255,255,255,.24);background:rgba(255,255,255,.09);border-radius:99px;padding:.32rem .72rem;color:#d9e8ff;font-size:.76rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;}
    .hero h1 {font-family:'Manrope',sans-serif;color:#fff;font-size:clamp(2rem,4vw,3.25rem);line-height:1.1;letter-spacing:-.045em;margin:.9rem 0 .55rem;font-weight:800;}
    .hero p {color:#d8e5f7;font-size:1.03rem;max-width:720px;margin:0;line-height:1.65;}
    .hero-foot {margin-top:1.3rem;display:flex;flex-wrap:wrap;gap:.55rem;}
    .hero-chip {padding:.38rem .72rem;border-radius:99px;background:rgba(255,255,255,.1);color:#edf5ff;border:1px solid rgba(255,255,255,.13);font-size:.82rem;font-weight:600;}
    .section-kicker {color:#3568f5;font-size:.72rem;text-transform:uppercase;letter-spacing:.11em;font-weight:800;margin-bottom:.25rem;}
    .section-heading {font-family:'Manrope',sans-serif;color:#142033;font-size:1.35rem;font-weight:800;letter-spacing:-.025em;margin:0 0 .35rem;}
    .section-subtitle {color:#64748b;font-size:.92rem;line-height:1.55;margin:0 0 .9rem;}
    .soft-panel {background:#fff;border:1px solid #e7edf5;border-radius:18px;padding:1rem 1.1rem;box-shadow:0 6px 22px rgba(20,32,51,.035);}
    .note-panel {border-left:4px solid #f0b44b;background:#fffaf0;color:#73521b;border-radius:12px;padding:.85rem 1rem;line-height:1.55;font-size:.88rem;}
    .sign-card {background:#fff;border:1px solid #e7edf5;border-radius:20px;overflow:hidden;box-shadow:0 8px 26px rgba(20,32,51,.045);height:100%;}
    .sign-gif-wrap {display:flex;justify-content:center;align-items:center;background:#f1f5fb;padding:.75rem;min-height:270px;border-bottom:1px solid #e7edf5;}
    .sign-gif-wrap img {display:block;width:100%;max-width:245px;height:300px;object-fit:contain;border-radius:12px;background:#fff;}
    .sign-card-body {padding:1rem 1.1rem 1.15rem;}
    .sign-card-body h3 {color:#142033;font-family:'Manrope',sans-serif;font-size:1.15rem;margin:0 0 .2rem;font-weight:800;}
    .sign-meaning {color:#3568f5;font-weight:700;font-size:.82rem;margin-bottom:.65rem;}
    .sign-card-body p {color:#526176;font-size:.88rem;line-height:1.55;margin:.3rem 0;}
    .demo-link {display:inline-block;margin-top:.7rem;color:#2757d8!important;font-size:.84rem;font-weight:800;text-decoration:none;}
    .demo-link:hover {text-decoration:underline;}
    .footer-note {color:#8190a4;font-size:.78rem;text-align:center;padding-top:.6rem;}
    @media(max-width:700px) {
      .block-container {padding-left:1rem;padding-right:1rem;}
      .hero {padding:1.45rem 1.25rem;border-radius:20px;}
      .sign-gif-wrap {min-height:210px;}
      .sign-gif-wrap img {height:235px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">🤟 &nbsp; Computer vision · Sequence learning</div>
      <h1>Sign Language Studio</h1>
      <p>
        A camera-based sign recognition demo powered by MediaPipe Holistic
        and your trained LSTM model. Make a sign, hold it in view, and see
        what the model recognises.
      </p>
      <div class="hero-foot">
        <span class="hero-chip">Hello</span>
        <span class="hero-chip">Thanks</span>
        <span class="hero-chip">I Love You</span>
        <span class="hero-chip">30-frame sequence model</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-kicker">Your sign recognition workspace</div>
    <div class="section-heading">Choose a workspace</div>
    <div class="section-subtitle">
      The learning guide opens first. Start the AI engine only when you want
      to use the webcam or analyze a recorded clip.
    </div>
    """,
    unsafe_allow_html=True,
)

active_view = st.radio(
    "Workspace",
    options=["Live camera", "Upload a clip", "Learn the signs", "About"],
    index=2,
    horizontal=True,
    label_visibility="collapsed",
    key="workspace_view",
)

if "recognition_enabled" not in st.session_state:
    st.session_state["recognition_enabled"] = False

# TensorFlow and the trained model are loaded only when recognition is requested.
model = None
model_error = None
if active_view in {"Live camera", "Upload a clip"}:
    if not st.session_state["recognition_enabled"]:
        st.info(
            "The AI engine is not loaded yet. Loading it uses additional memory. "
            "The learning guide does not need the AI engine."
        )
        if st.button(
            "Load AI recognition engine",
            type="primary",
            key="load_recognition_engine",
        ):
            st.session_state["recognition_enabled"] = True
            st.rerun()
    else:
        try:
            with st.spinner("Loading the trained sign-recognition model..."):
                model = load_model()
        except Exception as exc:
            model_error = str(exc)
            st.error("The trained model could not be loaded.")
            with st.expander("View model-loading details"):
                st.code(model_error)

if st.session_state["recognition_enabled"]:
    unload_col, status_col = st.columns([1, 3], vertical_alignment="center")
    with unload_col:
        if st.button("Unload AI engine", key="unload_recognition_engine"):
            st.session_state["recognition_enabled"] = False
            st.cache_resource.clear()
            import gc
            gc.collect()
            st.rerun()
    with status_col:
        st.caption(
            "Unload the model when you finish testing to release its cached resources."
        )

if active_view == "Live camera":
    st.markdown(
        """
        <div class="section-kicker">Live recognition</div>
        <div class="section-heading">Try a sign in front of your camera</div>
        <div class="section-subtitle">
          Press START, allow camera access, then keep your face, upper body,
          and signing hand clearly visible.
        </div>
        """,
        unsafe_allow_html=True,
    )

    camera_col, result_col = st.columns([1.12, 0.88], gap="large")

    with camera_col:
        with st.container(border=True):
            st.markdown("**CAMERA PREVIEW**")
            if model is None:
                context = None
                st.info("Camera recognition will be available when the model loads.")
            else:
                context = webrtc_streamer(
                    key="sign-language-live-camera",
                    video_processor_factory=lambda: SignLanguageProcessor(model),
                    rtc_configuration={
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    },
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 480},
                            "height": {"ideal": 360},
                            "frameRate": {"ideal": CAMERA_TARGET_FPS, "max": 10},
                        },
                        "audio": False,
                    },
                    video_html_attrs={
                        "style": {
                            "width": "100%",
                            "maxWidth": "100%",
                            "height": "auto",
                            "aspectRatio": "4 / 3",
                            "margin": "0 auto",
                            "borderRadius": "14px",
                            "objectFit": "contain",
                            "backgroundColor": "#101d38",
                        },
                        "controls": False,
                        "autoPlay": True,
                        "muted": True,
                        "playsInline": True,
                    },
                    async_processing=True,
                )
                st.caption("Camera preview is kept compact to reduce screen clutter.")

    with result_col:
        with st.container(border=True):
            st.markdown("**RECOGNITION RESULT**")

            @st.fragment(run_every="1s")
            def show_live_status(webrtc_context):
                if model is None:
                    st.warning("Live recognition is paused until the trained model loads.")
                    return
                processor = webrtc_context.video_processor if webrtc_context else None
                if processor is None:
                    st.info("Your camera is off. Press START in the preview.")
                    st.markdown(
                        """
                        <div class="note-panel">
                          Tip: sit about an arm's length from the camera and
                          keep your hands inside the frame.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    return

                status = processor.get_status()
                if status["label"]:
                    st.success("SIGN RECOGNISED")
                    st.markdown(
                        f"<div style='font-family:Manrope,sans-serif;"
                        f"font-size:2rem;font-weight:800;color:#142033;"
                        f"line-height:1.2;padding:.25rem 0 .65rem 0;'>"
                        f"{status['label']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.metric("Model confidence", f'{status["confidence"]:.1%}')
                elif status.get("candidate"):
                    st.caption("TOP CANDIDATE · NOT YET CONFIRMED")
                    st.markdown(
                        f"<div style='font-family:Manrope,sans-serif;"
                        f"font-size:1.8rem;font-weight:800;color:#2757d8;"
                        f"line-height:1.2;padding:.25rem 0 .45rem 0;'>"
                        f"{status['candidate']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.metric("Top-class probability", f'{status["confidence"]:.1%}')
                    st.caption(
                        "This is the model's current best guess, not a confirmed "
                        "recognition. Keep the sign steady for a more stable result."
                    )
                else:
                    st.markdown(
                        "<div style='font-family:Manrope,sans-serif;"
                        "font-size:1.8rem;font-weight:800;color:#64748b;"
                        "padding:.35rem 0;'>Listening…</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(status["message"])

                st.progress(
                    min(status["frames_collected"] / SEQUENCE_LENGTH, 1.0),
                    text=f'Collecting sequence · {status["frames_collected"]}/{SEQUENCE_LENGTH} frames',
                )

                with st.expander("See model probabilities"):
                    chart_data = {
                        ACTIONS[i]: float(status["probabilities"][i])
                        for i in range(len(ACTIONS))
                    }
                    st.bar_chart(chart_data, y_label="Probability")

            show_live_status(context)

        st.markdown(
            """
            <div class="soft-panel" style="margin-top:.85rem;">
              <div class="section-kicker">For a better result</div>
              <p style="margin:.25rem 0;color:#526176;font-size:.88rem;line-height:1.65;">
                ① Face the camera with your upper body visible.<br>
                ② Keep the signing hand inside the frame.<br>
                ③ Use even lighting and avoid a busy background.<br>
                ④ Hold one sign steadily for 2–3 seconds.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif active_view == "Upload a clip":
    st.markdown(
        """
        <div class="section-kicker">Video analysis</div>
        <div class="section-heading">Test a recorded sign</div>
        <div class="section-subtitle">
          Prefer not to use your webcam? Upload a short clip showing one
          sign clearly and analyze it here.
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_left, upload_right = st.columns([1.05, 0.95], gap="large")
    with upload_left:
        with st.container(border=True):
            uploaded_video = st.file_uploader(
                "Choose a video file",
                type=["mp4", "mov", "avi", "mkv"],
                help="A 2–5 second clip with your face, upper body and hand visible is recommended.",
                disabled=model is None,
            )
            if uploaded_video is not None:
                st.video(uploaded_video)
            run_analysis = st.button(
                "Analyze this video",
                type="primary",
                disabled=uploaded_video is None or model is None,
                use_container_width=True,
            )

    with upload_right:
        with st.container(border=True):
            st.markdown("**BEFORE YOU UPLOAD**")
            st.markdown(
                """
                - Record one sign per clip.
                - Keep your face and hand inside the frame.
                - Avoid fast camera movement.
                - Use at least 30 frames; 2–5 seconds is recommended.
                """
            )
            st.caption("Supported formats: MP4, MOV, AVI and MKV.")

    if run_analysis and uploaded_video is not None:
        progress = st.progress(0, text="Preparing video…")
        try:
            result = analyze_uploaded_video(
                uploaded_video.getvalue(), model, progress
            )
            st.success("Analysis complete")
            first, second, third = st.columns(3)
            first.metric("Predicted sign", result["label"])
            second.metric("Average confidence", f'{result["confidence"]:.1%}')
            third.metric("Sequences analyzed", result["windows"])
            with st.container(border=True):
                st.markdown("**MODEL PROBABILITIES**")
                st.bar_chart(
                    {
                        ACTIONS[i]: float(result["probabilities"][i])
                        for i in range(len(ACTIONS))
                    },
                    y_label="Average probability",
                )
                st.caption(
                    f'Processed {result["frames"]} frames. '
                    f'Landmarks were detected in {result["detected_frames"]} frames.'
                )
        except Exception as exc:
            progress.empty()
            st.error(f"Could not analyze this video: {exc}")

elif active_view == "Learn the signs":
    st.markdown(
        """
        <div class="section-kicker">Quick sign guide</div>
        <div class="section-heading">Learn the three signs</div>
        <div class="section-subtitle">
          Watch each short demonstration, then try the same movement in the
          Live camera workspace.
        </div>
        <div class="note-panel">
          <strong>Language note:</strong> These are standard American Sign
          Language (ASL) reference demonstrations, performed by a Deaf,
          fluent signer. ASL is not Indian Sign Language (ISL). Your model
          was trained on your own recorded examples, so its learned version
          may differ from these references. Use the signs you recorded when
          collecting your training data for the closest match.
        </div>
        """,
        unsafe_allow_html=True,
    )

    demos = {
        "Hello": {
            "meaning": "A greeting",
            "gif": "https://framerusercontent.com/images/HftfOq15g9I2o0WYKNp2dhlDQ.gif",
            "steps": "Hold your dominant hand flat, fingers together and palm facing outward. Start beside your forehead, then move the hand outward and away from your head in one smooth movement.",
            "source": "https://www.aisa.solutions/asl/signs/hello",
        },
        "Thanks": {
            "meaning": "Expressing gratitude",
            "gif": "https://framerusercontent.com/images/AoHnODN1F23v0NRbKp04RFkuVo.gif",
            "steps": "Touch the fingertips of your flat dominant hand to your lips, then move the hand forward and slightly down toward the person you are thanking. A warm expression helps.",
            "source": "https://www.aisa.solutions/asl/signs/thank-you",
        },
        "I Love You": {
            "meaning": "Expressing affection",
            "gif": "https://framerusercontent.com/images/G47vRLDlF9mb4epfbzYB3nsyZM.gif",
            "steps": "Raise your dominant hand near shoulder height with the palm facing outward. Extend your thumb, index finger and little finger; fold the middle and ring fingers down. Hold the handshape still.",
            "source": "https://www.aisa.solutions/asl/signs/i-love-you",
        },
    }

    selected_sign = st.selectbox(
        "Choose a sign to watch",
        options=list(demos.keys()),
        key="learn_sign_selector",
    )
    demo = demos[selected_sign]

    demo_image_col, demo_steps_col = st.columns([1, 1], gap="large")
    with demo_image_col:
        with st.container(border=True):
            st.markdown(f"**{selected_sign.upper()} · ANIMATED DEMONSTRATION**")
            st.markdown(
                f"""
                <div class="sign-gif-wrap" style="min-height:300px;">
                  <img
                    src="{demo['gif']}"
                    alt="Animated ASL demonstration for {selected_sign}"
                    loading="eager"
                    decoding="async"
                    referrerpolicy="no-referrer"
                  >
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Only the selected animation is loaded. Choose another sign "
                "above to switch demonstrations."
            )

    with demo_steps_col:
        with st.container(border=True):
            st.markdown(f"### {selected_sign}")
            st.markdown(
                f"<div class='sign-meaning'>{demo['meaning']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("**How to perform the sign**")
            st.write(demo["steps"])
            st.markdown(
                f'<a class="demo-link" href="{demo["source"]}" '
                'target="_blank" rel="noopener noreferrer">'
                "Open the full demonstration ↗</a>",
                unsafe_allow_html=True,
            )
            st.divider()
            st.markdown("**Practice it**")
            st.write(
                "Watch the movement once, copy it slowly, then repeat it "
                "naturally while keeping your hand and face visible."
            )

    st.markdown(
        """
        <div class="footer-note">
          Animated sign references: AISA School · ASL demonstrations by a Deaf,
          fluent signer. ASL is different from Indian Sign Language (ISL).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(" ")
    st.markdown("**Practice sequence**")
    st.markdown(
        "Choose one sign → watch the animation → copy the movement → open "
        "Live camera → hold the sign steadily → compare the prediction with "
        "the sign you intended."
    )

else:
    st.markdown(
        """
        <div class="section-kicker">Under the hood</div>
        <div class="section-heading">How this recognition system works</div>
        <div class="section-subtitle">
          This project combines landmark-based computer vision with a
          recurrent neural network for short motion sequences.
        </div>
        """,
        unsafe_allow_html=True,
    )
    info_left, info_right = st.columns(2, gap="large")
    with info_left:
        with st.container(border=True):
            st.markdown("**THE INFERENCE PIPELINE**")
            st.markdown(
                """
                1. MediaPipe Holistic detects pose, face and hand landmarks.
                2. Each frame is converted into 1,662 numerical features.
                3. A rolling window of 30 frames forms one input sequence.
                4. The trained stacked-LSTM model predicts one of three classes.
                """
            )
    with info_right:
        with st.container(border=True):
            st.markdown("**MODEL SPECIFICATION**")
            st.markdown(
                """
                - Input: 30 × 1,662
                - Architecture: 3 LSTM layers + 3 Dense layers
                - Output: 3 class probabilities
                - Classes: Hello, Thanks, I Love You
                - Runtime: TensorFlow / Keras
                """
            )

    st.warning(
        "This is a prototype trained for three selected labels. It is not a "
        "general-purpose sign-language translator and should not be used as "
        "an authoritative interpreter. Accuracy can vary with signer, camera "
        "angle, lighting, background and signing speed."
    )

st.markdown(
    """
    <div class="footer-note">
      Sign Language Studio · Streamlit · TensorFlow / Keras · MediaPipe Holistic
    </div>
    """,
    unsafe_allow_html=True,
)
