# Sign Language Detection

A real-time sign recognition prototype built with **MediaPipe Holistic**, **TensorFlow/Keras**, and **Streamlit**. The application recognizes three trained signs:

- Hello
- Thanks
- I Love You

It supports live webcam inference, uploaded-video inference, and a Learn the Signs guide with animated demonstrations for the three labels. The guide opens first without loading TensorFlow or MediaPipe; the recognition engine is loaded only when the user explicitly starts a recognition mode.

## Streamlit Community Cloud deployment

Open [Streamlit Community Cloud](https://share.streamlit.io/) and create an app with:

- Repository: **CodeWithRjUdit/Sign_Language_Detection**
- Branch: **main**
- Main file: **app.py**
- Python version in Advanced settings: **3.11**

## Run locally

Use Python 3.11. Create a virtual environment, activate it, and install requirements.txt. The deployment pins TensorFlow 2.16.1 with Keras 3.0.5 because the training notebook's saved model was produced with the Keras 3 API. Streamlit is pinned to 1.66.1.

Windows activation: .venv\Scripts\activate  
macOS/Linux activation: source .venv/bin/activate

Run the application with: streamlit run app.py

The trained model file **my_model.keras** must remain in the repository root beside app.py.

## How it works

1. MediaPipe Holistic detects pose, face, and hand landmarks in each video frame.
2. Each frame becomes a 1,662-element feature vector:
   - Pose: 33 landmarks × 4 values (x, y, z, visibility) = 132
   - Face: 468 landmarks × 3 values = 1,404
   - Left hand: 21 landmarks × 3 values = 63
   - Right hand: 21 landmarks × 3 values = 63
3. A rolling window of 30 frames is passed to the trained LSTM model.
4. The model returns probabilities for Hello, Thanks, and I Love You.
5. Live predictions are smoothed across recent windows.

**Model input shape:** (batch, 30, 1662)  
**Model output classes:** Hello, Thanks, I Love You

The missing-pose fallback is set to 132 values (33 × 4), keeping every extracted frame at exactly 1,662 features.

## How to use the app

### Live camera
1. Choose the **Live camera** workspace.
2. Click Start and grant camera permission in your browser.
3. Keep your upper body, face, and hands visible.
4. Perform one of the three trained signs and hold it steadily for around 2–3 seconds.
5. Read the confirmed sign when the model is stable. When confidence is lower, the app also shows its current top candidate and probability without presenting it as a confirmed result.

### Upload a video
1. Choose the Upload a clip workspace.
2. Click **Load AI recognition engine** if it is not already loaded.
3. Upload a short MP4, MOV, AVI, or MKV clip (maximum upload size: 25 MB).
4. Use a 2–5 second clip with one sign clearly visible.
5. Click Analyze this video.

Use either live camera or uploaded-video inference at a time. Choose **Unload AI engine** after testing to clear the cached model. TensorFlow itself remains imported for the lifetime of the server process, so reboot the app from Manage app if its memory remains high after testing.

### Learn the signs
Choose the **Learn the signs** workspace and select Hello, Thanks, or I Love You. The app loads one animated reference at a time to reduce browser load, alongside its gesture description and source link.

**Language note:** The embedded reference demonstrations are for American Sign Language (ASL). ASL is not Indian Sign Language (ISL), and sign languages vary by region and community. The model was trained on the project's own recorded examples; the reference demonstrations are learning aids and are not guaranteed to match the exact gestures in that training set.

Animated reference sources:
- [Hello — AISA School](https://www.aisa.solutions/asl/signs/hello)
- [Thank you — AISA School](https://www.aisa.solutions/asl/signs/thank-you)
- [I love you — AISA School](https://www.aisa.solutions/asl/signs/i-love-you)


## Repository contents

- app.py — Streamlit application and inference pipeline
- my_model.keras — trained LSTM model used by the application
- actions.h5 — retained original artifact; not required by the deployed inference app
- Sign_Language_Detection.ipynb — original data preparation, training, and experimentation notebook
- Final_Model.ipynb — final inference notebook

## Live camera performance

The live camera preview is constrained to a compact display. The browser is asked for a 480 × 360 camera stream at approximately 8 FPS (up to 10 FPS). Before landmark extraction, frames are resized so their longest side is at most 480 px. MediaPipe Holistic uses pose model complexity 1 (the bundled full model) to avoid a runtime model download into the read-only package directory, and the processor drops excess incoming frames rather than allowing a processing backlog to build up. These settings improve responsiveness on CPU-based hosting; actual frame rate still depends on available server resources and network conditions.

## Limitations

This prototype recognizes only the three classes represented in its training data. It is not a general-purpose sign-language translation system. The original dataset has limited signer diversity, so accuracy may vary for people whose signing style differs from the training examples. Lighting, camera angle, hand visibility, and signing speed can also affect predictions.

## Deployment notes

- Select Python 3.11 in the Community Cloud Advanced settings.
- The app opens on the lightweight Learn the signs workspace. TensorFlow and MediaPipe are imported only after the AI recognition engine is requested.
- Only the currently selected workspace is rendered; do not run live camera and uploaded-video inference simultaneously.
- The webcam is intentionally configured for a lower-resolution, lower-FPS stream to reduce CPU and memory pressure.
- TensorFlow 2.16.1 uses Keras 3, matching the Keras 3 model-save format indicated by the original training notebook. Do not downgrade TensorFlow to 2.15.x for this saved model.
- Streamlit Community Cloud uses the Python version selected in its deployment settings; runtime.txt is not used to select the version.
- The browser must be allowed to access the camera for live inference.
- Uploaded video is processed by the app. No API key is required.
