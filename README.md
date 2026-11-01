# Sign Language Detection

A real-time sign recognition prototype built with **MediaPipe Holistic**, **TensorFlow/Keras**, and **Streamlit**. The application recognizes three trained signs:

- Hello
- Thanks
- I Love You

It supports live webcam inference and uploaded-video inference.

## Streamlit Community Cloud deployment

Open [Streamlit Community Cloud](https://share.streamlit.io/) and create an app with:

- Repository: **CodeWithRjUdit/Sign_Language_Detection**
- Branch: **main**
- Main file: **app.py**
- Python version in Advanced settings: **3.11**

## Run locally

Use Python 3.11. Create a virtual environment, activate it, and install requirements.txt. The deployment pins TensorFlow 2.16.1 with Keras 3.0.5 because the training notebook's saved model was produced with the Keras 3 API.

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
1. Open the Live camera tab.
2. Click Start and grant camera permission in your browser.
3. Keep your upper body, face, and hands visible.
4. Perform one of the three trained signs and hold it steadily for around 2–3 seconds.
5. Read the predicted sign and model confidence below the video.

### Upload a video
1. Open the Upload a video tab.
2. Upload a short MP4, MOV, AVI, or MKV clip.
3. Use a 2–5 second clip with one sign clearly visible.
4. Click Analyze video.

## Repository contents

- app.py — Streamlit application and inference pipeline
- my_model.keras — trained LSTM model used by the application
- actions.h5 — retained original artifact; not required by the deployed inference app
- Sign_Language_Detection.ipynb — original data preparation, training, and experimentation notebook
- Final_Model.ipynb — final inference notebook

## Live camera performance

The live camera preview is constrained to a compact display (maximum 520 px wide). The browser is asked for a 640 × 480 camera stream at approximately 15 FPS (up to 20 FPS). Before landmark extraction, frames are resized so their longest side is at most 640 px. MediaPipe Holistic uses pose model complexity 0 to reduce CPU load, and the processor drops excess incoming frames rather than allowing a processing backlog to build up. These settings improve responsiveness on CPU-based hosting; actual frame rate still depends on available server resources and network conditions.

## Limitations

This prototype recognizes only the three classes represented in its training data. It is not a general-purpose sign-language translation system. The original dataset has limited signer diversity, so accuracy may vary for people whose signing style differs from the training examples. Lighting, camera angle, hand visibility, and signing speed can also affect predictions.

## Deployment notes

- Select Python 3.11 in the Community Cloud Advanced settings.
- TensorFlow 2.16.1 uses Keras 3, matching the Keras 3 model-save format indicated by the original training notebook. Do not downgrade TensorFlow to 2.15.x for this saved model.
- Streamlit Community Cloud uses the Python version selected in its deployment settings; runtime.txt is not used to select the version.
- The browser must be allowed to access the camera for live inference.
- Uploaded video is processed by the app. No API key is required.
