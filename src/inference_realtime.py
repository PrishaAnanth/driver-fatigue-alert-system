import cv2
import numpy as np
import yaml
import requests
import datetime
from tensorflow.keras.models import load_model
from src.utils.alert import trigger_full_alert   # sound + web alert
from src.utils.smooth import Smoother

# ------------------------------
# Config
# ------------------------------
SERVER_URL = "http://127.0.0.1:5000/alert"  # your Flask backend
DRIVER_ID = "DL123"  # unique driver ID, can be dynamic

# ------------------------------
# Load Model
# ------------------------------
print("📂 Loading driver state model...")
model = load_model("driver_state_model.h5")

# Get input size directly from model
input_shape = model.input_shape  # e.g. (None, 64, 64, 3)
IMG_SIZE = input_shape[1] if input_shape[1] else 64
print(f"✅ Model expects input size: {IMG_SIZE}x{IMG_SIZE}")

# Load optional params
try:
    cfg = yaml.safe_load(open("config.yaml"))
except FileNotFoundError:
    cfg = {}

SMOOTH_WINDOW = cfg.get("SMOOTH_WINDOW", 5)

# ------------------------------
# Webcam Init
# ------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Webcam not accessible. Check camera permissions.")

smoother = Smoother(window=SMOOTH_WINDOW)

# Behavior labels
labels = ["Safe", "Eyes Closed", "Yawn", "Phone Use", "Eating", "Look Side"]

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("🚗 Driver Safety Monitoring Started...")
print("👉 Press 'q' to quit.")

# ------------------------------
# Main Loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # first detected face
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    else:
        face_roi = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Preprocess
    img = face_roi.astype("float32") / 255.0
    X = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(X, verbose=0)[0]

    # ------------------------------
    # Handle Binary vs Multi-class
    # ------------------------------
    if pred.shape[-1] == 1:
        p = float(pred[0])
        debug_probs = {"Drowsy": p, "Safe": 1 - p}
        if p > 0.7:
            label = "Drowsy"
            score = smoother.update(p)
        else:
            label = "Safe"
            score = smoother.update(1 - p)
    else:
        debug_probs = {labels[i]: float(pred[i]) for i in range(len(labels))}
        label_idx = np.argmax(pred)
        label = labels[label_idx]
        score = smoother.update(pred[label_idx])

    # Debug print
    print(debug_probs)

    # Overlay
    cv2.putText(frame, f"{label} ({score:.2f})",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    if len(faces) > 0:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # ------------------------------
    # Trigger alert & notify server
    # ------------------------------
    if label != "Safe" and score > 0.8:
        print(f"⚠️ Alert Triggered: {label} ({score:.2f})")
        trigger_full_alert()

        # Send alert to server
        payload = {
            "driver_id": DRIVER_ID,
            "status": label,
            "timestamp": datetime.datetime.now().isoformat()
        }
        try:
            requests.post(SERVER_URL, json=payload, timeout=2)
            print("✅ Alert sent to server")
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")

    # Show webcam feed
    cv2.imshow("Driver Safety Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ------------------------------
# Cleanup
# ------------------------------
cap.release()
cv2.destroyAllWindows()
print("🛑 Driver Safety Monitoring Stopped.")
