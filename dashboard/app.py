# app.py
import os
import time
import threading
import datetime

import cv2
import numpy as np
import mediapipe as mp
import requests
import streamlit as st
from av import VideoFrame
from flask import Flask, request, jsonify
from flask_cors import CORS
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --------------------------- CONFIG ---------------------------
DRIVER_ID = "DL123"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "driver_state_model.h5")

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
SERVER_URL = f"http://localhost:{SERVER_PORT}/alert"

MAR_THRESHOLD = 0.75
YAWN_CONSEC_FRAMES = 10
CNN_DROWSY_THRESHOLD = 0.5
CNN_RATE_LIMIT_SEC = 5
EAR_THRESHOLD = 0.22
EAR_CONSEC_FRAMES = 12
FRAME_RESIZE = (640, 480)

# --------------------------- FLASK ALERT SERVER ---------------------------
app = Flask(__name__)
CORS(app)

alerts_log = []
current_alert = {"status": None, "driver_id": None, "timestamp": None}


@app.route("/alert", methods=["POST"])
def receive_alert():
    data = request.json
    driver_id = data.get("driver_id")
    status = data.get("status")
    timestamp = data.get("timestamp") or datetime.datetime.now().isoformat()
    if not driver_id or not status:
        return jsonify({"error": "driver_id or status missing"}), 400

    alert_msg = {
        "driver_id": driver_id,
        "status": status,
        "timestamp": timestamp,
        "message": f"⚠️ Driver {driver_id} is {status} at {datetime.datetime.now().strftime('%H:%M:%S')}"
    }

    alerts_log.append(alert_msg)
    if len(alerts_log) > 50:
        alerts_log.pop(0)

    current_alert.update({"status": status, "driver_id": driver_id, "timestamp": timestamp})
    print(alert_msg["message"])
    return jsonify({"success": True, "msg": alert_msg["message"]})


@app.route("/alerts", methods=["GET"])
def get_alerts():
    return jsonify({"alerts": alerts_log})


@app.route("/latest_alert", methods=["GET"])
def get_latest_alert():
    if alerts_log:
        return jsonify(alerts_log[-1])
    else:
        return jsonify({"driver_id": None, "status": "Alert", "timestamp": None, "message": ""})


def run_flask():
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, use_reloader=False)


# --------------------------- LOAD MODEL ---------------------------
from tensorflow.keras.models import load_model

try:
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("✅ CNN model loaded:", MODEL_PATH)
except Exception as e:
    print("⚠️ Could not load model:", e)
    model = None

# --------------------------- MEDIAPIPE & HELPERS ---------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_face_det = mp.solutions.face_detection
face_detector = mp_face_det.FaceDetection(min_detection_confidence=0.5)

LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [263, 387, 385, 362, 380, 373]
UPPER_LIP = [13]
LOWER_LIP = [14]
LEFT_MOUTH_CORNER = 78
RIGHT_MOUTH_CORNER = 308


def eye_aspect_ratio(landmarks, eye_idxs):
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_idxs])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return 0.0 if C == 0 else (A + B) / (2.0 * C)


def mouth_aspect_ratio(landmarks):
    top = np.array([landmarks[UPPER_LIP[0]].x, landmarks[UPPER_LIP[0]].y])
    bottom = np.array([landmarks[LOWER_LIP[0]].x, landmarks[LOWER_LIP[0]].y])
    left = np.array([landmarks[LEFT_MOUTH_CORNER].x, landmarks[LEFT_MOUTH_CORNER].y])
    right = np.array([landmarks[RIGHT_MOUTH_CORNER].x, landmarks[RIGHT_MOUTH_CORNER].y])
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return 0.0 if horizontal == 0 else vertical / horizontal


def send_alert_to_server_local(status):
    payload = {"driver_id": DRIVER_ID, "status": status, "timestamp": datetime.datetime.now().isoformat()}
    try:
        requests.post(SERVER_URL, json=payload, timeout=2)
    except Exception:
        pass


# --------------------------- VIDEO TRANSFORMER ---------------------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.mar_buffer = []
        self.counter_yawn = 0
        self.counter_ear = 0
        self.last_sent_time = 0

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = cv2.resize(frame.to_ndarray(format="bgr24"), FRAME_RESIZE)
        ih, iw, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        det = face_detector.process(rgb)
        mesh = face_mesh.process(rgb)

        cnn_says_drowsy = False
        yawning = False
        ear_closed = False

        # --- CNN-based drowsiness ---
        if model and det.detections:
            d = det.detections[0]
            bboxC = d.location_data.relative_bounding_box
            x, y, w, h = int(max(0, bboxC.xmin * iw)), int(max(0, bboxC.ymin * ih)), int(max(1, bboxC.width * iw)), int(max(1, bboxC.height * ih))
            x2, y2 = min(iw, x + w), min(ih, y + h)
            face_roi = img[y:y2, x:x2]
            if face_roi.size != 0:
                face_resized = cv2.resize(face_roi, (64, 64))
                face_norm = face_resized.astype("float32") / 255.0
                face_in = np.expand_dims(face_norm, axis=0)
                try:
                    pred = model.predict(face_in, verbose=0)[0][0]
                except Exception:
                    pred = 0.0
                if pred > CNN_DROWSY_THRESHOLD:
                    cnn_says_drowsy = True
                color = (0, 0, 255) if cnn_says_drowsy else (0, 255, 0)
                cv2.rectangle(img, (x, y), (x2, y2), color, 2)
                cv2.putText(img, f"CNN:{pred:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- EAR & MAR ---
        if mesh.multi_face_landmarks:
            landmarks = mesh.multi_face_landmarks[0].landmark
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_LANDMARKS)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_LANDMARKS)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(landmarks)
            self.mar_buffer.append(mar)
            if len(self.mar_buffer) > 5:
                self.mar_buffer.pop(0)
            avg_mar = float(np.mean(self.mar_buffer))
            cv2.putText(img, f"EAR:{ear:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
            cv2.putText(img, f"MAR:{avg_mar:.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2)

            if avg_mar > MAR_THRESHOLD:
                self.counter_yawn += 1
            else:
                self.counter_yawn = 0
            if self.counter_yawn >= YAWN_CONSEC_FRAMES:
                yawning = True

            if ear < EAR_THRESHOLD:
                self.counter_ear += 1
            else:
                self.counter_ear = 0
            ear_closed = self.counter_ear >= EAR_CONSEC_FRAMES

        # --- Decide final status ---
        now = time.time()
        if yawning:
            new_status = "Drowsy - Yawning"
        elif cnn_says_drowsy or ear_closed:
            new_status = "Drowsy - Eyes Closed"
        else:
            new_status = "Alert"

        # --- Update alert and send to server (rate-limited) ---
        if new_status != current_alert.get("status") or (now - self.last_sent_time > CNN_RATE_LIMIT_SEC):
            current_alert.update({"status": new_status, "driver_id": DRIVER_ID,
                                  "timestamp": datetime.datetime.now().strftime("%H:%M:%S")})
            send_alert_to_server_local(new_status)
            self.last_sent_time = now

        # overlay text
        if new_status == "Alert":
            cv2.putText(img, "Driver Normal", (30, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
        else:
            cv2.putText(img, f"⚠️ {new_status}", (30, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        return VideoFrame.from_ndarray(img, format="bgr24")


# --------------------------- STREAMLIT UI ---------------------------
st.set_page_config(page_title="Driver Safety Monitor", page_icon="🚗")
st.title("Smart Driver Fatigue Alert System")

st.markdown("""
**How it works**
- CNN model classifies face crops as *drowsy* vs *alert*.
- Mediapipe FaceMesh computes MAR for yawning; EAR used as backup for eye-closure.
- Alerts are posted to internal Flask `/alert` endpoint and shown persistently until cleared.
""")

col1, col2 = st.columns([2, 1])
with col1:
    webrtc_streamer(key="driver-monitor", video_transformer_factory=VideoTransformer)

with col2:
    alert_area = st.empty()

    def ui_alert_loop():
        while True:
            st_status = current_alert.get("status")
            if st_status == "Alert":
                alert_area.success(f"✅ Driver normal at {current_alert.get('timestamp')}")
            elif st_status:
                alert_area.warning(f"⚠️ Driver {DRIVER_ID} is {st_status} at {current_alert.get('timestamp')}")
            time.sleep(0.3)

    threading.Thread(target=ui_alert_loop, daemon=True).start()

# start flask alert server in background
threading.Thread(target=run_flask, daemon=True).start()
st.info(f"Internal Flask alert server running on http://localhost:{SERVER_PORT}")
