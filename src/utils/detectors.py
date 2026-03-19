import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Indexes for eyes and mouth landmarks (based on MediaPipe's 468-point model)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
OUTER_LIPS = [61, 291, 81, 311, 78, 308, 14, 13, 82, 312]

# Frame buffer for stability (avoid single-frame false detections)
EAR_BUFFER = deque(maxlen=10)
MAR_BUFFER = deque(maxlen=10)

# Parameters
EYE_AR_THRESH = 0.25       # Eye aspect ratio threshold (lower = eyes closed)
MOUTH_AR_THRESH = 0.7      # Mouth aspect ratio threshold (higher = yawn)
EYE_AR_CONSEC_FRAMES = 15  # Frames to confirm drowsiness
MOUTH_AR_CONSEC_FRAMES = 15  # Frames to confirm yawn

# Counters
eye_frame_counter = 0
mouth_frame_counter = 0

def euclidean_distance(point1, point2):
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def eye_aspect_ratio(landmarks, eye_points):
    """Compute Eye Aspect Ratio (EAR)."""
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_points]
    return (euclidean_distance(p2, p6) + euclidean_distance(p3, p5)) / (2.0 * euclidean_distance(p1, p4))

def mouth_aspect_ratio(landmarks):
    """Compute Mouth Aspect Ratio (MAR)."""
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    left_lip = landmarks[78]
    right_lip = landmarks[308]
    return euclidean_distance(top_lip, bottom_lip) / euclidean_distance(left_lip, right_lip)

def detect_face(frame):
    """Detect face landmarks and return MediaPipe results."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    return results

def analyze_driver_state(frame):
    """
    Detects if the driver is drowsy or yawning based on facial landmarks.
    Returns:
        state (dict): {"drowsy": bool, "yawning": bool, "EAR": float, "MAR": float}
    """
    global eye_frame_counter, mouth_frame_counter

    results = detect_face(frame)
    h, w = frame.shape[:2]
    drowsy = False
    yawning = False
    EAR = None
    MAR = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert normalized coordinates to pixel positions
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # Compute EAR & MAR
            EAR = (eye_aspect_ratio(landmarks, LEFT_EYE) + eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2.0
            MAR = mouth_aspect_ratio(landmarks)

            EAR_BUFFER.append(EAR)
            MAR_BUFFER.append(MAR)

            avg_EAR = np.mean(EAR_BUFFER)
            avg_MAR = np.mean(MAR_BUFFER)

            # Drowsiness Detection
            if avg_EAR < EYE_AR_THRESH:
                eye_frame_counter += 1
            else:
                eye_frame_counter = 0

            if eye_frame_counter >= EYE_AR_CONSEC_FRAMES:
                drowsy = True
                eye_frame_counter = 0  # reset after detection

            # Yawning Detection (more stable)
            if avg_MAR > MOUTH_AR_THRESH:
                mouth_frame_counter += 1
            else:
                mouth_frame_counter = 0

            if mouth_frame_counter >= MOUTH_AR_CONSEC_FRAMES:
                yawning = True
                mouth_frame_counter = 0  # reset after detection

    return {
        "drowsy": drowsy,
        "yawning": yawning,
        "EAR": round(EAR, 3) if EAR else None,
        "MAR": round(MAR, 3) if MAR else None
    }

def draw_overlays(frame, state):
    """Overlay EAR, MAR, and state info on the frame."""
    color = (0, 255, 0)
    if state["drowsy"]:
        color = (0, 0, 255)
        cv2.putText(frame, "DROWSINESS DETECTED!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    elif state["yawning"]:
        color = (0, 255, 255)
        cv2.putText(frame, "YAWNING DETECTED!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.putText(frame, f"EAR: {state['EAR']}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"MAR: {state['MAR']}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame
