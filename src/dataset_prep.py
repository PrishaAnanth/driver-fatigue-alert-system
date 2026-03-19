import os, cv2, numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import yaml

cfg = yaml.safe_load(open("config.yaml"))
IMG_SIZE, SEQ_LEN = cfg["IMAGE_SIZE"], cfg["SEQ_LEN"]

def prepare_frames(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for label in os.listdir(input_dir):
        path = os.path.join(input_dir, label)
        for vid in os.listdir(path):
            cap = cv2.VideoCapture(os.path.join(path, vid))
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(f"{output_dir}/{label}_{count}.jpg", frame)
                count += 1
            cap.release()

if __name__ == "__main__":
    prepare_frames("data/raw", "data/frames")
