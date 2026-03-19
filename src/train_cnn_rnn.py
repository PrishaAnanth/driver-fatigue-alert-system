import numpy as np, yaml
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

cfg = yaml.safe_load(open("config.yaml"))
IMG_SIZE, SEQ_LEN, EPOCHS, LR = cfg["IMAGE_SIZE"], cfg["SEQ_LEN"], cfg["EPOCHS"], cfg["LEARNING_RATE"]

def build_model(num_classes):
    base_cnn = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    model = Sequential([
        TimeDistributed(base_cnn, input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)),
        TimeDistributed(GlobalAveragePooling2D()),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(LR), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Dummy call
if __name__ == "__main__":
    model = build_model(num_classes=6)
    model.summary()
