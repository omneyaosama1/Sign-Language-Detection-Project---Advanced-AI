import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# Settings
MODEL_PATH = "../models/best_model.h5"
IMAGE_SIZE = (128, 128)  # Must match training size
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# Load model with custom objects
model = tf.keras.models.load_model(MODEL_PATH)

# Prediction smoother
class PredictionSmoother:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)
    
    def smooth(self, pred):
        self.window.append(pred)
        return np.mean(self.window, axis=0)

smoother = PredictionSmoother(window_size=3)

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    resized = cv2.resize(frame, IMAGE_SIZE)
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)

    # Predict with smoothing
    raw_pred = model.predict(input_tensor, verbose=0)
    smoothed_pred = smoother.smooth(raw_pred)
    pred_idx = np.argmax(smoothed_pred)
    confidence = smoothed_pred[0][pred_idx] * 100

    # Display
    cv2.putText(frame,
               f"{CLASS_NAMES[pred_idx]} ({confidence:.1f}%)",
               (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX,
               1, (0, 255, 0), 2)
    cv2.imshow('ASL Recognition', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()