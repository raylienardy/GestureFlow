import cv2
import numpy as np
import mediapipe as mp
from train import landmarks_to_array, process_frame  # reuse fungsi dari train
from tensorflow import keras

model = keras.models.load_model('model.h5', compile=False)
# Load labels dari labels.json
import json
with open('labels.json') as f:
    labels = json.load(f)
    idx2label = {int(k): v for k, v in labels.items()}

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
buffer = []
while True:
    ret, frame = cap.read()
    if not ret: break
    display, hand_detected, arr = process_frame(frame, hands)
    buffer.append(arr)
    if len(buffer) > 30: buffer.pop(0)
    if len(buffer) == 30:
        seq = np.array(buffer)
        # Reshape untuk LSTM [1,30,126]
        pred = model.predict(np.expand_dims(seq, axis=0), verbose=0)[0]
        cls = np.argmax(pred)
        conf = pred[cls]
        label = idx2label[cls]
        cv2.putText(display, f"{label} ({conf:.2f})", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Test", display)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()