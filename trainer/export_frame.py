import json
import cv2
import mediapipe as mp
from train import process_frame  # menggunakan fungsi yang sama persis dengan training

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

print("Tekan 'S' untuk menyimpan frame, 'Q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror seperti di trainer
    frame = cv2.flip(frame, 1)
    display = frame.copy()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Gambar landmark untuk visual
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                display, hand_lm, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Export Frame", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Gunakan fungsi yang sama dengan trainer
        _, _, arr = process_frame(frame, hands)
        with open("exported_frame.json", "w") as f:
            json.dump(arr.tolist(), f)
        print("✅ Array disimpan ke exported_frame.json")
        print("   Nilai 5 pertama:", arr[:5])

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()