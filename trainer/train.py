#!/usr/bin/env python3
"""
GestureFlow Trainer - Rekam gestur dinamis untuk dilatih.
Usage: python train.py <nama_gerakan> <jumlah_rekaman>
Contoh: python train.py halo 30
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Konfigurasi
MAX_HANDS = 2
PER_HAND = 21 * 3          # x, y, z per titik
FEAT_DIM = PER_HAND * MAX_HANDS  # 126

def landmarks_to_array(multi_hand_landmarks):
    """
    Konversi hasil deteksi tangan menjadi array 1D (126,).
    - Tidak ada tangan -> semua nol.
    - 1 tangan -> 63 pertama terisi, sisanya nol.
    - 2 tangan -> diurutkan berdasarkan wrist.x, digabung.
    """
    arr = np.zeros(FEAT_DIM, dtype=np.float32)
    if multi_hand_landmarks is None:
        return arr

    hands = list(multi_hand_landmarks)
    # Urutkan tangan dari kiri ke kanan (wrist.x)
    hands.sort(key=lambda h: h.landmark[0].x)

    for i, hand in enumerate(hands[:MAX_HANDS]):
        pts = []
        for lm in hand.landmark:
            pts.extend([lm.x, lm.y, lm.z])
        start = i * PER_HAND
        arr[start:start + PER_HAND] = np.array(pts, dtype=np.float32)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="Nama gerakan (misal: halo)")
    parser.add_argument("count", type=int, help="Jumlah rekaman (misal: 30)")
    args = parser.parse_args()

    label = args.label
    target = args.count
    os.makedirs("data", exist_ok=True)

    print(f"=== GestureFlow Trainer ===")
    print(f"Label: {label} | Target: {target} rekaman")
    print("Instruksi: Tekan 'R' untuk mulai rekam, 'Q' untuk keluar.")
    print("Setelah countdown 3-2-1, lakukan gerakan selama 2 detik.")
    print("=================================")

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera.")
        sys.exit(1)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    recorded = 0
    state = "waiting"   # waiting / countdown / recording
    state_start = 0
    countdown_seconds = 3
    record_duration = 2.0
    frames_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # Deteksi tangan setiap frame untuk feedback visual
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        hand_detected = results.multi_hand_landmarks is not None

        # Gambar landmark jika tangan terdeteksi
        if hand_detected:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        # Info tampilan
        cv2.putText(display, f"Label: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, f"Recorded: {recorded}/{target}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        key = cv2.waitKey(1) & 0xFF

        if state == "waiting":
            cv2.putText(display, "Tekan 'R' untuk rekam", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if key == ord('r'):
                state = "countdown"
                state_start = time.time()
                print(f"  [{recorded+1}/{target}] Memulai countdown...")

        elif state == "countdown":
            elapsed = time.time() - state_start
            remaining = countdown_seconds - int(elapsed)
            if remaining <= 0:
                # Mulai rekam
                state = "recording"
                state_start = time.time()
                frames_buffer = []
                print("  GO! Rekam selama 2 detik...")
            else:
                cv2.putText(display, f"Countdown: {remaining}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        elif state == "recording":
            elapsed = time.time() - state_start
            # Ambil data landmark (selalu array 126, homogen)
            landmark_array = landmarks_to_array(results.multi_hand_landmarks)
            frames_buffer.append(landmark_array)

            cv2.putText(display, f"Merekam... {elapsed:.1f}s / {record_duration}s",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if elapsed >= record_duration:
                # Resample jadi SEQ_LEN frame (30)
                seq_len = 30
                if len(frames_buffer) > 0:
                    all_frames = np.array(frames_buffer)          # (n_frames, 126)
                    indices = np.linspace(0, all_frames.shape[0] - 1, seq_len, dtype=int)
                    sampled = all_frames[indices]                 # (30, 126)
                else:
                    sampled = np.zeros((seq_len, FEAT_DIM), dtype=np.float32)

                # Simpan ke .npz
                fname = f"data/{label}_{recorded+1}_{int(time.time())}.npz"
                np.savez_compressed(fname, sequence=sampled, label=label)
                recorded += 1
                print(f"  Tersimpan: {fname} ({recorded}/{target})")
                state = "waiting"

                if recorded >= target:
                    print(f"=== Selesai. {recorded} sampel tersimpan ===")
                    break

        cv2.imshow("GestureFlow Trainer", display)
        if key == ord('q'):
            print("Dihentikan oleh pengguna.")
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()