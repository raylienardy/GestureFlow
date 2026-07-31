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

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

        # Info tampilan
        cv2.putText(display, f"Label: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(display, f"Recorded: {recorded}/{target}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        key = cv2.waitKey(1) & 0xFF

        if state == "waiting":
            cv2.putText(display, "Tekan 'R' untuk rekam", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        elif state == "recording":
            elapsed = time.time() - state_start
            # Ambil frame dengan MediaPipe
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                # Simpan landmark mentah (21 titik * 3 koordinat) per tangan
                # Untuk satu frame: list of hands, masing-masing 21 * 3
                frame_landmarks = []
                for hand_lm in results.multi_hand_landmarks:
                    pts = []
                    for lm in hand_lm.landmark:
                        pts.extend([lm.x, lm.y, lm.z])
                    frame_landmarks.append(pts)
                frames_buffer.append(frame_landmarks)
                for hand_lm in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        display, hand_lm, mp_hands.HAND_CONNECTIONS)
            else:
                # Jika tidak terdeteksi, isi dengan list kosong
                frames_buffer.append([])

            cv2.putText(display, f"Merekam... {elapsed:.1f}s / {record_duration}s",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            if elapsed >= record_duration:
                # Simpan sequence
                # Resample jadi SEQ_LEN frame (30)
                seq_len = 30
                if len(frames_buffer) > 0:
                    # Ambil index merata
                    indices = np.linspace(0, len(frames_buffer)-1, seq_len, dtype=int)
                    sampled = [frames_buffer[i] for i in indices]
                else:
                    sampled = [[] for _ in range(seq_len)]
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
