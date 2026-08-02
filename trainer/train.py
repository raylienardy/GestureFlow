#!/usr/bin/env python3
"""
GestureFlow Trainer - Rekam gestur dinamis untuk dilatih.
Usage: python train.py <nama_gerakan> <jumlah_rekaman>
Contoh: python train.py halo 30

Fitur:
  R  -> Rekam satu sampel (manual)
  F  -> Auto Record ON/OFF (rekam terus menerus tanpa countdown setelah pertama)
  Q  -> Keluar (hapus semua rekaman yang belum selesai jika auto)
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

MAX_HANDS = 2
PER_HAND = 21 * 3
FEAT_DIM = PER_HAND * MAX_HANDS  # 126

def landmarks_to_array(multi_hand_landmarks):
    arr = np.zeros(FEAT_DIM, dtype=np.float32)
    if multi_hand_landmarks is None:
        return arr
    hands = list(multi_hand_landmarks)
    hands.sort(key=lambda h: h.landmark[0].x)
    for i, hand in enumerate(hands[:MAX_HANDS]):
        pts = []
        for lm in hand.landmark:
            pts.extend([lm.x, lm.y, lm.z])
        start = i * PER_HAND
        arr[start:start + PER_HAND] = np.array(pts, dtype=np.float32)
    return arr


def draw_text_with_background(img, text, position, font_scale=0.7,
                              fg_color=(255, 255, 255), bg_color=(0, 0, 0),
                              thickness=2):
    """Gambar teks dengan background hitam agar mudah dibaca di segala kondisi."""
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x, y - h - baseline), (x + w, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, fg_color, thickness, cv2.LINE_AA)


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
    print("Instruksi:")
    print("  R  -> Rekam manual satu sampel")
    print("  F  -> Auto Record ON/OFF (rekam terus menerus)")
    print("  Q  -> Keluar (hapus semua file yang direkam di sesi ini)")
    print("=================================")

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
    state = "waiting"
    state_start = 0
    countdown_seconds = 3
    record_duration = 2.0
    frames_buffer = []
    saved = False
    auto_mode = False
    session_files = []          # lacak file yang dibuat di sesi auto

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # Deteksi tangan
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        hand_detected = results.multi_hand_landmarks is not None

        # Gambar landmark
        if hand_detected:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        # Info tampilan
        draw_text_with_background(display, f"Label: {label}", (10, 30))
        draw_text_with_background(display, f"Recorded: {recorded}/{target}", (10, 65))

        mode_color = (0, 255, 0) if auto_mode else (0, 0, 255)
        draw_text_with_background(display, f"Auto: {'ON' if auto_mode else 'OFF'}",
                                  (display.shape[1] - 200, 30), fg_color=mode_color)

        key = cv2.waitKey(1) & 0xFF

        # Toggle auto mode
        if key == ord('f'):
            auto_mode = not auto_mode
            if auto_mode:
                print("  Auto Record: ON (rekam otomatis)")
                # Hanya hitung mundur saat pertama kali
                if state == "waiting":
                    state = "countdown"
                    state_start = time.time()
                    saved = False
                    session_files = []
                    print(f"  [{recorded+1}/{target}] Auto memulai countdown...")
            else:
                print("  Auto Record: OFF")

        # Keluar & hapus file yang dibuat di sesi ini
        if key == ord('q'):
            print("Dihentikan oleh pengguna.")
            if session_files:
                print("  Menghapus file yang baru direkam...")
                for f in session_files:
                    if os.path.exists(f):
                        os.remove(f)
                        print(f"    Dihapus: {f}")
            break

        if state == "waiting":
            draw_text_with_background(display, "Tekan 'R' (manual) atau 'F' (auto)", (10, 100))
            if key == ord('r'):
                state = "countdown"
                state_start = time.time()
                saved = False
                session_files = []   # manual tidak lacak, kecuali mau
                print(f"  [{recorded+1}/{target}] Memulai countdown...")

        elif state == "countdown":
            elapsed = time.time() - state_start
            remaining = countdown_seconds - int(elapsed)
            if remaining <= 0:
                state = "recording"
                state_start = time.time()
                frames_buffer = []
                print("  GO! Rekam selama 2 detik...")
            else:
                draw_text_with_background(display, f"Countdown: {remaining}", (10, 100),
                                          fg_color=(0, 0, 255))

        elif state == "recording":
            elapsed = time.time() - state_start
            landmark_array = landmarks_to_array(results.multi_hand_landmarks)
            frames_buffer.append(landmark_array)

            draw_text_with_background(display, f"Merekam... {elapsed:.1f}s / {record_duration}s",
                                      (10, 100), fg_color=(0, 255, 255))

            if elapsed >= record_duration and not saved:
                saved = True
                seq_len = 30
                if len(frames_buffer) > 0:
                    all_frames = np.array(frames_buffer)
                    indices = np.linspace(0, all_frames.shape[0]-1, seq_len, dtype=int)
                    sampled = all_frames[indices]
                else:
                    sampled = np.zeros((seq_len, FEAT_DIM), dtype=np.float32)

                fname = f"data/{label}_{recorded+1}_{int(time.time())}.npz"
                np.savez_compressed(fname, sequence=sampled, label=label)
                session_files.append(fname)
                recorded += 1
                print(f"  Tersimpan: {fname} ({recorded}/{target})")

                if recorded >= target:
                    print(f"=== Selesai. {recorded} sampel tersimpan ===")
                    break

                if auto_mode:
                    # Langsung rekam ulang tanpa countdown
                    state = "recording"
                    state_start = time.time()
                    saved = False
                    frames_buffer = []
                    print(f"  [{recorded+1}/{target}] Auto melanjutkan rekam...")
                else:
                    state = "waiting"

        cv2.imshow("GestureFlow Trainer", display)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()