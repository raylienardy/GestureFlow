#!/usr/bin/env python3
"""
Konversi gambar gestur BISINDO ke dataset .npz.
Usage:
  python train_image.py <label> <folder_gambar>
Contoh:
  python train_image.py A BISINDO/A
"""

import sys
import os
import time
import numpy as np
import cv2
import mediapipe as mp

# ==================================================
# KONSTANTA (sama dengan train.py)
# ==================================================
MP_HANDS = mp.solutions.hands
MAX_HANDS = 2
PER_HAND = 21 * 3          # x, y, z per titik
FEAT_DIM = PER_HAND * MAX_HANDS  # 126
SEQ_LEN = 30               # ulangi frame yang sama 30 kali

MIN_DETECTION_CONF = 0.5
DATA_DIR = "data"


def landmarks_to_array(multi_hand_landmarks, multi_handedness=None):
    """
    Konversi hasil deteksi tangan ke array 1D (126,).
    Tangan diurutkan berdasarkan handedness (Left, Right).
    Jika handedness tidak tersedia, urutkan berdasarkan wrist.x.
    """
    arr = np.zeros(FEAT_DIM, dtype=np.float32)
    if multi_hand_landmarks is None:
        return arr

    hands = list(multi_hand_landmarks)
    if multi_handedness:
        labels = [h.classification[0].label for h in multi_handedness]
        # Left = 0, Right = 1
        hands_sorted = [hand for _, hand in sorted(zip(labels, hands), key=lambda x: 0 if x[0] == 'Left' else 1)]
    else:
        hands_sorted = sorted(hands, key=lambda h: h.landmark[0].x)

    for i, hand in enumerate(hands_sorted[:MAX_HANDS]):
        pts = []
        for lm in hand.landmark:
            pts.extend([lm.x, lm.y, lm.z])
        start = i * PER_HAND
        arr[start:start + PER_HAND] = np.array(pts, dtype=np.float32)
    return arr


def process_folder(label, folder_path):
    """Proses semua gambar di folder, simpan sebagai .npz."""
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' tidak ditemukan.")
        sys.exit(1)

    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f.lower())[1] in exts]

    if not files:
        print(f"Tidak ada gambar di folder '{folder_path}'.")
        sys.exit(1)

    print(f"Ditemukan {len(files)} gambar untuk label '{label}'.")

    # Inisialisasi MediaPipe Hands dengan static_image_mode=True untuk akurasi lebih baik pada gambar
    hands = MP_HANDS.Hands(static_image_mode=True, max_num_hands=MAX_HANDS,
                           min_detection_confidence=MIN_DETECTION_CONF)
    saved = 0
    skipped = 0

    os.makedirs(DATA_DIR, exist_ok=True)

    for fname in sorted(files):
        path = os.path.join(folder_path, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"  Lewati {fname}: tidak bisa dibaca.")
            skipped += 1
            continue

        # Mirror gambar (sama seperti trainer webcam)
        img = cv2.flip(img, 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            print(f"  Lewati {fname}: tidak ada tangan terdeteksi.")
            skipped += 1
            continue

        # Konversi landmark ke array dengan handedness
        arr = landmarks_to_array(results.multi_hand_landmarks, results.multi_handedness)

        # Simpan sebagai .npz (format sama dengan train.py)
        # Ulangi array yang sama 30 kali untuk membuat sequence statis
        sequence = np.tile(arr, (SEQ_LEN, 1))  # shape (30, 126)

        out_name = os.path.join(DATA_DIR, f"{label}_{saved+1}_{int(time.time())}.npz")
        np.savez_compressed(out_name, sequence=sequence, label=label)
        saved += 1
        print(f"  {fname} -> {os.path.basename(out_name)}")

    hands.close()
    print(f"\nSelesai: {saved} disimpan, {skipped} dilewati.")
    print("Sekarang jalankan 'python train_model.py' untuk melatih ulang model.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_image.py <label> <folder_gambar>")
        print("Contoh: python train_image.py A BISINDO/A")
        sys.exit(1)

    label = sys.argv[1]
    folder = sys.argv[2]
    process_folder(label, folder)