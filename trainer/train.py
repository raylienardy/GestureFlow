#!/usr/bin/env python3
"""
GestureFlow Trainer - Rekam gestur dinamis untuk dilatih.
Usage: python train.py <nama_gerakan> <jumlah_rekaman>
Contoh: python train.py halo 30

Fitur:
  R  -> Rekam manual satu sampel
  F  -> Auto Record ON/OFF
  Q  -> Keluar (hapus semua file yang direkam di sesi ini)
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2
import mediapipe as mp

# ==================================================
# KONSTANTA (semua magic number dikumpulkan di sini)
# ==================================================
MP_HANDS = mp.solutions.hands
MP_DRAWING = mp.solutions.drawing_utils
MP_STYLES = mp.solutions.drawing_styles

MAX_HANDS = 2
PER_HAND = 21 * 3          # x, y, z per titik
FEAT_DIM = PER_HAND * MAX_HANDS  # 126

SEQ_LEN = 30               # frame yang diresample
COUNTDOWN_SECONDS = 3
RECORD_DURATION = 2.0      # detik
AUTO_DELAY = 1.0           # jeda antar rekaman di mode auto

MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.5

DATA_DIR = "data"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 2


def landmarks_to_array(multi_hand_landmarks, multi_handedness=None):
    arr = np.zeros(FEAT_DIM, dtype=np.float32)
    if multi_hand_landmarks is None:
        return arr
    hands = list(multi_hand_landmarks)
    # Gunakan handedness untuk urutan: Left di indeks 0, Right di indeks 1
    if multi_handedness:
        labels = [h.classification[0].label for h in multi_handedness]
        # Left = 0, Right = 1
        hands_sorted = [hand for _, hand in sorted(zip(labels, hands), key=lambda x: 0 if x[0]=='Left' else 1)]
    else:
        hands_sorted = sorted(hands, key=lambda h: h.landmark[0].x)
    for i, hand in enumerate(hands_sorted[:MAX_HANDS]):
        pts = []
        for lm in hand.landmark:
            pts.extend([lm.x, lm.y, lm.z])
        start = i * PER_HAND
        arr[start:start+PER_HAND] = np.array(pts, dtype=np.float32)
    return arr


def draw_text_with_background(img, text, position, fg_color=(255,255,255), bg_color=(0,0,0)):
    """Teks dengan latar hitam agar terbaca di segala kondisi."""
    (w, h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    x, y = position
    cv2.rectangle(img, (x, y - h - baseline), (x + w, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, fg_color, THICKNESS, cv2.LINE_AA)


def draw_progress_bar(img, x, y, width, height, percent, fg_color=(0,255,0)):
    """Progress bar horizontal sederhana."""
    cv2.rectangle(img, (x, y), (x + width, y + height), (200,200,200), -1)
    fill_width = int(width * percent / 100)
    cv2.rectangle(img, (x, y), (x + fill_width, y + height), fg_color, -1)


def init_camera():
    """Inisialisasi webcam dan MediaPipe Hands."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera.")
        sys.exit(1)

    hands = MP_HANDS.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF
    )
    return cap, hands


def process_frame(frame, hands):
    frame = cv2.flip(frame, 1)
    display = frame.copy()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    hand_detected = results.multi_hand_landmarks is not None
    if hand_detected:
        for hand_lm in results.multi_hand_landmarks:
            MP_DRAWING.draw_landmarks(...)
    arr = landmarks_to_array(results.multi_hand_landmarks, results.multi_handedness)
    return display, hand_detected, arr


def draw_ui(display, label, recorded, target, auto_mode, state, elapsed=None):
    """Tambahkan semua teks dan progress bar."""
    # Label & jumlah
    draw_text_with_background(display, f"Label: {label}", (10, 30))
    draw_text_with_background(display, f"Recorded: {recorded}/{target}", (10, 65))

    # Mode auto
    mode_color = (0,255,0) if auto_mode else (0,0,255)
    draw_text_with_background(display, f"Auto: {'ON' if auto_mode else 'OFF'}",
                              (display.shape[1]-200, 30), fg_color=mode_color)

    # State info
    if state == "waiting":
        draw_text_with_background(display, "Tekan 'R' (manual) atau 'F' (auto)", (10, 100))
    elif state == "countdown":
        if elapsed is not None:
            remain = COUNTDOWN_SECONDS - int(elapsed)
            draw_text_with_background(display, f"Countdown: {remain}", (10, 100), fg_color=(0,0,255))
    elif state == "go":
        draw_text_with_background(display, "GO!", (10, 100), fg_color=(0,255,0))
    elif state == "recording":
        if elapsed is not None:
            pct = min(100, int(elapsed / RECORD_DURATION * 100))
            draw_text_with_background(display, f"Merekam... {elapsed:.1f}s", (10, 100), fg_color=(0,255,255))
            draw_progress_bar(display, 10, 130, 300, 10, pct)
    elif state == "delay":
        if elapsed is not None:
            draw_text_with_background(display, f"Jeda... {AUTO_DELAY - elapsed:.1f}s", (10, 100))


def handle_keyboard(key, auto_mode, state, recorded, target, session_files, label, hand_detected):
    """Mengembalikan tuple (auto_mode, quit_flag, new_state, new_state_start, saved_flag) atau None jika tidak ada perubahan."""
    quit_flag = False

    if key == ord('f'):
        auto_mode = not auto_mode
        if auto_mode:
            print("  Auto Record: ON")
        else:
            print("  Auto Record: OFF")
        return auto_mode, quit_flag, None, None, None

    if key == ord('q'):
        print("Dihentikan oleh pengguna.")
        if session_files:
            print("  Menghapus file yang baru direkam...")
            for f in session_files:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"    Dihapus: {f}")
        quit_flag = True
        return auto_mode, quit_flag, None, None, None

    if state == "waiting" and key == ord('r'):
        if not hand_detected:
            print("  Tangan tidak terdeteksi. Countdown dibatalkan.")
            return auto_mode, quit_flag, None, None, None
        print(f"  [{recorded+1}/{target}] Memulai countdown...")
        return auto_mode, quit_flag, "countdown", time.time(), False

    if state == "waiting" and auto_mode and recorded < target:
        if not hand_detected:
            # Tidak munculkan error terus-menerus, cukup skip
            return auto_mode, quit_flag, None, None, None
        print(f"  [{recorded+1}/{target}] Auto memulai countdown...")
        return auto_mode, quit_flag, "countdown", time.time(), False

    return auto_mode, quit_flag, None, None, None


def save_sequence(label, recorded, frames_buffer):
    """Resample dan simpan sequence ke .npz, kembalikan nama file."""
    if len(frames_buffer) > 0:
        all_frames = np.array(frames_buffer)
        indices = np.linspace(0, all_frames.shape[0]-1, SEQ_LEN, dtype=int)
        sampled = all_frames[indices]
    else:
        sampled = np.zeros((SEQ_LEN, FEAT_DIM), dtype=np.float32)

    fname = os.path.join(DATA_DIR, f"{label}_{recorded+1}_{int(time.time())}.npz")
    np.savez_compressed(fname, sequence=sampled, label=label)
    return fname


def cleanup(cap, hands):
    """Tutup kamera dan MediaPipe."""
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="Nama gerakan (misal: halo)")
    parser.add_argument("count", type=int, help="Jumlah rekaman (misal: 30)")
    args = parser.parse_args()

    label = args.label
    target = args.count
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"=== GestureFlow Trainer ===")
    print(f"Label: {label} | Target: {target} rekaman")
    print("Instruksi: R=Manual, F=Auto, Q=Keluar")
    print("=================================")

    cap, hands = init_camera()

    recorded = 0
    state = "waiting"
    state_start = 0
    frames_buffer = []
    saved = False
    auto_mode = False
    session_files = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display, hand_detected, land_arr = process_frame(frame, hands)
        now = time.time()

        # Gambar UI
        elapsed = now - state_start if state in ("countdown", "recording", "delay", "go") else None
        draw_ui(display, label, recorded, target, auto_mode, state, elapsed)

        # Keyboard handling
        key = cv2.waitKey(1) & 0xFF
        updated_auto, quit_flag, new_state, new_start, reset_saved = handle_keyboard(
            key, auto_mode, state, recorded, target, session_files, label, hand_detected
        )
        if quit_flag:
            break
        if updated_auto is not None:
            auto_mode = updated_auto
        if new_state is not None:
            state = new_state
            state_start = new_start if new_start is not None else now
        if reset_saved is not None:
            saved = reset_saved

        # State machine
        if state == "countdown":
            if now - state_start >= COUNTDOWN_SECONDS:
                state = "go"
                state_start = now

        elif state == "go":
            if now - state_start >= 0.3:
                state = "recording"
                state_start = now
                frames_buffer = []
                print("  GO! Rekam...")

        elif state == "recording":
            frames_buffer.append(land_arr)
            if now - state_start >= RECORD_DURATION and not saved:
                saved = True
                fname = save_sequence(label, recorded, frames_buffer)
                session_files.append(fname)   # selalu lacak
                recorded += 1
                print(f"  Tersimpan: {fname} ({recorded}/{target})")

                if recorded >= target:
                    print(f"=== Selesai. {recorded} sampel tersimpan ===")
                    break

                if auto_mode:
                    state = "delay"
                    state_start = now
                    print("  Jeda 1 detik...")
                else:
                    state = "waiting"

        elif state == "delay":
            if now - state_start >= AUTO_DELAY:
                if auto_mode and recorded < target:
                    if hand_detected:
                        state = "countdown"
                        state_start = now
                        saved = False
                        print(f"  [{recorded+1}/{target}] Auto memulai countdown...")
                    else:
                        # Tangan tidak terdeteksi, tetap di delay sampai ada tangan
                        pass
                else:
                    state = "waiting"

        cv2.imshow("GestureFlow Trainer", display)

    cleanup(cap, hands)

if __name__ == "__main__":
    main()