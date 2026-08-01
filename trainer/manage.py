#!/usr/bin/env python3
"""
Kelola daftar bahasa isyarat yang sudah direkam.
Usage: python manage.py
"""

import os
import sys
import glob
from collections import defaultdict

DATA_DIR = "data"

def get_label_from_filename(fname):
    """Ambil label dari nama file: halo_1_1234567890.npz -> halo"""
    base = os.path.basename(fname)
    # Format: <label>_<angka>_<timestamp>.npz
    parts = base.rsplit("_", 2)
    if len(parts) >= 3:
        return parts[0]
    return "unknown"

def list_gestures():
    """Kembalikan dictionary {label: jumlah file}"""
    files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    counts = defaultdict(int)
    for f in files:
        label = get_label_from_filename(f)
        counts[label] += 1
    return dict(sorted(counts.items()))

def delete_gesture(label):
    """Hapus semua file .npz dengan label tertentu."""
    pattern = os.path.join(DATA_DIR, f"{label}_*.npz")
    files = glob.glob(pattern)
    if not files:
        print(f"  Tidak ada file untuk '{label}'.")
        return 0
    for f in files:
        os.remove(f)
    print(f"  {len(files)} file untuk '{label}' berhasil dihapus.")
    return len(files)

def print_list(gestures):
    """Tampilkan daftar label dan jumlah sampel."""
    if not gestures:
        print("\n  (belum ada data)\n")
        return
    print("\n  Daftar Bahasa Isyarat:")
    print("  ┌────┬──────────────────┬──────────┐")
    print("  │ No │ Label            │ Sampel   │")
    print("  ├────┼──────────────────┼──────────┤")
    for i, (label, count) in enumerate(gestures.items(), 1):
        print(f"  │ {i:<2} │ {label:<16} │ {count:<8} │")
    print("  └────┴──────────────────┴──────────┘")
    print()

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 50)
    print("        GestureFlow - Manajemen Data")
    print("=" * 50)
    print("Perintah: (d) hapus, (q) keluar\n")

    while True:
        gestures = list_gestures()
        print_list(gestures)

        if not gestures:
            print("Belum ada data. Silakan rekam dengan train.py")
            break

        cmd = input("Masukkan perintah (d/q): ").strip().lower()

        if cmd == 'q':
            print("Sampai jumpa!")
            break
        elif cmd == 'd':
            try:
                num = int(input("Nomor label yang akan dihapus: "))
                labels = list(gestures.keys())
                if 1 <= num <= len(labels):
                    label = labels[num - 1]
                    confirm = input(f"Yakin hapus semua '{label}'? (y/n): ").strip().lower()
                    if confirm == 'y':
                        delete_gesture(label)
                    else:
                        print("  Dibatalkan.")
                else:
                    print("  Nomor tidak valid.")
            except ValueError:
                print("  Masukkan angka.")
        else:
            print("  Perintah tidak dikenal. Gunakan 'd' atau 'q'.")
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDihentikan.")
        sys.exit(0)