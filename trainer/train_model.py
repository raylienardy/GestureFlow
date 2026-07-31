#!/usr/bin/env python3
"""
Melatih model LSTM dari data .npz hasil rekaman trainer.
Usage:
  python train_model.py              # pakai default data/ dan output/
  python train_model.py --data_dir data --out_model model.h5
"""

import os
import argparse
import json
import numpy as np
from collections import Counter

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Konfigurasi
SEQ_LEN = 30
FEAT_DIM = 126          # 2 tangan x 63

def load_all_data(data_dir):
    """Memuat semua .npz, mengembalikan X (N,30,126) dan y (label string)."""
    files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    if not files:
        raise ValueError(f"Tidak ada file .npz di {data_dir}.")

    X_list, y_list = [], []
    label_set = set()

    for fname in files:
        path = os.path.join(data_dir, fname)
        data = np.load(path, allow_pickle=True)
        seq = data['sequence']          # expected (30, 126)
        label = str(data['label'])

        if seq.shape != (SEQ_LEN, FEAT_DIM):
            print(f"  Lewati {fname}: shape {seq.shape} tidak sesuai.")
            continue

        X_list.append(seq)
        y_list.append(label)
        label_set.add(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    print(f"Dimuat {len(X)} sampel, label unik: {sorted(label_set)}")
    return X, y, sorted(label_set)


def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Masking(mask_value=0.0),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.4),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--out_model', default='model.h5')
    parser.add_argument('--out_labels', default='labels.json')
    parser.add_argument('--tfjs_dir', default='../web/public/model')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 1. Muat data
    X, y_str, labels = load_all_data(args.data_dir)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    y_idx = np.array([label_to_idx[l] for l in y_str])
    num_classes = len(labels)

    # One-hot
    y_cat = tf.keras.utils.to_categorical(y_idx, num_classes)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=args.test_size, random_state=args.seed, stratify=y_idx
    )

    # 3. Class weights
    y_train_idx = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(y_train_idx),
                                         y=y_train_idx)
    cw = {i: float(w) for i, w in enumerate(class_weights)}
    print("Class weights:", cw)

    # 4. Bangun & latih model
    model = build_model(input_shape=(SEQ_LEN, FEAT_DIM), num_classes=num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=cw,
        callbacks=callbacks
    )

    # 5. Evaluasi
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    # 6. Simpan model .h5
    model.save(args.out_model)
    print(f"Model disimpan ke {args.out_model}")

    # 7. Konversi ke TensorFlow.js
    print("Mengonversi ke TensorFlow.js...")
    os.system(f'tensorflowjs_converter --input_format=keras {args.out_model} {args.tfjs_dir}')
    print(f"Model TFjs disimpan ke {args.tfjs_dir}")

    # 8. Simpan labels.json
    with open(args.out_labels, 'w') as f:
        json.dump(idx_to_label, f, indent=2)
    # Salin juga ke folder web
    with open(os.path.join(args.tfjs_dir, 'labels.json'), 'w') as f:
        json.dump(idx_to_label, f, indent=2)
    print(f"Labels disimpan ke {args.out_labels} dan {args.tfjs_dir}/labels.json")


if __name__ == "__main__":
    main()