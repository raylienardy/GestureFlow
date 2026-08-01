# GestureFlow

Aplikasi web pengenal gerakan tangan dinamis berbasis AI.  
Rekam gestur kustom → latih model LSTM → deteksi real-time di browser (tanpa backend).

## Struktur Proyek

```
GestureFlow/
├── trainer/               # Python: rekam & latih model
│   ├── train.py           # Rekam gestur
│   ├── train_model.py     # Latih model LSTM
│   ├── manage.py          # Kelola data gestur
│   └── data/              # Dataset .npz
├── converter/             # Python: konversi H5 → TensorFlow.js
│   ├── convert_model.py   # Script konversi
│   └── venv/              # Virtual environment khusus
├── web/                   # Web: TensorFlow.js + UI
│   ├── src/
│   │   ├── index.html
│   │   ├── css/style.css
│   │   └── js/
│   │       ├── main.js          # Loop utama & UI
│   │       ├── handDetector.js  # MediaPipe Hands
│   │       ├── preprocessor.js  # Normalisasi landmark
│   │       ├── classifier.js    # Load model & inferensi
│   │       └── model/           # Model TFjs + labels.json
│   └── vercel.json
└── README.md
```

---

## 1. Training (Merekam Gestur)

### Persyaratan

- Python 3.10
- Webcam

### Setup Awal (sekali saja)

```bash
cd trainer
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
# atau:
venv\Scripts\activate          # Windows CMD / PowerShell
pip install -r requirements.txt
```

### Merekam Gestur Baru

```bash
python train.py <nama_gerakan> <jumlah_sampel>
```

Contoh:

```bash
python train.py halo 30
```

**Instruksi:**

- Setelah GUI terbuka, tekan **R** untuk memulai hitung mundur 3-2-1.
- Lakukan gerakan selama 2 detik (rekaman otomatis).
- Ulangi hingga jumlah tercapai.
- Tekan **Q** untuk keluar paksa.

Hasil rekaman disimpan di `trainer/data/` dengan format: `<label>_<n>_<timestamp>.npz`

**Tips:**

- Rekam minimal 30 sampel per label.
- Lakukan variasi: jarak, sudut, kecepatan berbeda.
- Pastikan tangan terlihat jelas, pencahayaan cukup.

### Melihat & Menghapus Data

```bash
python manage.py
```

Akan muncul daftar label beserta jumlah sampel. Tekan:

- **d** untuk menghapus label
- **q** untuk keluar

---

## 2. Melatih Model

Setelah mengumpulkan data minimal 2 label berbeda, latih model LSTM:

```bash
cd trainer
source venv/Scripts/activate   # pastikan venv aktif
python train_model.py
```

**Output:**

- `model.h5` — model Keras
- `labels.json` — mapping indeks ke label
- Folder `data/` tetap utuh (bisa dipakai ulang)

**Catatan:**

- Data akan otomatis dibagi 80% train, 20% test.
- Class weights dihitung otomatis untuk menangani ketidakseimbangan.
- Akurasi tes akan ditampilkan di akhir.

---

## 3. Konversi Model ke TensorFlow.js

### Setup Konverter (sekali saja)

```bash
cd converter
python -m venv venv
source venv/Scripts/activate
pip install tensorflow==2.10.1
pip install tensorflowjs==3.18.0
pip install numpy==1.23.5
```

### Jalankan Konversi

```bash
python convert_model.py
```

Hasil konversi akan muncul di `web/src/model/`:

- `model.json`
- `group1-shard1of1.bin`
- `labels.json`

> ⚠️ Folder `converter/` hanya untuk konversi, **tidak** untuk training.

---

## 4. Menjalankan Web

### Prasyarat

- Node.js (untuk menjalankan server lokal)

### Jalankan Web Server

```bash
cd web
npx serve src
```

Buka browser di `http://localhost:3000`.

### Penggunaan

1. Klik **Mulai Deteksi**.
2. Izinkan akses kamera.
3. Lakukan gestur yang sudah direkam (misal "halo", "salam").
4. Setelah 30 frame terkumpul, hasil prediksi akan muncul.
5. **Double-click** pada area kamera untuk menyembunyikan/menampilkan overlay kerangka tangan.
6. Klik **Berhenti** untuk menghentikan deteksi.

### Catatan

- Browser yang didukung: Chrome, Edge, Firefox (terbaru).
- Webcam harus tersedia dan tidak digunakan aplikasi lain.
- Latar belakang sederhana akan meningkatkan akurasi.

---

## 5. Deployment ke Vercel (Opsional)

1. Upload folder `web/` ke GitHub.
2. Di Vercel, import repository tersebut.
3. Atur root folder ke `web` dan output ke `src`.
4. Deploy. Model akan otomatis di-host bersama static assets.

---

## Alur Kerja Lengkap

```
Rekam gestur (train.py)
        ↓
   Dataset (.npz)
        ↓
Latih model (train_model.py)
        ↓
   model.h5 + labels.json
        ↓
Konversi ke TFjs (convert_model.py)
        ↓
   model.json + shard + labels.json
        ↓
Web siap digunakan (npx serve src)
        ↓
   Deteksi real-time di browser
```

---

## Troubleshooting

| Masalah                          | Solusi                                                      |
| -------------------------------- | ----------------------------------------------------------- |
| `ModuleNotFoundError: mediapipe` | Jalankan `pip install -r requirements.txt` di trainer/      |
| Kamera tidak terbuka             | Pastikan webcam tidak digunakan aplikasi lain               |
| Akurasi rendah                   | Tambah sampel, variasikan gerakan, rekam di kondisi berbeda |
| Model tidak bisa dimuat di web   | Pastikan file model.json dan shard ada di `web/src/model/`  |
| Error saat konversi              | Pastikan sudah setup converter sesuai langkah di atas       |
