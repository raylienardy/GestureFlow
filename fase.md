Kita akan buat proyek baru **GestureFlow** — aplikasi web AI pengenal gerakan dinamis berbasis TensorFlow.js, dengan trainer Python untuk merekam gerakan kustom. Semua berjalan tanpa API, model di-host di Vercel.

---

## 📌 **FASE-FASE PENGEMBANGAN GESTUREFLOW**

### **Fase 1 – Inisialisasi Proyek & Struktur Folder**

**Tujuan:** Menyiapkan dua folder utama, dependensi, dan konfigurasi dasar.

**Langkah:**

1. Buat folder root `GestureFlow/` dengan subfolder:
   ```
   GestureFlow/
   ├── trainer/          # Python: rekam dataset
   └── web/              # Web: TensorFlow.js + UI
       ├── public/
       │   └── model/    # Model tfjs & labels.json
       ├── src/
       │   ├── js/
       │   │   ├── camera.js
       │   │   ├── handDetector.js
       │   │   ├── preprocessor.js
       │   │   └── classifier.js
       │   ├── css/
       │   │   └── style.css
       │   └── index.html
       └── vercel.json
   ```
2. Di `trainer/`, install package: `opencv-python mediapipe numpy tensorflow`.

---

### **Fase 2 – Trainer Python (Rekam Gerakan)**

**Tujuan:** Membangun GUI interaktif untuk merekam sekuens gerakan dengan countdown dan auto-save.

**Fitur:**

- Jalankan `python train.py salam 30`
- GUI menampilkan webcam, instruksi.
- Tekan `R` → countdown 3-2-1 → rekam 2 detik → otomatis simpan `.npz` → kembali ke mode menunggu.
- Ulangi hingga jumlah tercapai, lalu program selesai.

**Kode `trainer/train.py`** akan saya berikan setelah Fase ini disetujui.

---

### **Fase 3 – Pelatihan Model LSTM**

**Tujuan:** Melatih model dengan data `.npz` dan mengekspor ke `tfjs`.

**Langkah:**

1. Gunakan arsitektur LSTM dari `gesture_ai.py`: 2 layer LSTM (128, 64) + dropout.
2. Input: `(30, 126)` — 30 frame, 126 fitur (2 tangan × 63, tanpa velocity).
3. Output: softmax sesuai jumlah kelas.
4. Simpan `.h5` → konversi ke `tfjs` dengan `tensorflowjs_converter`.
5. Copy `model.json` dan shard weights ke `web/public/model/`.
6. Buat `labels.json` berdasarkan mapping.

---

### **Fase 4 – Konfigurasi TensorFlow.js & Preprocessing**

**Tujuan:** Memuat model di browser dan menyamakan preprocessing dengan Python.

**Langkah:**

1. Gunakan MediaPipe Hands JS (CDN) untuk ekstrak landmark 21 titik.
2. Preprocessing:
   - Translasi dengan wrist.
   - Normalisasi skala.
   - Susun vektor 126 (2 tangan dipadatkan).
   - Buffer 30 frame.
3. Inference setiap buffer penuh → prediksi label + confidence.

---

### **Fase 5 – Antarmuka Web (Google Doodle Style)**

**Tujuan:** Membangun UI interaktif, playful, dan responsif.

**Fitur:**

- Kamera di tengah dengan overlay animasi titik tangan.
- Kotak "Status" yang berubah warna saat mendeteksi.
- Nama gerakan yang terdeteksi muncul dengan animasi skala.
- Confidence bar dinamis.
- Tombol "Mulai Deteksi" / "Berhenti".
- Desain warna cerah, rounded corner, animasi halus.

**File utama:**

- `index.html`: struktur utama.
- `style.css`: tema ala Google Games.
- `camera.js`: stream webcam.
- `handDetector.js`: MediaPipe Hands.
- `preprocessor.js`: fungsi normalisasi.
- `classifier.js`: load model, inferensi.

---

### **Fase 6 – Integrasi & Deployment ke Vercel**

**Tujuan:** Men-deploy aplikasi web statis agar bisa diakses publik.

**Langkah:**

1. Buat `vercel.json` untuk mengatur routing:
   ```json
   {
     "builds": [{ "src": "index.html", "use": "@vercel/static" }],
     "routes": [{ "src": "/(.*)", "dest": "/index.html" }]
   }
   ```
2. Simpan di `web/`.
3. Hubungkan ke GitHub dan deploy via Vercel.

---

## ✅ **Rencana Eksekusi**

Kita akan mulai dari **Fase 1** dan **Fase 2** sekarang. Setelah itu lanjutkan bertahap. Apakah Anda setuju dengan struktur dan alur di atas? Jika ada yang ingin diubah (misal: jumlah frame, format input, atau fitur tambahan), sampaikan sekarang sebelum saya buat kodenya. Setelah konfirmasi, saya langsung berikan kode lengkap untuk trainer.
