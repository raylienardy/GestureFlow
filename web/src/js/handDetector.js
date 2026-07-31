/**
 * handDetector.js
 * Membungkus MediaPipe Hands (CDN) agar mudah digunakan.
 */

export class HandDetector {
  constructor() {
    this.hands = null;
    this.onResultsCallback = null;
  }

  async init() {
    // Pastikan window.Hands sudah tersedia (dari CDN)
    if (!window.Hands) {
      throw new Error(
        "MediaPipe Hands tidak ditemukan. Pastikan CDN dimuat di HTML.",
      );
    }
    this.hands = new window.Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    await this.hands.initialize();
  }

  /**
   * Mulai deteksi pada elemen video.
   * @param {HTMLVideoElement} video
   * @param {Function} callback dipanggil setiap kali hasil deteksi siap.
   */
  start(video, callback) {
    this.onResultsCallback = callback;

    const hands = this.hands;
    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    hands.onResults((results) => {
      if (this.onResultsCallback) {
        this.onResultsCallback(results);
      }
    });

    // Kirim frame dari video ke MediaPipe setiap animasi
    const processFrame = async () => {
      if (video.readyState >= 2) {
        await hands.send({ image: video });
      }
      requestAnimationFrame(processFrame);
    };
    processFrame();
  }

  stop() {
    this.onResultsCallback = null;
    if (this.hands) {
      this.hands.close();
    }
  }

  /**
   * Gambar landmark ke canvas.
   */
  static draw(ctx, results, width, height, mirror = true) {
    if (!results.multiHandLandmarks) return;

    ctx.save();
    if (mirror) {
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
    }

    const landmarks = results.multiHandLandmarks;
    const connections = results.multiHandLandmarks
      ? window.HAND_CONNECTIONS || getDefaultConnections()
      : [];

    for (const hand of landmarks) {
      // Gambar titik
      ctx.fillStyle = "#FF4081";
      for (const lm of hand) {
        ctx.beginPath();
        ctx.arc(lm.x * width, lm.y * height, 3, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Gambar garis
      ctx.strokeStyle = "#FF4081";
      ctx.lineWidth = 2;
      for (const [i, j] of connections) {
        const a = hand[i];
        const b = hand[j];
        ctx.beginPath();
        ctx.moveTo(a.x * width, a.y * height);
        ctx.lineTo(b.x * width, b.y * height);
        ctx.stroke();
      }
    }

    ctx.restore();
  }
}

// Koneksi default jika tidak tersedia global
function getDefaultConnections() {
  return [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
  ];
}
