/**
 * main.js
 * Mengatur kamera, loop deteksi, dan UI.
 */

import { HandDetector } from "./handDetector.js";
import { landmarksToFeature } from "./preprocessor.js";
import { GestureClassifier } from "./classifier.js";

// DOM elements
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const statusDiv = document.getElementById("status");
const labelSpan = document.getElementById("label");
const confidenceSpan = document.getElementById("confidence");

let detector, classifier, stream;
let showOverlay = true;

async function init() {
  try {
    statusDiv.textContent = "Memuat model...";
    btnStart.disabled = true;

    detector = new HandDetector();
    classifier = new GestureClassifier();
    await Promise.all([detector.init(), classifier.init()]);

    statusDiv.textContent = 'Siap. Klik "Mulai Deteksi".';
    btnStart.disabled = false;
  } catch (err) {
    statusDiv.textContent = "Gagal memuat model: " + err.message;
    console.error(err);
  }
}

async function startDetection() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: 640, height: 480 },
    });
    video.srcObject = stream;
    video.play();

    // Tunggu video siap
    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      detector.start(video, onHandResults);
    };

    btnStart.disabled = true;
    btnStop.disabled = false;
    statusDiv.textContent = "Mendeteksi...";
  } catch (err) {
    statusDiv.textContent = "Gagal akses kamera: " + err.message;
    console.error(err);
  }
}

function onHandResults(results) {
  // Preprocessing
  const feature = landmarksToFeature(results.multiHandLandmarks);
  classifier.addFrame(feature);

  // Gambar overlay
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (results.multiHandLandmarks && showOverlay) {
    HandDetector.draw(ctx, results, canvas.width, canvas.height);
  }

  // Inferensi jika buffer penuh
  if (classifier.isReady()) {
    const pred = classifier.predict();
    if (pred) {
      labelSpan.textContent = pred.label;
      confidenceSpan.textContent = `${Math.round(pred.confidence * 100)}%`;
      if (pred.confidence > 0.7) {
        statusDiv.textContent = `Terdeteksi: ${pred.label}`;
      } else {
        statusDiv.textContent = "Kurang yakin...";
      }
    }
  } else {
    // Tampilkan progres buffer
    statusDiv.textContent = `Mengumpulkan frame... (${classifier.buffer.length}/${30})`;
  }
}

function stopDetection() {
  detector.stop();
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  classifier.buffer = [];
  btnStart.disabled = false;
  btnStop.disabled = true;
  labelSpan.textContent = "-";
  confidenceSpan.textContent = "-";
  statusDiv.textContent = "Berhenti";
}

// Event listeners
btnStart.addEventListener("click", startDetection);
btnStop.addEventListener("click", stopDetection);

// Double-click canvas untuk toggle overlay
canvas.addEventListener("dblclick", () => {
  showOverlay = !showOverlay;
  canvas.style.opacity = showOverlay ? "1" : "0.2";
});

// Start
init();
