import { HandDetector } from "./handDetector.js";
import { landmarksToFeature } from "./preprocessor.js";
import { GestureClassifier } from "./classifier.js";

// DOM
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const statusBadge = document.getElementById("status-badge");
const statusSpan = document.getElementById("status");
const labelSpan = document.getElementById("label");
const confidenceBar = document.getElementById("confidence-bar");
const confidenceText = document.getElementById("confidence");
const instruction = document.getElementById("instruction");

let detector, classifier, stream;
let showOverlay = true;
let lastLabel = "";

// Untuk deteksi stabilitas
let lastFeature = null;
let stableCount = 0;
const STABLE_THRESHOLD = 0.01;
const REQUIRED_STABLE_FRAMES = 10;

document.getElementById("btn-export").addEventListener("click", () => {
  if (!lastFeature) {
    alert("Tangan belum terdeteksi. Tunjukkan tangan dulu.");
    return;
  }
  const json = JSON.stringify(lastFeature);
  console.log("EXPORTED_FRAME:", json);
  alert(
    "Array diekspor ke Console (F12). Salin nilai 5 pertama dan bandingkan dengan Python.",
  );
});

async function init() {
  try {
    statusSpan.textContent = "Memuat model...";
    btnStart.disabled = true;
    detector = new HandDetector();
    classifier = new GestureClassifier();
    await Promise.all([detector.init(), classifier.init()]);
    statusSpan.textContent = "Siap. Klik Mulai Deteksi.";
    btnStart.disabled = false;
  } catch (e) {
    statusSpan.textContent = "Gagal: " + e.message;
  }
}

async function startDetection() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
  });
  video.srcObject = stream;
  video.onloadedmetadata = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    detector.start(video, onFrame);
  };
  video.play();
  btnStart.disabled = true;
  btnStop.disabled = false;
  statusBadge.classList.add("active");
  statusSpan.textContent = "Mendeteksi...";
  instruction.textContent = "Lakukan gerakan dan tahan sebentar";
}

function onFrame(results) {
  const handDetected =
    results.multiHandLandmarks && results.multiHandLandmarks.length > 0;

  // Overlay
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (handDetected && showOverlay) {
    HandDetector.draw(ctx, results, canvas.width, canvas.height);
  }

  if (!handDetected) {
    // Reset buffer jika tangan hilang
    classifier.buffer = [];
    stableCount = 0;
    lastFeature = null;
    statusSpan.textContent = "Tangan tidak terdeteksi";
    return;
  }

  const feature = landmarksToFeature(
    results.multiHandLandmarks,
    results.multiHandedness,
  );

  // Cek stabilitas: bandingkan dengan frame sebelumnya
  if (lastFeature) {
    let diff = 0;
    for (let i = 0; i < feature.length; i++) {
      diff += Math.abs(feature[i] - lastFeature[i]);
    }
    if (diff < STABLE_THRESHOLD) {
      stableCount++;
    } else {
      stableCount = 0;
    }
  }
  lastFeature = feature;

  // Hanya tambahkan ke buffer jika pose stabil selama beberapa frame
  if (stableCount >= REQUIRED_STABLE_FRAMES) {
    classifier.addFrame(feature);
    statusSpan.textContent = `Mengumpulkan... (${classifier.buffer.length}/30)`;
  } else {
    // Reset buffer jika belum stabil agar tidak tercampur gerakan transisi
    classifier.buffer = [];
    statusSpan.textContent = "Tunggu stabil...";
  }

  // Inferensi jika buffer penuh
  if (classifier.isReady()) {
    const pred = classifier.predict();
    if (pred) {
      const confPercent = Math.round(pred.confidence * 100);
      labelSpan.textContent = pred.label;
      confidenceBar.style.width = confPercent + "%";
      confidenceText.textContent = confPercent + "%";

      if (pred.label !== lastLabel) {
        labelSpan.classList.remove("detected");
        void labelSpan.offsetWidth;
        labelSpan.classList.add("detected");
        lastLabel = pred.label;
      }

      if (pred.confidence > 0.7) {
        statusSpan.textContent = `Terdeteksi: ${pred.label}`;
        instruction.style.opacity = 0;
      } else {
        statusSpan.textContent = "Kurang yakin...";
        instruction.style.opacity = 1;
      }
    }
  }
}

function stopDetection() {
  if (detector) detector.stop();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  btnStart.disabled = false;
  btnStop.disabled = true;
  statusBadge.classList.remove("active");
  statusSpan.textContent = "Berhenti";
  labelSpan.textContent = "-";
  confidenceBar.style.width = "0%";
  confidenceText.textContent = "-";
  instruction.style.opacity = 1;
  lastLabel = "";
  labelSpan.classList.remove("detected");
  classifier.buffer = [];
  stableCount = 0;
  lastFeature = null;
}

btnStart.addEventListener("click", startDetection);
btnStop.addEventListener("click", stopDetection);
canvas.addEventListener("dblclick", () => {
  showOverlay = !showOverlay;
  canvas.style.opacity = showOverlay ? "1" : "0.2";
});

init();
