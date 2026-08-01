import { HandDetector } from "./handDetector.js";
import { landmarksToFeature } from "./preprocessor.js";
import { GestureClassifier } from "./classifier.js";

// DOM elements
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
    detector.start(video, onResults);
  };
  video.play();
  btnStart.disabled = true;
  btnStop.disabled = false;
  statusBadge.classList.add("active");
  statusSpan.textContent = "Mendeteksi...";
  instruction.textContent = "Lakukan gerakan dengan mantap";
}

function onResults(results) {
  const feature = landmarksToFeature(results.multiHandLandmarks);
  classifier.addFrame(feature);

  // Overlay
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (results.multiHandLandmarks && showOverlay) {
    HandDetector.draw(ctx, results, canvas.width, canvas.height);
  }

  // Inferensi
  if (classifier.isReady()) {
    const pred = classifier.predict();
    if (pred) {
      const confPercent = Math.round(pred.confidence * 100);
      labelSpan.textContent = pred.label;
      confidenceBar.style.width = confPercent + "%";
      confidenceText.textContent = confPercent + "%";

      if (pred.label !== lastLabel) {
        labelSpan.classList.remove("detected");
        void labelSpan.offsetWidth; // reflow
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
  } else {
    statusSpan.textContent = `Mengumpulkan frame... (${classifier.buffer.length}/30)`;
    confidenceBar.style.width =
      Math.round((classifier.buffer.length / 30) * 100) + "%";
    confidenceText.textContent = "";
    instruction.style.opacity = 1;
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
}

// Event listeners
btnStart.addEventListener("click", startDetection);
btnStop.addEventListener("click", stopDetection);
canvas.addEventListener("dblclick", () => {
  showOverlay = !showOverlay;
  canvas.style.opacity = showOverlay ? "1" : "0.2";
});

init();
