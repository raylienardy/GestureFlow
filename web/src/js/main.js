import { HandDetector } from "./handDetector.js";
import { landmarksToFeature } from "./preprocessor.js";
import { GestureClassifier } from "./classifier.js";

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
    statusDiv.textContent = "Siap. Klik Mulai Deteksi.";
    btnStart.disabled = false;
  } catch (e) {
    statusDiv.textContent = "Gagal: " + e.message;
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
}

function onResults(results) {
  const feature = landmarksToFeature(results.multiHandLandmarks);
  classifier.addFrame(feature);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (results.multiHandLandmarks && showOverlay) {
    HandDetector.draw(ctx, results, canvas.width, canvas.height);
  }
  if (classifier.isReady()) {
    const pred = classifier.predict();
    if (pred) {
      labelSpan.textContent = pred.label;
      confidenceSpan.textContent = Math.round(pred.confidence * 100) + "%";
      statusDiv.textContent =
        pred.confidence > 0.7 ? `Terdeteksi: ${pred.label}` : "Kurang yakin...";
    }
  } else {
    statusDiv.textContent = `Mengumpulkan frame... (${classifier.buffer.length}/30)`;
  }
}

function stopDetection() {
  if (detector) detector.stop();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  btnStart.disabled = false;
  btnStop.disabled = true;
  labelSpan.textContent = "-";
  confidenceSpan.textContent = "-";
  statusDiv.textContent = "Berhenti";
}

btnStart.addEventListener("click", startDetection);
btnStop.addEventListener("click", stopDetection);
canvas.addEventListener("dblclick", () => {
  showOverlay = !showOverlay;
  canvas.style.opacity = showOverlay ? "1" : "0.2";
});
init();
