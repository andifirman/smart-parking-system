const form = document.querySelector("#upload-form");
const imageInput = document.querySelector("#image-input");
const fileLabel = document.querySelector("#file-label");
const analyzeButton = document.querySelector("#analyze-button");
const message = document.querySelector("#message");
const results = document.querySelector("#results");
const detectionsContainer = document.querySelector("#detections");
const template = document.querySelector("#detection-template");
const cameraButton = document.querySelector("#camera-button");
const cameraPanel = document.querySelector("#camera-panel");
const cameraPreview = document.querySelector("#camera-preview");
const cameraCanvas = document.querySelector("#camera-canvas");
const cameraState = document.querySelector("#camera-state");
const startCameraButton = document.querySelector("#start-camera-button");
const captureButton = document.querySelector("#capture-button");
const closeCameraButton = document.querySelector("#close-camera-button");
const sensorState = document.querySelector("#sensor-state");
const armSensorButton = document.querySelector("#arm-sensor-button");
const simulateSensorButton = document.querySelector("#simulate-sensor-button");
let cameraStream = null;
let sensorArmed = false;
let lastSensorEventId = null;
let analysisInProgress = false;

const formatPercent = (value) => `${(Number(value) * 100).toFixed(1)}%`;

imageInput.addEventListener("change", () => {
  fileLabel.textContent = imageInput.files[0]?.name ?? "Choose image";
});

async function checkHealth() {
  const state = document.querySelector("#service-state");
  try {
    const response = await fetch("/health");
    const data = await response.json();
    state.className = `service-state ${data.models_loaded ? "ready" : "error"}`;
    state.lastChild.textContent = data.models_loaded ? `Models ready · OCR ${data.devices.recognizer}` : "Models unavailable";
  } catch {
    state.className = "service-state error";
    state.lastChild.textContent = "Service unavailable";
  }
}

function showMessage(text, isError = false) {
  message.hidden = false;
  message.textContent = text;
  message.className = `message${isError ? " error" : ""}`;
}

function renderResults(data) {
  results.hidden = false;
  message.hidden = true;
  document.querySelector("#annotated-image").src = `data:image/jpeg;base64,${data.annotated_image_base64}`;
  document.querySelector("#processing-time").textContent = `${data.processing_ms.toFixed(1)} ms`;
  document.querySelector("#detection-count").textContent = `${data.detection_count} detected`;
  detectionsContainer.replaceChildren();

  if (!data.detections.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No license plate passed the detection threshold.";
    detectionsContainer.append(empty);
    return;
  }

  data.detections.forEach((detection) => {
    const node = template.content.cloneNode(true);
    node.querySelector(".plate-text").textContent = detection.plate_text || "UNREADABLE";
    const badge = node.querySelector(".status-badge");
    badge.textContent = detection.status;
    badge.classList.add(detection.status.toLowerCase());
    node.querySelector(".plate-crop").src = `data:image/jpeg;base64,${detection.crop_image_base64}`;
    node.querySelector(".detection-confidence").textContent = formatPercent(detection.detection_confidence);
    node.querySelector(".ocr-confidence").textContent = formatPercent(detection.ocr_confidence);
    node.querySelector(".geometry-method").textContent =
      detection.geometry_method === "segmentation_rectified" ? "Segmented + rectified" : "Bounding box";
    node.querySelector(".ocr-variant").textContent = detection.ocr_variant;
    node.querySelector(".ocr-latency").textContent = `${detection.ocr_latency_ms.toFixed(1)} ms`;
    detectionsContainer.append(node);
  });
}

async function analyzeImage(file) {
  if (analysisInProgress) return;
  analysisInProgress = true;
  analyzeButton.disabled = true;
  captureButton.disabled = true;
  analyzeButton.textContent = "Analyzing...";
  showMessage("Loading models and analyzing image...");
  const payload = new FormData();
  payload.append("image", file, file.name || "camera-capture.jpg");

  try {
    const response = await fetch("/api/anpr", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Analysis failed.");
    renderResults(data);
  } catch (error) {
    results.hidden = true;
    showMessage(error.message, true);
  } finally {
    analysisInProgress = false;
    analyzeButton.disabled = false;
    captureButton.disabled = !cameraStream;
    analyzeButton.textContent = "Analyze plate";
  }
}

function stopCamera() {
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraPreview.srcObject = null;
  cameraState.textContent = "Camera is off";
  startCameraButton.disabled = false;
  captureButton.disabled = true;
  sensorArmed = false;
  armSensorButton.disabled = true;
  armSensorButton.textContent = "Arm sensor";
}

async function startCamera() {
  if (cameraStream) return true;
  if (!navigator.mediaDevices?.getUserMedia) {
    showMessage("Camera access is not supported by this browser.", true);
    return false;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
    cameraPreview.srcObject = cameraStream;
    await cameraPreview.play();
    cameraState.textContent = `${cameraPreview.videoWidth} × ${cameraPreview.videoHeight}`;
    startCameraButton.disabled = true;
    captureButton.disabled = false;
    armSensorButton.disabled = false;
    return true;
  } catch (error) {
    showMessage(`Camera could not be started: ${error.message}`, true);
    return false;
  }
}

async function armSensor() {
  if (!cameraStream) return;
  await pollSensor();
  sensorArmed = true;
  armSensorButton.textContent = "Disarm sensor";
  cameraState.textContent = "Camera ready · waiting for vehicle";
}

cameraButton.addEventListener("click", async () => {
  cameraPanel.hidden = false;
  cameraPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  if (await startCamera()) await armSensor();
});

startCameraButton.addEventListener("click", async () => {
  await startCamera();
});

function captureCameraFrame() {
  const width = cameraPreview.videoWidth;
  const height = cameraPreview.videoHeight;
  if (!width || !height) return;
  cameraCanvas.width = width;
  cameraCanvas.height = height;
  cameraCanvas.getContext("2d").drawImage(cameraPreview, 0, 0, width, height);
  cameraCanvas.toBlob(
    (blob) => blob && analyzeImage(new File([blob], "camera-capture.jpg", { type: "image/jpeg" })),
    "image/jpeg",
    0.92,
  );
}

captureButton.addEventListener("click", () => {
  captureCameraFrame();
});

async function pollSensor() {
  try {
    const response = await fetch("/api/sensor/status");
    const data = await response.json();
    const distance = data.distance_cm == null ? "" : ` · ${Number(data.distance_cm).toFixed(1)} cm`;
    sensorState.textContent = data.connected
      ? `Sensor connected on ${data.port}${distance}`
      : data.enabled
        ? `Sensor waiting on ${data.port}`
        : "Sensor disabled · configure ARDUINO_PORT";
    if (lastSensorEventId === null) lastSensorEventId = data.event_id;
    if (data.event_id > lastSensorEventId) {
      lastSensorEventId = data.event_id;
      if (sensorArmed && cameraStream && !analysisInProgress) {
        cameraState.textContent = `Vehicle detected${distance} · capturing`;
        captureCameraFrame();
      }
    }
  } catch {
    sensorState.textContent = "Sensor service unavailable";
  }
}

armSensorButton.addEventListener("click", async () => {
  if (!cameraStream) {
    showMessage("Start the camera before arming the sensor.", true);
    return;
  }
  if (!sensorArmed) {
    await armSensor();
    return;
  }
  sensorArmed = false;
  armSensorButton.textContent = sensorArmed ? "Disarm sensor" : "Arm sensor";
  cameraState.textContent = sensorArmed ? "Camera ready · waiting for vehicle" : "Camera ready · sensor disarmed";
});

simulateSensorButton.addEventListener("click", async () => {
  try {
    const response = await fetch("/api/sensor/simulate", { method: "POST" });
    if (!response.ok) throw new Error("Simulation is disabled.");
    await pollSensor();
  } catch (error) {
    showMessage(error.message, true);
  }
});

closeCameraButton.addEventListener("click", () => {
  stopCamera();
  cameraPanel.hidden = true;
});

window.addEventListener("beforeunload", stopCamera);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!imageInput.files.length) return;
  analyzeImage(imageInput.files[0]);
});

checkHealth();
pollSensor();
window.setInterval(pollSensor, 500);
