
const API_URL = "http://127.0.0.1:8000/predict";

const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");
const predictBtn = document.getElementById("predictBtn");
const statusMessage = document.getElementById("statusMessage");

const emptyState = document.getElementById("emptyState");
const resultContainer = document.getElementById("resultContainer");

let selectedFile = null;

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];

  selectedFile = null;
  predictBtn.disabled = true;
  resultContainer.classList.add("hidden");
  emptyState.classList.remove("hidden");
  previewContainer.classList.add("hidden");
  statusMessage.textContent = "";

  if (!file) return;

  const allowedTypes = ["image/jpeg", "image/png"];

  if (!allowedTypes.includes(file.type)) {
    statusMessage.textContent = "Please select a JPG or PNG image.";
    statusMessage.className = "status error";
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    statusMessage.textContent = "Image must be smaller than 10 MB.";
    statusMessage.className = "status error";
    return;
  }

  selectedFile = file;

  imagePreview.src = URL.createObjectURL(file);
  previewContainer.classList.remove("hidden");
  predictBtn.disabled = false;
});

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  predictBtn.disabled = true;
  predictBtn.textContent = "Analyzing...";
  statusMessage.className = "status";
  statusMessage.textContent = "Running model inference...";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      let message = "Prediction failed.";
      try {
        const error = await response.json();
        message = error.detail || message;
      } catch (_) {}
      throw new Error(message);
    }

    const result = await response.json();

    const displayNames = {
      "Healthy": "Healthy Bone",
      "Simple": "Simple Fracture",
      "Wedge_Complex": "Wedge / Complex Fracture"
    };

    document.getElementById("predictionLabel").textContent =
      displayNames[result.prediction] || result.prediction;

    document.getElementById("confidenceValue").textContent =
      (result.confidence * 100).toFixed(2) + "%";

    const probabilityBars = document.getElementById("probabilityBars");
    probabilityBars.innerHTML = "";

    for (const [className, probability] of Object.entries(result.probabilities)) {
      const percentage = Math.max(0, Math.min(100, probability * 100));

      const row = document.createElement("div");
      row.className = "probability-row";

      const label = document.createElement("div");
      label.className = "probability-label";

      const name = document.createElement("span");
      name.textContent = displayNames[className] || className;

      const value = document.createElement("span");
      value.textContent = percentage.toFixed(2) + "%";

      label.append(name, value);

      const track = document.createElement("div");
      track.className = "progress-track";

      const fill = document.createElement("div");
      fill.className = "progress-fill";
      fill.style.width = percentage + "%";

      track.appendChild(fill);
      row.append(label, track);
      probabilityBars.appendChild(row);
    }

    emptyState.classList.add("hidden");
    resultContainer.classList.remove("hidden");
    statusMessage.textContent = "Analysis completed.";

  } catch (error) {
    statusMessage.textContent = error.message;
    statusMessage.className = "status error";
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Classify X-ray →";
  }
});
