import torch
from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from model import build_model

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= MODEL LOADING =================
print("=" * 50)
print("Starting model loading...")
print("=" * 50)

MODEL_PATH = "models/plant_disease_classifier.pth"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

print(f"Loading model from: {MODEL_PATH}")
print(f"Model size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")

start_time = time.time()

try:
    torch_home = os.getenv("TORCH_HOME", "/app/torch_cache")
    weights_path = f"{torch_home}/hub/checkpoints/efficientnet_b0_rwightman-3dd342df.pth"

    if os.path.exists(weights_path):
        print("✓ PyTorch weights found in cache")
    else:
        print("⚠ PyTorch weights not found (will download if needed)")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded in {time.time() - start_time:.2f}s")
    print(f"✓ Classes: {num_classes}")
    print(f"✓ Device: {device}")

except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = None
    class_names = []

print("=" * 50)

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def clean_label(label):
    return (
        label.replace("___", " – ")
             .replace("__", " ")
             .replace("_", " ")
    )

def predict_image(image_path):
    if model is None:
        return "Model not loaded", 0.0

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return clean_label(class_names[pred.item()]), confidence.item()

# ================= ROUTES =================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = confidence = image_path = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)
            prediction, confidence = predict_image(image_path)
            confidence = round(confidence * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

@app.route("/health")
def health():
    return {
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None
    }, 200

@app.route("/ready")
def ready():
    return ({"ready": True}, 200) if model else ({"ready": False}, 503)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f" Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
