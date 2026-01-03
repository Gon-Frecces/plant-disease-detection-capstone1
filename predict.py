import torch
from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image
import os
import requests
from pathlib import Path
import gdown  # For Google Drive files
import time

from model import build_model

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== MODEL DOWNLOAD LOGIC ======
def download_model():
    """Download model file if it doesn't exist"""
    MODEL_PATH = "models/plant_disease_classifier.pth"
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # If model already exists, skip download
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return MODEL_PATH
    
    print("Downloading model file...")
    
    MODEL_URL = os.getenv(
        'MODEL_URL',
        'https://drive.google.com/uc?id=1NeGIjmkg_OroQerpVnlho5nKWqf51oI2' 
    )
    
    try:
        print(f"Downloading from: {MODEL_URL}")
        
        if 'drive.google.com' in MODEL_URL:
            # Download from Google Drive using gdown
            import gdown
            
            # Extract file ID from URL
            if 'uc?id=' in MODEL_URL:
                file_id = MODEL_URL.split('uc?id=')[1]
            else:
                file_id = MODEL_URL.split('/d/')[1].split('/')[0]
            
            # Alternative direct download URL
            direct_url = f'https://drive.google.com/uc?id={file_id}&export=download'
            
            # Try gdown first
            print("Using gdown to download...")
            gdown.download(direct_url, MODEL_PATH, quiet=False)
            
            # Verify file was downloaded
            if os.path.exists(MODEL_PATH):
                file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
                print(f"Model downloaded: {file_size:.2f} MB")
            else:
                raise Exception("Download failed - file not created")
            
        else:
            # Download from direct URL
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
            
            print(f"\n✓ Model downloaded to {MODEL_PATH}")
            
    except Exception as e:
        print(f" Error downloading model: {e}")
        print("Trying alternative download method...")
        
        # Alternative: Use requests with cookies for Google Drive
        try:
            file_id = '1NeGIjmkg_OroQerpVnlho5nKWqf51oI2'
            session = requests.Session()
            
            # First, get the confirmation page
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = session.get(url, stream=True)
            
            # Handle Google Drive's virus scan warning
            for key, value in response.cookies.items():
                if 'download_warning' in key:
                    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
                    response = session.get(url, stream=True)
                    break
            
            # Download the file
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
            
            print(f"\n✓ Model downloaded via alternative method")
            
        except Exception as e2:
            print(f" All download methods failed: {e2}")
            raise
    
    return MODEL_PATH

# ====== LOAD MODEL ======
print("=" * 50)
print("Starting model loading process...")
print("=" * 50)

MODEL_PATH = download_model()  # Download if needed

print("Loading model weights...")
start_time = time.time()

checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = checkpoint["class_names"]
num_classes = len(class_names)

model = build_model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

load_time = time.time() - start_time
print(f"Model loaded successfully in {load_time:.2f} seconds!")
print(f"Number of classes: {num_classes}")
print(f"Device: {device}")
print("=" * 50)

# ====== REST OF YOUR CODE ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def clean_label(label):
    label = label.replace("___", " – ")
    label = label.replace("__", " ")
    label = label.replace("_", " ")
    return label

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    raw_label = class_names[pred.item()]
    cleaned_label = clean_label(raw_label)

    return cleaned_label, confidence.item()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

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

# Health check endpoint
@app.route("/health")
def health():
    return {"status": "healthy", "model_loaded": True}, 200

# Simple test endpoint
@app.route("/test")
def test():
    return {"message": "Flask app is running", "model_ready": True}

if __name__ == "__main__":
    print(" Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=False)