import torch
from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image
import os

from model import build_model


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_PATH = "models/plant_disease_classifier.pth"

checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = checkpoint["class_names"]
num_classes = len(class_names)

model = build_model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


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



if __name__ == "__main__":
    app.run(debug=True)
