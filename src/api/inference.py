import torch
import torch.nn as nn
from torch import Tensor
from pathlib import Path
from PIL import Image
from typing import cast
from torchvision import models, transforms

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path("models/best_v4_model.pth")

def build_model():
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Linear(in_features, 2)

    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=DEVICE,
    ))

    model.to(DEVICE)
    model.eval()

    return model

model = build_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image: Image.Image):
    image_tensor = cast(Tensor, transform(image))

    image_tensor = (image_tensor.unsqueeze(0).to(DEVICE))

    with torch.no_grad():
        output = model(image_tensor)

        probabilities = (torch.softmax(output, dim=1))

        ok_prob = probabilities[0][0].item()
        defect_prob = probabilities[0][1].item()

        confidence, predicted = (torch.max(probabilities, dim=1))

    prediction = ("OK" if predicted.item() == 0
        else "Defective"
    )
    
    if confidence.item() < 0.95:
        prediction = "MANUAL_REVIEW"

    return {
        "prediction": prediction,
        "confidence": round(confidence.item(), 4),
        "ok_probability": round(ok_prob, 4),
        "defect_probability": round(defect_prob, 4),
        "model_version": "v4"
    }