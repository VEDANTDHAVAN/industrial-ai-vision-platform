import torch
import torch.nn as nn
from pathlib import Path
from typing import cast
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score,
    recall_score, confusion_matrix
)

from torchvision import models, transforms
from torch.utils.data import DataLoader
from drift_dataset import DriftDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data/drifted_test")
MODEL_PATH = Path("models/best_v7_focal_loss.pth")

def build_model():
    model = models.efficientnet_b0(weights=None)

    classifier_layer = cast(
        nn.Linear, model.classifier[1]
    )

    in_features = classifier_layer.in_features

    model.classifier[1] = nn.Linear(in_features, 2)

    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=DEVICE
    ))

    model.to(DEVICE)
    model.eval()

    return model

def evaluate_drift():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = DriftDataset(
        root_dir=DATA_DIR,
        transform=transform
    )

    loader = DataLoader(
        dataset, batch_size=32, shuffle=False
    )

    model = build_model()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.cpu().numpy())

    print("\nAccuracy:", accuracy_score(y_true, y_pred))
    print("\nPrecision:", precision_score(y_true, y_pred))
    print("\nRecall:", recall_score(y_true, y_pred))
    print("\nF1:", f1_score(y_true, y_pred))
    print("\nConfusion Matrix:", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate_drift()

