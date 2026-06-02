import torch
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from torchvision import models, transforms
from torch.utils.data import DataLoader

from dataset import CastingBinaryDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data/casting_data/casting_data")
MODEL_PATH = Path("models/defect_resnet18.pth")

def build_model():
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features

    model.fc = torch.nn.Linear(in_features, 2)

    model.load_state_dict(
        torch.load(
            MODEL_PATH, map_location=DEVICE
        )
    )

    model.to(DEVICE)

    return model

def evaluate():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = CastingBinaryDataset(
        root_dir=DATA_DIR,
        split="test",
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    model = build_model()

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs, dim=1
            )

            y_true.extend(labels.numpy())

            y_pred.extend(predictions.cpu().numpy())

    print("\nAccuracy")
    print(
        accuracy_score(y_true, y_pred)
    )

    print("\nPrecision")
    print(
        precision_score(y_true, y_pred)
    )

    print("\nRecall")
    print(
        recall_score(y_true, y_pred)
    )

    print("\nF1")
    print(
        f1_score(y_true, y_pred)
    )

    print("\nConfusion Matrix")
    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred
        )
    )

if __name__ == "__main__":
    evaluate()