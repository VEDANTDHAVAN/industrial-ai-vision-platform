import torch
import torch.nn as nn
from typing import cast
from pathlib import Path
from torchvision import models, transforms
from torch.utils.data import DataLoader
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT / "src"))
from dataset import CastingBinaryDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("models/best_v6_efficientnet.pth")
DATA_DIR = Path("data/casting_data/casting_data")
DRIFTED_DATA_DIR = Path("data/")

def build_model():
    model = models.efficientnet_b0(
        weights=None
    )

    classifier_layer = cast(
        nn.Linear, model.classifier[1]
    )

    in_features = classifier_layer.in_features

    model.classifier[1] = nn.Linear(
        in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

def analyze():
    drifted_dataset = CastingBinaryDataset(
        root_dir=DRIFTED_DATA_DIR, split="drifted_test", transform=get_transform()
    )

    loader = DataLoader(drifted_dataset, batch_size=32, shuffle=False)
    model = build_model()

    correct_confidences = []
    wrong_confidences = []

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

            for i in range(len(labels)):
                total += 1

                if predicted[i] == labels[i]:
                    correct += 1
                    correct_confidences.append(
                        confidence[i].item()
                    )
                else:
                    wrong_confidences.append(
                        confidence[i].item()
                    )
    
    accuracy = correct / total
    print("\n=== Confidence Report ===\n")
    print("Accuracy: ", round(accuracy, 4))
    print("Correct Predictions: ", len(correct_confidences))

    print("Wrong Predictions: ", len(wrong_confidences))
    print("\nAverage Confidence (Correct):",
       round(sum(correct_confidences) / len(correct_confidences), 4)   
    )

    if wrong_confidences:
        print(
            "Average Confidence (wrong):",
            round(sum(wrong_confidences)/len(wrong_confidences), 4)
        )

if __name__ == "__main__":
    analyze()