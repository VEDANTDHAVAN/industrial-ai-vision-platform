import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import DataLoader

from dataset import CastingBinaryDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data/casting_data/casting_data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    return model.to(DEVICE)

def train():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = CastingBinaryDataset(
        root_dir=DATA_DIR,
        split="train",
        transform=train_transform
    )

    test_dataset = CastingBinaryDataset(
        root_dir=DATA_DIR,
        split="test",
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, shuffle=True,
        batch_size=BATCH_SIZE, 
    )

    test_loader = DataLoader(
        test_dataset, shuffle=False,
        batch_size=BATCH_SIZE,
    )

    model = build_model()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(f"Using device: {DEVICE}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(
                loss=train_loss / len(train_loader),
                acc= 100 * correct / total
            )

        print(f"Epoch {epoch + 1}: "
            f"Loss={train_loss / len(train_loader):.4f}, "
            f"Accuracy={100 * correct / total:.2f}%")
        
    model_path = MODEL_DIR / "defect_resnet18.pth"

    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()