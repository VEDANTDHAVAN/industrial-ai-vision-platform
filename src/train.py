import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from mlflow import pytorch as mlflow_pytorch

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from torchvision import models, transforms
from torch.utils.data import DataLoader

from dataset import CastingBinaryDataset
from utils.early_stopping import EarlyStopping

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data/casting_data/casting_data")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 5e-5


def build_model():
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        2
    )

    return model.to(DEVICE)


def get_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(10),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3
            )
        ], p=0.3),

        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=3
            )
        ], p=0.2),

        transforms.RandomApply([
            transforms.RandomAdjustSharpness(
                sharpness_factor=0.7
            )
        ], p=0.2),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


def build_datasets():

    train_transform, eval_transform = get_transforms()

    base_dataset = CastingBinaryDataset(
        root_dir=DATA_DIR,
        split="train",
        transform=None
    )

    train_samples, val_samples = train_test_split(
        base_dataset.samples,
        test_size=0.2,
        random_state=42,
        stratify=[
            label
            for _, label in base_dataset.samples
        ]
    )

    train_dataset = CastingBinaryDataset(
        root_dir=DATA_DIR,
        split="train",
        transform=train_transform,
        samples=train_samples
    )

    val_dataset = CastingBinaryDataset(
        root_dir=DATA_DIR,
        split="train",
        transform=eval_transform,
        samples=val_samples
    )

    test_dataset = CastingBinaryDataset(
        root_dir=DATA_DIR,
        split="test",
        transform=eval_transform
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


def evaluate(
    model,
    loader,
    criterion
):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    avg_loss = total_loss / len(loader)

    accuracy = (
        100 * correct / total
    )

    return avg_loss, accuracy


def train():

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = build_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    model = build_model()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    early_stopping = EarlyStopping(
        patience=5
    )

    best_val_loss = float("inf")

    mlflow.set_experiment(
        "Casting_Defect_Detection"
    )

    with mlflow.start_run(
        run_name="resnet18_v3_augmentation"
    ):
        mlflow.log_params({
            "model": "resnet18",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset)
        })

        print(f"\nUsing device: {DEVICE}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        for epoch in range(EPOCHS):
            model.train()

            running_loss = 0
            correct = 0
            total = 0

            loop = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{EPOCHS}"
            )

            for images, labels in loop:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(
                    outputs, labels
                )

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(
                    outputs, 1
                    )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

                loop.set_postfix(
                    loss=running_loss / (loop.n + 1),
                    acc=100 * correct / total
                )

            train_loss = (
                running_loss / len(train_loader)
            )

            train_acc = (
                100 * correct / total
            )

            val_loss, val_acc = evaluate(
                model, val_loader, criterion
            )

            print(f"\nEpoch {epoch+1}")

            print(f"Train Loss: {train_loss:.4f}")

            print(f"Train Acc: {train_acc:.2f}%")

            print(f"Val Loss: {val_loss:.4f}")

            print(f"Val Acc: {val_acc:.2f}%")

            mlflow.log_metrics({
                "train_loss": train_loss, "train_accuracy": train_acc,
                "val_loss": val_loss, "val_accuracy": val_acc,
            }, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                model_path = (MODEL_DIR / "best_v4_model.pth")

                torch.save(model.state_dict(), model_path)

                mlflow.log_metric("best_val_loss", best_val_loss)

                print("Best model saved.")

            if early_stopping.step(val_loss):
                print("\nEarly stopping triggered.")
                break

        print(f"\nTraining complete.")

        print(f"Best validation loss: "
              f"{best_val_loss:.4f}"
        )

        print(f"Model saved at: "
              f"{MODEL_DIR/'best_v4_model.pth'}"
        )
        print("Logging checkpoint artifact...")
        mlflow.log_artifact(str(MODEL_DIR / "best_v4_model.pth"))

        print("Logging PyTorch model...") 
        logged_model = mlflow_pytorch.log_model(
            pytorch_model=model, name="casting_defect_model"
        )

        print("Registering model...")
        mlflow.register_model(
            model_uri=logged_model.model_uri,
            name="CastingDefectDetector"
        )

        print("Model registered successfully.")


if __name__ == "__main__":
    train()