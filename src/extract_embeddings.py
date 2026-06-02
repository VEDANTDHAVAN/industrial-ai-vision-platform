import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm

from torchvision import models, transforms
from torch.utils.data import DataLoader

from dataset import CastingBinaryDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data/casting_data/casting_data")
MODEL_PATH = Path("models/defect_resnet18.pth")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

""" Load ResNet without Final Layer
why?
We don't want: Good / Defective
We want: 512-dimensional representation
"""
def build_feature_extractor():
    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features, 2
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH, map_location=DEVICE
        )
    )

    feature_extractor = torch.nn.Sequential(
        *list(model.children())[:-1]
    )

    feature_extractor.to(DEVICE)
    feature_extractor.eval()

    return feature_extractor

"""
Why This Works

Original:
Image → ResNet → 512 Features → Classification Layer → 2 Classes

After removing final layer:
Image → ResNet → 512 Features
"""

# Dataset Loader
def get_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = CastingBinaryDataset(
        root_dir=DATA_DIR,
        split="test",
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    return loader

# Extract Embeddings
def extract_embeddings():
    loader = get_loader()

    model = build_feature_extractor()

    embeddings = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in tqdm(loader):
            images = images.to(DEVICE)
            features = model(images)

            features = features.squeeze(-1).squeeze(-1)

            embeddings.append(
                features.cpu().numpy()
            )
            
            labels.append(
                batch_labels.numpy()
            )

    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)

    print("Embedding Shape:", embeddings.shape)
    print("Labels Shape:", labels.shape)

    np.save(
        OUTPUT_DIR/"embeddings.npy",
        embeddings
    )

    np.save(
        OUTPUT_DIR/"labels.npy",
        labels
    )

    print("Saved embeddings.")

# Main Function
if __name__ == "__main__":
    extract_embeddings()