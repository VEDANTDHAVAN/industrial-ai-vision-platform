from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from typing import Any
from torch.utils.data import Dataset


class CastingBinaryDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        split_dir = self.root_dir / split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            label = 0 if "ok" in class_dir.name.lower() else 1

            for ext in ("*.jpeg", "*.jpg", "*.png"):
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, label))

        if not self.samples:
            raise ValueError(f"No images found in: {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label