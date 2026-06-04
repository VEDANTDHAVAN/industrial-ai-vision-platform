from pathlib import Path
from typing import Optional, Callable, Any

from PIL import Image
from torch.utils.data import Dataset

class DriftDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue

            label = 0 if "ok" in class_dir.name.lower() else 1

            for ext in ("*.jpeg", "*.jpg", "*.png"):
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label