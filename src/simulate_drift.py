# Step 1: Create Drift Dataset
from pathlib import Path

import cv2
import numpy as np

INPUT_DIR = Path(
    "data/casting_data/casting_data/test"
)

OUTPUT_DIR = Path(
    "data/drifted_test"
)

def add_brightness(image):
    return cv2.convertScaleAbs(
        image, alpha=0.7, beta=-40
    )

def add_blur(image):
    return cv2.GaussianBlur(
        image, (11, 11), 0
    )

def add_noise(image):
    noise = np.random.normal(0, 25, image.shape)

    noisy = image + noise

    return np.clip(
        noisy, 0, 255
    ).astype(np.uint8)

def process_folder():
    for class_dir in INPUT_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        output_class_dir = OUTPUT_DIR / class_dir.name
        output_class_dir.mkdir(
            parents=True, exist_ok=True
        )

        for img_path in class_dir.glob("*.jpeg"):
            image = cv2.imread(str(img_path))

            image = add_brightness(image)
            image = add_blur(image)
            image = add_noise(image)

            save_path = (output_class_dir/img_path.name)

            cv2.imwrite(str(save_path), image)

    print(f"Saved drifted dataset to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_folder()  
