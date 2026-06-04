import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, cast
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from torch import Tensor
from torchvision.models.resnet import ResNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("models/defect_resnet18.pth")
IMAGE_PATH = Path("data/casting_data/casting_data/test/def_front/cast_def_0_7.jpeg")
OUTPUT_PATH = Path("outputs/gradcam_result.png")


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        self.gradients: Optional[Tensor] = None
        self.activations: Optional[Tensor] = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output) -> None:
        self.activations = cast(Tensor, output).detach()

    def save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = cast(Tensor, grad_output[0]).detach()

    def generate(self, input_tensor: Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed. Gradients or activations are missing.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()

        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        return cam_np

def build_model() -> ResNet:
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def preprocess_image(image_path: Path) -> Tuple[Image.Image, Tensor]:
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image_tensor = cast(Tensor, transform(image))
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    return image, input_tensor


def overlay_heatmap(original_image: Image.Image, cam: np.ndarray) -> np.ndarray:
    original_image = original_image.resize((224, 224))
    original_np = np.asarray(original_image, dtype=np.uint8)

    heatmap = cv2.resize(cam.astype(np.float32), (224, 224))
    heatmap_uint8 = np.asarray(255*heatmap, dtype=np.uint8)

    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.6 * original_np + 0.4 * colored_heatmap).astype(np.uint8)

    return overlay


def main():
    model = build_model()

    original_image, input_tensor = preprocess_image(IMAGE_PATH)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = int(torch.argmax(probabilities, dim=1).item())

    class_name = "OK" if predicted_class == 0 else "Defective"
    confidence = float(probabilities[0][predicted_class].item())

    print("Prediction:", class_name)
    print("Confidence:", round(confidence, 4))

    target_layer = model.layer4[-1]

    gradcam = GradCAM(model, target_layer)

    cam = gradcam.generate(input_tensor, predicted_class)

    overlay = overlay_heatmap(original_image, cam)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image.resize((224, 224)))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Grad-CAM: {class_name}")
    plt.axis("off")

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.show()

    print(f"Saved Grad-CAM result to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()