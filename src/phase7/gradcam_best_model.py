import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_factory import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


output_dir = Path("experiments/gradcam_results")
output_dir.mkdir(parents=True, exist_ok=True)


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


dataset = datasets.ImageFolder(
    "data/dataset/test",
    transform=transform
)


num_classes = len(dataset.classes)

model = get_model("resnet50", num_classes)

model.load_state_dict(
    torch.load("models/best_model_phase7.pth", map_location=device)
)

model = model.to(device)

model.eval()


target_layer = model.layer4[-1]


features = []
gradients = []


def forward_hook(module, input, output):

    features.append(output)


def backward_hook(module, grad_in, grad_out):

    gradients.append(grad_out[0])


target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)


def generate_gradcam(image_tensor):

    features.clear()
    gradients.clear()

    output = model(image_tensor)

    class_idx = torch.argmax(output)

    loss = output[:, class_idx]

    model.zero_grad()

    loss.backward()

    grad = gradients[0]

    fmap = features[0]

    weights = torch.mean(grad, dim=(2,3), keepdim=True)

    cam = torch.sum(weights * fmap, dim=1).squeeze()

    cam = torch.relu(cam)

    cam = cam.detach().cpu().numpy()

    cam = cv2.resize(cam, (224,224))

    cam = cam - np.min(cam)

    cam = cam / np.max(cam)

    return cam


for i in range(10):

    img_path, label = dataset.samples[i]

    image = Image.open(img_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    cam = generate_gradcam(image_tensor)

    image_np = np.array(image.resize((224,224)))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    overlay = heatmap * 0.4 + image_np

    save_path = output_dir / f"gradcam_{i}.png"

    cv2.imwrite(str(save_path), overlay)

    print("saved:", save_path)