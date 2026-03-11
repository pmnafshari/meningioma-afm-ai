import sys
from pathlib import Path
import torch
import cv2
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.afm_classifier import AFMClassifier
from src.evaluation.gradcam import GradCAM
from src.evaluation.gradcam_overlay import overlay_gradcam


model = AFMClassifier()

model.load_state_dict(torch.load("results/model.pth"))

model.eval()


image_path = "data/dataset/train/curve_0.png"

image = cv2.imread(image_path)

image = cv2.resize(image, (224,224))

image = image / 255.0

image_tensor = torch.tensor(image).permute(2,0,1).float().unsqueeze(0)


target_layer = model.backbone.layer4

cam = GradCAM(model, target_layer)

heatmap = cam.generate(image_tensor, class_idx=0)


overlay_gradcam(
    image_path,
    heatmap,
    "results/gradcam_overlay.png"
)