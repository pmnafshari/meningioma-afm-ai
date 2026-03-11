import sys
from pathlib import Path
import torch
import cv2
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.afm_classifier import AFMClassifier
from src.evaluation.gradcam import GradCAM


model = AFMClassifier()

model.load_state_dict(torch.load("results/model.pth"))

model.eval()


image = cv2.imread("data/dataset/train/curve_0.png")

image = cv2.resize(image, (224,224))

image = image / 255.0

image = torch.tensor(image).permute(2,0,1).float().unsqueeze(0)


target_layer = model.backbone.layer4

cam = GradCAM(model, target_layer)

heatmap = cam.generate(image, class_idx=0)

heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

cv2.imwrite("results/gradcam.png", heatmap)