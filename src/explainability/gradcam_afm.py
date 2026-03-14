import torch
import torchvision.models as models
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# load model
# ----------------------

model = models.resnet50()

model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load("models/afm_resnet50.pth"))

model = model.to(device)
model.eval()

# ----------------------
# image transform
# ----------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# ----------------------
# load sample image
# ----------------------

image_path = "/Users/hitson/Documents/Codes/meningioma-afm-ai/data/dataset/test/MENINGOTHELIAL_G1/753B_161.png"

image = Image.open(image_path).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(device)

# ----------------------
# GradCAM
# ----------------------

target_layer = model.layer4[-1]

cam = GradCAM(model=model, target_layers=[target_layer])

grayscale_cam = cam(input_tensor=input_tensor)[0]

# ----------------------
# visualize
# ----------------------

image_np = np.array(image.resize((224,224))).astype(np.float32) / 255

cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

cv2.imwrite("gradcam_result.png", cam_image)

print("GradCAM saved as gradcam_result.png")