import sys
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.afm_classifier import AFMClassifier


model = AFMClassifier(
    model_name="resnet18",
    num_classes=2,
    pretrained=True,
    freeze_backbone=False
)

x = torch.randn(4,3,224,224)

y = model(x)

print("model output shape")
print(y.shape)