import torch
import torch.nn as nn
import torchvision.models as models


class AFMClassifier(nn.Module):

    def __init__(self,
                 model_name="resnet18",
                 num_classes=2,
                 pretrained=True,
                 freeze_backbone=False):

        super().__init__()

        if model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif model_name == "efficientnet":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError("model not supported")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        features = self.backbone(x)

        out = self.classifier(features)

        return out