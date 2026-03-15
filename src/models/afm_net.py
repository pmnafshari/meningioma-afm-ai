import torch
import torch.nn as nn
import torchvision.models as models


class AFMNet(nn.Module):

    def __init__(self, num_classes):

        super(AFMNet, self).__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Linear(128, num_classes)


    def forward(self, x):

        features = self.feature_extractor(x)

        attn = self.attention(features)

        features = features * attn

        pooled = self.pool(features)

        pooled = pooled.view(pooled.size(0), 1, -1)

        gru_out, _ = self.gru(pooled)

        output = self.classifier(gru_out[:, -1, :])

        return output