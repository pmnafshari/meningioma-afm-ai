import torch
import torch.nn as nn
import torchvision.models as models


# ----------------------------------
# CNN + LSTM Hybrid Model
# ----------------------------------

class CNNLSTM(nn.Module):

    def __init__(self, num_classes=3):
        super(CNNLSTM, self).__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):

        batch_size = x.size(0)

        features = self.cnn(x)

        features = features.view(batch_size, 1, -1)

        lstm_out, _ = self.lstm(features)

        out = self.fc(lstm_out[:, -1, :])

        return out


# ----------------------------------
# Model Factory
# ----------------------------------

def get_model(model_name, num_classes):

    if model_name == "resnet50":

        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model


    elif model_name == "efficientnet":

        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

        return model


    elif model_name == "vit":

        model = models.vit_b_16(
            weights=models.ViT_B_16_Weights.DEFAULT
        )

        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            num_classes
        )

        return model


    elif model_name == "cnn_lstm":

        return CNNLSTM(num_classes)


    else:

        raise ValueError("model not supported")