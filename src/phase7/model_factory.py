import torchvision.models as models
import torch.nn as nn


def get_model(model_name, num_classes):

    if model_name == "resnet18":

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":

        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet":

        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:

        raise ValueError("unknown model")

    return model