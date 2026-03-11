import torch
import cv2
import numpy as np
from torchvision import models


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):

        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    def generate(self, image, class_idx):

        output = self.model(image)

        self.model.zero_grad()

        loss = output[:, class_idx]

        loss.backward()

        weights = torch.mean(self.gradients, dim=(2,3), keepdim=True)

        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(cam, (224,224))

        cam = cam - cam.min()

        cam = cam / cam.max()

        return cam