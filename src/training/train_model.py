import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.afm_classifier import AFMClassifier
from src.training.afm_dataloader import create_dataloader


class Trainer:

    def __init__(
        self,
        model_name="resnet18",
        num_classes=2,
        learning_rate=0.001,
        epochs=5,
        batch_size=16
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AFMClassifier(
            model_name=model_name,
            num_classes=num_classes
        ).to(self.device)

        self.epochs = epochs

        self.train_loader = create_dataloader(
            "data/dataset/train",
            batch_size=batch_size
        )

        self.val_loader = create_dataloader(
            "data/dataset/val",
            batch_size=batch_size
        )

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

    def train_epoch(self):

        self.model.train()

        total_loss = 0

        for images in self.train_loader:

            images = images.to(self.device)

            labels = torch.zeros(images.size(0), dtype=torch.long).to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def train(self):

        final_loss = None

        for epoch in range(self.epochs):

            loss = self.train_epoch()

            final_loss = loss

            print("epoch", epoch)
            print("train loss", loss)

        torch.save(self.model.state_dict(), "results/model.pth")

        return final_loss
