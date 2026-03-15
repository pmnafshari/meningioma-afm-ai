import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_factory import get_model
from experiment_logger import ExperimentLogger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    total = 0
    correct = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def evaluate(model, loader):

    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def main():

    print("phase 7 experiment started")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        "data/dataset/train",
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        "data/dataset/val",
        transform=transform
    )

    num_classes = len(train_dataset.classes)

    logger = ExperimentLogger()

    model_names = [
    "resnet50",
    "efficientnet",
    "vit",
    "cnn_lstm",
    "afm_net"
]

    learning_rates = [0.001, 0.0003, 0.0001]

    batch_sizes = [16, 32]

    epochs = 3


    for model_name in model_names:

        for lr in learning_rates:

            for batch_size in batch_sizes:

                print("running experiment")
                print("model:", model_name)
                print("lr:", lr)
                print("batch:", batch_size)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False
                )

                model = get_model(model_name, num_classes)

                model = model.to(device)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=lr
                )

                criterion = torch.nn.CrossEntropyLoss()

                for epoch in range(epochs):

                    train_acc = train_one_epoch(
                        model,
                        train_loader,
                        optimizer,
                        criterion
                    )

                val_acc = evaluate(model, val_loader)

                logger.log(
                    model_name,
                    lr,
                    batch_size,
                    "adam",
                    val_acc
                )

                print("train accuracy:", train_acc)
                print("val accuracy:", val_acc)
                print("------------------------")


if __name__ == "__main__":
    main()