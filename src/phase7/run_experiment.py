import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_factory import get_model
from experiment_logger import ExperimentLogger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(model, train_loader, optimizer, criterion):

    model.train()

    total = 0
    correct = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)

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

    num_classes = len(train_dataset.classes)

    logger = ExperimentLogger()

    models_list = ["resnet18", "resnet50", "efficientnet"]

    learning_rates = [0.001, 0.0003, 0.0001]

    batch_sizes = [16, 32]


    for model_name in models_list:

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

                model = get_model(model_name, num_classes)

                model = model.to(device)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=lr
                )

                criterion = torch.nn.CrossEntropyLoss()

                accuracy = run_training(
                    model,
                    train_loader,
                    optimizer,
                    criterion
                )

                logger.log(
                    model_name,
                    lr,
                    batch_size,
                    "adam",
                    accuracy
                )

                print("accuracy:", accuracy)
                print("-------------------")


if __name__ == "__main__":
    main()