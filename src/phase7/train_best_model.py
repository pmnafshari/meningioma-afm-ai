import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_factory import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


train_dataset = datasets.ImageFolder(
    "data/dataset/train",
    transform=transform_train
)

val_dataset = datasets.ImageFolder(
    "data/dataset/val",
    transform=transform_val
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


num_classes = len(train_dataset.classes)

model = get_model("resnet50", num_classes)

model = model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003
)

criterion = torch.nn.CrossEntropyLoss()

epochs = 25

best_val_acc = 0


for epoch in range(epochs):

    model.train()

    total = 0
    correct = 0
    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    train_acc = correct / total


    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    val_acc = correct / total


    print("epoch:", epoch)
    print("train accuracy:", train_acc)
    print("val accuracy:", val_acc)
    print("--------------")


    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(model.state_dict(), "models/best_model_phase7.pth")

        print("best model saved")


print("training finished")