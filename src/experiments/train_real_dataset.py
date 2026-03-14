import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Transforms
# -------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])


# -------------------------
# Dataset
# -------------------------

train_dataset = datasets.ImageFolder(
    "data/dataset/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "data/dataset/val",
    transform=val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)


num_classes = len(train_dataset.classes)

print("classes:", train_dataset.classes)
print("train size:", len(train_dataset))
print("val size:", len(val_dataset))


# -------------------------
# Model
# -------------------------

model = models.resnet50(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

criterion = nn.CrossEntropyLoss()


# -------------------------
# Training
# -------------------------

epochs = 20

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    train_accuracy = 100 * correct / total


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()


    val_accuracy = 100 * val_correct / val_total


    print("epoch:", epoch)
    print("train loss:", total_loss)
    print("train accuracy:", train_accuracy)
    print("val accuracy:", val_accuracy)
    print("----------------------------------")


print("training finished")


# -------------------------
# Save model
# -------------------------

Path("models").mkdir(exist_ok=True)

torch.save(model.state_dict(), "models/afm_resnet50.pth")

print("model saved")