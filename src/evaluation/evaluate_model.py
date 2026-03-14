import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

dataset = datasets.ImageFolder("data/dataset/test", transform=transform)

loader = DataLoader(dataset, batch_size=16, shuffle=False)

num_classes = len(dataset.classes)

model = models.resnet50()

model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load("models/afm_resnet50.pth"))

model = model.to(device)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs,1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("confusion matrix")
print(confusion_matrix(all_labels, all_preds))

print("\nclassification report")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))