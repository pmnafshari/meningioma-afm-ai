import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_factory import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_dir = Path("experiments/final_results")
results_dir.mkdir(parents=True, exist_ok=True)


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


test_dataset = datasets.ImageFolder(
    "data/dataset/test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


num_classes = len(test_dataset.classes)

model = get_model("resnet50", num_classes)

model.load_state_dict(
    torch.load("models/best_model_phase7.pth", map_location=device)
)

model = model.to(device)

model.eval()


all_preds = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())


cm = confusion_matrix(all_labels, all_preds)

report = classification_report(
    all_labels,
    all_preds,
    target_names=test_dataset.classes
)


print("confusion matrix")
print(cm)

print()
print("classification report")
print(report)


# save confusion matrix
cm_df = pd.DataFrame(cm)
cm_df.to_csv(results_dir / "confusion_matrix.csv", index=False)


# save classification report
with open(results_dir / "classification_report.txt", "w") as f:
    f.write(report)


# save predictions
pred_df = pd.DataFrame({
    "true_label": all_labels,
    "predicted_label": all_preds
})

pred_df.to_csv(results_dir / "predictions.csv", index=False)


print()
print("results saved in experiments/final_results")