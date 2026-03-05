import sys
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.afm_classifier import AFMClassifier
from src.training.afm_dataloader import create_dataloader


class Evaluator:

    def __init__(self, model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AFMClassifier()

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.to(self.device)

        self.val_loader = create_dataloader("data/dataset/val")

    def evaluate(self):

        self.model.eval()

        preds = []
        labels = []

        with torch.no_grad():

            for images in self.val_loader:

                images = images.to(self.device)

                outputs = self.model(images)

                predicted = torch.argmax(outputs, dim=1)

                preds.extend(predicted.cpu().numpy())

                labels.extend([0] * len(predicted))

        acc = accuracy_score(labels, preds)

        cm = confusion_matrix(labels, preds)

        print("accuracy")
        print(acc)

        print("confusion matrix")
        print(cm)