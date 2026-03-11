import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.training.train_model import Trainer


class CrossValidator:

    def __init__(self, k=5):

        self.k = k

    def run(self):

        print("starting cross validation")

        kfold = KFold(n_splits=self.k, shuffle=True)

        results = []

        for fold in range(self.k):

            print("fold", fold)

            trainer = Trainer(
                model_name="resnet18",
                epochs=3
            )

            losses = trainer.train()

            results.append(losses[-1])

        print("cross validation results")
        print(results)

        print("average loss")
        print(np.mean(results))