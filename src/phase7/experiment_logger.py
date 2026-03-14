import csv
from pathlib import Path

class ExperimentLogger:

    def __init__(self, file_path="experiments/results.csv"):

        self.file_path = Path(file_path)

        if not self.file_path.exists():

            with open(self.file_path, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "model",
                    "learning_rate",
                    "batch_size",
                    "optimizer",
                    "accuracy"
                ])


    def log(self, model, lr, batch, optimizer, accuracy):

        with open(self.file_path, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                model,
                lr,
                batch,
                optimizer,
                accuracy
            ])