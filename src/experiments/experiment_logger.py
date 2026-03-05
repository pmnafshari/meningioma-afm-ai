import csv
from pathlib import Path


class ExperimentLogger:

    def __init__(self, file_path="results/experiments.csv"):

        self.file_path = Path(file_path)

        if not self.file_path.exists():

            with open(self.file_path, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "model",
                    "learning_rate",
                    "batch_size",
                    "epochs",
                    "final_loss"
                ])

    def log(self, model, learning_rate, batch_size, epochs, final_loss):

        with open(self.file_path, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                model,
                learning_rate,
                batch_size,
                epochs,
                final_loss
            ])