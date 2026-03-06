import csv
from pathlib import Path
import matplotlib.pyplot as plt


class ExperimentPlotter:

    def __init__(self, file_path="results/experiments.csv"):

        self.file_path = Path(file_path)

    def load_results(self):

        models = []
        losses = []

        with open(self.file_path, "r") as f:

            reader = csv.DictReader(f)

            for row in reader:

                models.append(row["model"])
                losses.append(float(row["final_loss"]))

        return models, losses

    def plot_loss(self):

        models, losses = self.load_results()

        plt.figure()

        plt.bar(models, losses)

        plt.xlabel("model")
        plt.ylabel("final loss")

        plt.title("model comparison")

        plt.savefig("results/model_comparison.png")

        plt.show()