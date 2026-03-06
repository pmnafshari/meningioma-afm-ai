import csv
from pathlib import Path


class ExperimentAnalyzer:

    def __init__(self, file_path="results/experiments.csv"):

        self.file_path = Path(file_path)

    def load_results(self):

        results = []

        with open(self.file_path, "r") as f:

            reader = csv.DictReader(f)

            for row in reader:

                row["final_loss"] = float(row["final_loss"])

                results.append(row)

        return results

    def best_model(self):

        results = self.load_results()

        best = min(results, key=lambda x: x["final_loss"])

        return best

    def ranking(self):

        results = self.load_results()

        results = sorted(results, key=lambda x: x["final_loss"])

        return results