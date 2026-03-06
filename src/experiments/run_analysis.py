import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.experiments.analyze_experiments import ExperimentAnalyzer


analyzer = ExperimentAnalyzer()

best = analyzer.best_model()

print("best model")
print(best)

print()

print("model ranking")

ranking = analyzer.ranking()

for r in ranking:

    print(r)