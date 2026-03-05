import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.evaluation.evaluate_model import Evaluator


evaluator = Evaluator("results/model.pth")

evaluator.evaluate()