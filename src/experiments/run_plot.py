import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.experiments.plot_experiments import ExperimentPlotter


plotter = ExperimentPlotter()

plotter.plot_loss()