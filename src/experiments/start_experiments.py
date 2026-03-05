import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.experiments.run_experiment import run_experiment


experiments = [

    {
        "model_name": "resnet18",
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 3
    },

    {
        "model_name": "resnet50",
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 3
    },

    {
        "model_name": "efficientnet",
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 3
    }

]


for exp in experiments:

    run_experiment(exp)