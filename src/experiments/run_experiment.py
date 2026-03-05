import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.training.train_model import Trainer


def run_experiment(config):

    print("starting experiment")

    print("model", config["model_name"])
    print("learning rate", config["learning_rate"])
    print("batch size", config["batch_size"])
    print("epochs", config["epochs"])

    trainer = Trainer(
        model_name=config["model_name"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        epochs=config["epochs"]
    )

    trainer.train()