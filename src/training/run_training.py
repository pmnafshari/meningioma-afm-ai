import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.training.train_model import Trainer


trainer = Trainer(
    model_name="resnet18",
    num_classes=2,
    learning_rate=0.001,
    epochs=3,
    batch_size=16
)

trainer.train()