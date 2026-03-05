import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.training.afm_dataloader import create_dataloader


loader = create_dataloader("data/dataset/train")

for batch in loader:

    print("batch shape")
    print(batch.shape)

    break