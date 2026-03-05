import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.datasets.split_dataset import DatasetSplitter


splitter = DatasetSplitter()

train_count, val_count, test_count = splitter.split()

print("train images", train_count)
print("validation images", val_count)
print("test images", test_count)