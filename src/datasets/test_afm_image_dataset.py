import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.datasets.afm_image_dataset import AFMImageDataset


dataset = AFMImageDataset("data/dataset/train")

print("dataset size")
print(len(dataset))

image = dataset[0]

print("image tensor shape")
print(image.shape)