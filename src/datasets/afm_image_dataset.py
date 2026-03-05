import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2


class AFMImageDataset(Dataset):

    def __init__(self, dataset_dir):

        self.dataset_dir = Path(dataset_dir)

        self.images = list(self.dataset_dir.glob("*.png"))

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]

        image = cv2.imread(str(img_path))

        image = cv2.resize(image, (224, 224))

        image = image / 255.0

        image = torch.tensor(image).permute(2,0,1).float()

        return image