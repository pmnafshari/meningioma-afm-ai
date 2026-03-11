import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import torchvision.transforms as transforms


class AFMImageDataset(Dataset):

    def __init__(self, dataset_dir, augment=False):

        self.dataset_dir = Path(dataset_dir)

        self.images = list(self.dataset_dir.glob("*.png"))

        if augment:

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])

        else:

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]

        image = cv2.imread(str(img_path))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)

        return image