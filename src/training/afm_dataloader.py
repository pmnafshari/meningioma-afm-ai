import torch
from torch.utils.data import DataLoader

from src.datasets.afm_image_dataset import AFMImageDataset


def create_dataloader(dataset_path, batch_size=16, augment=False):

    dataset = AFMImageDataset(dataset_path, augment=augment)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader