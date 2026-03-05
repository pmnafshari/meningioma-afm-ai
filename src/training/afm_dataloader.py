import torch
from torch.utils.data import DataLoader

from src.datasets.afm_image_dataset import AFMImageDataset


def create_dataloader(dataset_path, batch_size=16):

    dataset = AFMImageDataset(dataset_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader