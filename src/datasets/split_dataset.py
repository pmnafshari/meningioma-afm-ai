import random
import shutil
from pathlib import Path


class DatasetSplitter:

    def __init__(self, dataset_dir="data/dataset"):

        self.dataset_dir = Path(dataset_dir)

        self.train_dir = self.dataset_dir / "train"
        self.val_dir = self.dataset_dir / "val"
        self.test_dir = self.dataset_dir / "test"

    def split(self, train_ratio=0.7, val_ratio=0.2):

        images = list(self.dataset_dir.glob("curve_*.png"))

        random.shuffle(images)

        total = len(images)

        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        for img in train_images:
            shutil.move(str(img), self.train_dir / img.name)

        for img in val_images:
            shutil.move(str(img), self.val_dir / img.name)

        for img in test_images:
            shutil.move(str(img), self.test_dir / img.name)

        return len(train_images), len(val_images), len(test_images)