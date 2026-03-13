import shutil
import pandas as pd
from pathlib import Path
import random


class AFMRealDatasetBuilder:

    def __init__(self):

        self.images_dir = Path("data/curve_images")
        self.labels_file = Path("labels.csv")
        self.dataset_dir = Path("data/dataset")

        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15

        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def load_labels(self):

        df = pd.read_csv(self.labels_file)

        label_map = {}

        for _, row in df.iterrows():

            label_map[row["sample"]] = row["label"]

        return label_map


    def collect_images(self, label_map):

        dataset = []

        for img in self.images_dir.glob("*.png"):

            name = img.stem

            sample = name.split("_")[0]

            if sample in label_map:

                dataset.append((img, label_map[sample]))

        return dataset


    def split_dataset(self, dataset):

        random.shuffle(dataset)

        n = len(dataset)

        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train = dataset[:train_end]
        val = dataset[train_end:val_end]
        test = dataset[val_end:]

        return train, val, test


    def save_split(self, split, split_name):

        for img, label in split:

            save_dir = self.dataset_dir / split_name / label

            save_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy(img, save_dir / img.name)


    def build(self):

        label_map = self.load_labels()

        dataset = self.collect_images(label_map)

        train, val, test = self.split_dataset(dataset)

        self.save_split(train, "train")
        self.save_split(val, "val")
        self.save_split(test, "test")

        print("dataset built")