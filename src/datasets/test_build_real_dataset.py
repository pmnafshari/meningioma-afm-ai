import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


curve_dir = Path("data/curve_images")
dataset_dir = Path("data/dataset")

labels = pd.read_csv("labels.csv")

dataset_dir.mkdir(parents=True, exist_ok=True)

images = list(curve_dir.glob("*.png"))

data = []

for img in images:

    name = img.stem

    # گرفتن sample id
    sample = name.split("_")[0]

    data.append({
        "path": img,
        "sample": sample
    })

df = pd.DataFrame(data)

df = df.merge(labels, on="sample", how="inner")

train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

splits = {
    "train": train,
    "val": val,
    "test": test
}

for split_name, split_df in splits.items():

    for _, row in split_df.iterrows():

        label = row["label"]

        dest_dir = dataset_dir / split_name / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(row["path"], dest_dir)

print("dataset built")