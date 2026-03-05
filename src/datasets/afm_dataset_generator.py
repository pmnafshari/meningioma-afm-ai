from pathlib import Path


class AFMDatasetGenerator:

    def __init__(self, raw_data_dir="data/raw_afm", output_dir="data/dataset"):

        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_afm_files(self):

        files = list(self.raw_data_dir.glob("*.h5"))
        return files

    def create_dataset_structure(self):

        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"
        test_dir = self.output_dir / "test"

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        return train_dir, val_dir, test_dir