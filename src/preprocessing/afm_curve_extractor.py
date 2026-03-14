import matplotlib
matplotlib.use("Agg")

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class AFMCurveExtractor:

    def __init__(self, input_dir="data/raw_afm/materials", output_dir="data/curve_images"):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)


    def extract_curves(self):

        h5_files = list(self.input_dir.rglob("*.h5"))

        for file_path in h5_files:

            sample = file_path.parent.name.replace(" ", "").replace("_","")
            self.process_file(file_path, sample)


    def process_file(self, file_path, sample):

        with h5py.File(file_path, "r") as f:

            for map_key in f.keys():

                group = f[map_key]

                if "force" not in group:
                    continue

                curve = np.array(group["force"])

                self.save_curve_image(curve, sample, map_key)


    def save_curve_image(self, curve, sample, index):

        plt.figure(figsize=(3,3))
        plt.plot(curve, color="black", linewidth=1)
        plt.axis("off")

        filename = f"{sample}_{index}.png".replace("__","_")

        save_path = self.output_dir / filename

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()