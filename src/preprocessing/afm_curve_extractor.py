import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class AFMCurveExtractor:

    def __init__(self, input_dir="data/raw_afm", output_dir="data/curve_images"):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)


    def extract_curves(self):

        h5_files = list(self.input_dir.rglob("*.h5"))

        for file in h5_files:

            self.process_file(file)


    def process_file(self, file_path):

        with h5py.File(file_path, "r") as f:

            for key in f.keys():

                data = np.array(f[key])

                if data.ndim < 3:
                    continue

                maps = data.shape[0]
                points = data.shape[1]

                for m in range(maps):
                    for p in range(points):

                        curve = data[m, p, :, 0]

                        self.save_curve_image(
                            curve,
                            file_path.stem,
                            f"{m}_{p}"
                        )


    def save_curve_image(self, curve, sample_name, index):

        plt.figure(figsize=(3,3))

        plt.plot(curve, color="black", linewidth=1)

        plt.axis("off")

        filename = f"{sample_name}_{index}.png"

        save_path = self.output_dir / filename

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

        plt.close()