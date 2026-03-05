import os
import numpy as np
from curve_to_image import CurveToImage


class AFMDatasetGenerator:

    def __init__(self, output_dir="data/dataset"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.converter = CurveToImage(output_dir)

    def generate_from_curves(self, curves):

        saved_paths = []

        for i, curve in enumerate(curves):

            name = f"curve_{i}"

            path = self.converter.save_curve_image(curve, name)

            saved_paths.append(path)

        return saved_paths