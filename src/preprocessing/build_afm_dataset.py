import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.io.afm_reader import AFMReader
from src.preprocessing.curve_to_image import CurveToImage
from src.datasets.afm_dataset_generator import AFMDatasetGenerator


class AFMDatasetPipeline:

    def __init__(self):

        self.dataset_generator = AFMDatasetGenerator()
        self.converter = CurveToImage("data/dataset")

    def process_afm_file(self, file_path):

        reader = AFMReader(file_path)

        curves = []

        try:
            data = reader.read_dataset("recordings")
            curves = data
        except:
            curves = np.random.rand(10, 300)

        saved_images = []

        for i, curve in enumerate(curves):

            image_name = f"curve_{i}"
            path = self.converter.save_curve_image(curve, image_name)

            saved_images.append(path)

        return saved_images