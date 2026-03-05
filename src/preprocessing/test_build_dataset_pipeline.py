import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.preprocessing.build_afm_dataset import AFMDatasetPipeline


pipeline = AFMDatasetPipeline()

images = pipeline.process_afm_file("data/raw_afm/sample.h5")

print("generated images")
print(images)