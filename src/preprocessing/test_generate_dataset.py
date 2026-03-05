import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.datasets.afm_dataset_generator import AFMDatasetGenerator


generator = AFMDatasetGenerator()

print("AFM files found:")
print(generator.list_afm_files())

train_dir, val_dir, test_dir = generator.create_dataset_structure()

print("dataset folders created:")
print(train_dir)
print(val_dir)
print(test_dir)