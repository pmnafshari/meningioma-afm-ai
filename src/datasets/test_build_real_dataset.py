import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.datasets.build_real_afm_dataset import AFMRealDatasetBuilder


builder = AFMRealDatasetBuilder()

builder.build()