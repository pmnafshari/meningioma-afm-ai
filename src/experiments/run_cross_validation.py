import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.experiments.cross_validation import CrossValidator


cv = CrossValidator(k=5)

cv.run()