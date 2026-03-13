import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.preprocessing.afm_curve_extractor import AFMCurveExtractor


extractor = AFMCurveExtractor()

extractor.extract_curves()

print("curve images generated")