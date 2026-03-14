import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.preprocessing.afm_curve_extractor import AFMCurveExtractor


extractor = AFMCurveExtractor(
    input_dir="data/raw_afm/materials",
    output_dir="data/curve_images"
)

extractor.extract_curves()

print("curve images generated")