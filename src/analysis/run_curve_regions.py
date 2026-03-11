import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.analysis.afm_curve_regions import detect_curve_regions
from src.analysis.plot_curve_regions import plot_curve_regions


image_path = "data/dataset/train/curve_0.png"

contact, adhesion = detect_curve_regions(image_path)

plot_curve_regions(
    image_path,
    contact,
    adhesion,
    "results/curve_regions.png"
)

print("curve regions saved")