import numpy as np
from curve_to_image import CurveToImage


curve = np.random.rand(300)

converter = CurveToImage()

path = converter.save_curve_image(curve, "test_curve")

print("Image saved at:", path)