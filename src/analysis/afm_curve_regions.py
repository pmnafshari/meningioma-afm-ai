import cv2
import numpy as np


def detect_curve_regions(image_path):

    image = cv2.imread(image_path, 0)

    image = cv2.resize(image, (224,224))

    edges = cv2.Canny(image, 50,150)

    curve_pixels = np.where(edges > 0)

    x = curve_pixels[1]

    contact = int(np.percentile(x, 30))

    adhesion = int(np.percentile(x, 80))

    return contact, adhesion