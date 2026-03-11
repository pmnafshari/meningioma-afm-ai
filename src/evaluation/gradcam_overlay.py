import cv2
import numpy as np


def overlay_gradcam(original_image_path, heatmap, output_path):

    image = cv2.imread(original_image_path)

    image = cv2.resize(image, (224,224))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(output_path, overlay)