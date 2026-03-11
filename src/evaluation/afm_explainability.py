import cv2
import numpy as np


def create_explainability_figure(image_path, heatmap, contact, adhesion, output_path):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (224,224))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    height = overlay.shape[0]

    cv2.line(overlay, (contact,0), (contact,height), (0,255,0),2)

    cv2.line(overlay, (adhesion,0), (adhesion,height), (255,0,0),2)

    cv2.putText(overlay,"contact",(contact+5,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

    cv2.putText(overlay,"adhesion",(adhesion+5,60),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)

    cv2.imwrite(output_path, overlay)