import matplotlib.pyplot as plt
from pathlib import Path
import cv2

gradcam_dir = Path("results/gradcam_examples")

classes = [
    "MENINGOTHELIAL_G1",
    "MENINGOTHELIAL_G2",
    "PSAMMOMATOUS_G1"
]

images = {}

for c in classes:
    imgs = list(gradcam_dir.glob(f"{c}*.png"))[:3]
    images[c] = imgs

fig, axes = plt.subplots(3, 3, figsize=(9,9))

for col, c in enumerate(classes):

    for row in range(3):

        img_path = images[c][row]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[row, col].imshow(img)
        axes[row, col].axis("off")

        if row == 0:
            axes[row, col].set_title(c)

plt.tight_layout()

output_path = "results/gradcam_panel.png"

plt.savefig(output_path, dpi=300)

print("panel saved to", output_path)