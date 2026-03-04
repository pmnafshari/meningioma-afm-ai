import os
import numpy as np
import matplotlib.pyplot as plt


class CurveToImage:

    def __init__(self, output_dir="data/processed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_curve_image(self, curve, name):

        plt.figure(figsize=(4,4))
        plt.plot(curve)

        plt.xlabel("Indentation")
        plt.ylabel("Force")

        plt.title("AFM Force Curve")

        save_path = os.path.join(self.output_dir, f"{name}.png")

        plt.savefig(save_path)
        plt.close()

        return save_path