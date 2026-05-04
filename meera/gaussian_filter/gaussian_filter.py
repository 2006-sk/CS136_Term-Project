"""Gaussian filtering demo for the CS136 term project."""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.image_loader import load_images_from_folder


def main() -> None:
    # TODO: Implement Gaussian blur parameters, comparison plots, and reporting.
    dataset_dir = os.path.join(_ROOT, "datasets", "marine_science")
    images = load_images_from_folder(dataset_dir)

    out_dir = os.path.join(os.path.dirname(__file__), "output_images")
    os.makedirs(out_dir, exist_ok=True)

    if not images:
        print(f"No images found in {dataset_dir}. Add .jpg or .png files to run the pipeline.")
        return

    for filename, bgr in images:
        blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.5)
        stem, _ = os.path.splitext(filename)
        out_path = os.path.join(out_dir, f"{stem}_gaussian.png")
        cv2.imwrite(out_path, blurred)

        rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 4))
        plt.imshow(rgb)
        plt.title("Gaussian blur (placeholder)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stem}_gaussian_preview.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
