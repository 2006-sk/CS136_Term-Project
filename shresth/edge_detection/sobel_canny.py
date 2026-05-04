"""Sobel and Canny edge detection for the CS136 term project."""

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
    # TODO: Compare Sobel/Canny responses, tune thresholds, and summarize findings.
    dataset_dir = os.path.join(_ROOT, "datasets", "MarineBiology")
    images = load_images_from_folder(dataset_dir)

    out_dir = os.path.join(os.path.dirname(__file__), "output_images")
    os.makedirs(out_dir, exist_ok=True)

    if not images:
        print(f"No images found in {dataset_dir}. Add .jpg or .png files to run the pipeline.")
        return

    for filename, bgr in images:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
        edges = cv2.Canny(gray, 80, 160)

        stem, _ = os.path.splitext(filename)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_sobel.png"), sobel_mag)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_canny.png"), edges)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(sobel_mag, cmap="gray")
        plt.title("Sobel magnitude")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap="gray")
        plt.title("Canny edges")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stem}_edges_preview.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
