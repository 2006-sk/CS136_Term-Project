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

# On-disk folders under datasets/ (match repo layout; correspond to marine / geology / anthropology themes)
_DATASET_SUBDIRS = ("MarineBiology", "Geology", "Anthropology")


def _load_all_dataset_images():
    all_items = []
    for sub in _DATASET_SUBDIRS:
        folder = os.path.join(_ROOT, "datasets", sub)
        for name, img in load_images_from_folder(folder):
            all_items.append((f"{sub}/{name}", name, img))
    return all_items


def main() -> None:
    images = _load_all_dataset_images()
    out_dir = os.path.join(os.path.dirname(__file__), "output_images")
    os.makedirs(out_dir, exist_ok=True)

    if not images:
        print("No images found under datasets/. Expected MarineBiology, Geology, Anthropology.")
        return

    for rel_key, filename, bgr in images:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_x = cv2.convertScaleAbs(gx)
        sobel_y = cv2.convertScaleAbs(gy)
        sobel_combined = cv2.convertScaleAbs(cv2.magnitude(gx, gy))

        canny_tight = cv2.Canny(gray, 100, 200)
        canny_loose = cv2.Canny(gray, 50, 150)

        stem = os.path.splitext(rel_key.replace(os.sep, "_"))[0]

        cv2.imwrite(os.path.join(out_dir, f"{stem}_sobel_x.png"), sobel_x)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_sobel_y.png"), sobel_y)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_sobel_combined.png"), sobel_combined)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_canny.png"), canny_tight)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes[0, 0].imshow(gray, cmap="gray")
        axes[0, 0].set_title("Grayscale")
        axes[0, 1].imshow(sobel_x, cmap="gray")
        axes[0, 1].set_title("Sobel X")
        axes[0, 2].imshow(sobel_y, cmap="gray")
        axes[0, 2].set_title("Sobel Y")
        axes[1, 0].imshow(sobel_combined, cmap="gray")
        axes[1, 0].set_title("Sobel magnitude")
        axes[1, 1].imshow(canny_tight, cmap="gray")
        axes[1, 1].set_title("Canny tight (100, 200)")
        axes[1, 2].imshow(canny_loose, cmap="gray")
        axes[1, 2].set_title("Canny loose (50, 150)")
        for ax in axes.ravel():
            ax.axis("off")
        plt.suptitle(f"Edges: {rel_key}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stem}_comparison.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
