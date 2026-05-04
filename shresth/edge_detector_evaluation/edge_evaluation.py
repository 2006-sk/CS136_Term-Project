"""Edge detector evaluation helpers for the CS136 term project."""

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
    # TODO: Add precision/recall or boundary metrics comparing detectors to references.
    dataset_dir = os.path.join(_ROOT, "datasets", "Anthropology")
    images = load_images_from_folder(dataset_dir)

    out_dir = os.path.join(os.path.dirname(__file__), "output_images")
    os.makedirs(out_dir, exist_ok=True)

    if not images:
        print(f"No images found in {dataset_dir}. Add .jpg or .png files to run the pipeline.")
        return

    for filename, bgr in images:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 60, 120)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_vis = cv2.convertScaleAbs(laplacian)

        stem, _ = os.path.splitext(filename)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_canny_eval.png"), canny)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_laplacian_eval.png"), laplacian_vis)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(canny, cmap="gray")
        plt.title("Canny")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(laplacian_vis, cmap="gray")
        plt.title("Laplacian")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stem}_edge_eval_preview.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
