"""Hough transform line detection for the CS136 term project."""

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
    # TODO: Tune Hough parameters, visualize peaks, and document failure cases.
    dataset_dir = os.path.join(_ROOT, "datasets", "Geology")
    images = load_images_from_folder(dataset_dir)

    out_dir = os.path.join(os.path.dirname(__file__), "output_images")
    os.makedirs(out_dir, exist_ok=True)

    if not images:
        print(f"No images found in {dataset_dir}. Add .jpg or .png files to run the pipeline.")
        return

    for filename, bgr in images:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

        overlay = bgr.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        stem, _ = os.path.splitext(filename)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_hough_lines.png"), overlay)

        plt.figure(figsize=(6, 4))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Hough lines (placeholder)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stem}_hough_preview.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
