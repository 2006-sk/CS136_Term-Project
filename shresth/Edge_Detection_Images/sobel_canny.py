"""Sobel and Canny edge detection for the CS136 term project."""

import os
import sys

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.image_loader import load_images_from_folder

_DATASET_SUBDIRS = ("MarineBiology", "Geology", "Anthropology")


def _load_all_dataset_images():
    all_items = []
    for sub in _DATASET_SUBDIRS:
        folder = os.path.join(_ROOT, "datasets", sub)
        for name, img in load_images_from_folder(folder):
            all_items.append((name, img))
    return all_items


def main() -> None:
    images = _load_all_dataset_images()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    if not images:
        print("No images found under datasets/. Expected MarineBiology, Geology, Anthropology.")
        return

    for filename, bgr in images:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_x = cv2.convertScaleAbs(gx)
        sobel_y = cv2.convertScaleAbs(gy)
        sobel_combined = cv2.convertScaleAbs(cv2.magnitude(gx, gy))

        plain_canny = cv2.Canny(gray, 100, 200)

        stem, _ = os.path.splitext(filename)

        cv2.imwrite(os.path.join(out_dir, f"plain_sobel_x_{stem}.png"), sobel_x)
        cv2.imwrite(os.path.join(out_dir, f"plain_sobel_y_{stem}.png"), sobel_y)
        cv2.imwrite(os.path.join(out_dir, f"plain_sobel_combined_{stem}.png"), sobel_combined)
        cv2.imwrite(os.path.join(out_dir, f"plain_canny_{stem}.png"), plain_canny)


if __name__ == "__main__":
    main()
