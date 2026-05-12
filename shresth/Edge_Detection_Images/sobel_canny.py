"""Sobel and Canny edge detection for the CS136 term project.

Scans every image under the three dataset theme folders (on disk: MarineBiology,
Geology, Anthropology - marine_science / geology / anthropology). Output PNGs are
not fixed filenames: for each discovered image this script writes four files into
the local output_images/ directory at runtime (plain_sobel_x_*, plain_sobel_y_*,
etc.).
"""

import os

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Repository folder names under datasets/ (themes: marine_science, geology, anthropology)
_DATASET_SUBDIRS = ("MarineBiology", "Geology", "Anthropology")


def _is_image_filename(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".jpg", ".jpeg", ".png"))


def discover_all_dataset_images():
    """Walk each dataset subfolder with os.listdir; yield (subfolder, filename, BGR image)."""
    for sub in _DATASET_SUBDIRS:
        folder = os.path.join(_ROOT, "datasets", sub)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not _is_image_filename(fname):
                continue
            path = os.path.join(folder, fname)
            if not os.path.isfile(path):
                continue
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            yield sub, fname, bgr


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_images")
    os.makedirs(out_dir, exist_ok=True)

    images = list(discover_all_dataset_images())
    if not images:
        print("No images found under datasets/. Expected MarineBiology, Geology, Anthropology.")
        return

    for sub, filename, bgr in images:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_x = cv2.convertScaleAbs(gx)
        sobel_y = cv2.convertScaleAbs(gy)
        sobel_combined = cv2.convertScaleAbs(cv2.magnitude(gx, gy))

        plain_canny = cv2.Canny(gray, 100, 200)

        base, _ = os.path.splitext(filename)
        stem = f"{sub}_{base}"

        cv2.imwrite(os.path.join(out_dir, f"plain_sobel_x_{stem}.png"), sobel_x)
        cv2.imwrite(os.path.join(out_dir, f"plain_sobel_y_{stem}.png"), sobel_y)
        cv2.imwrite(os.path.join(out_dir, f"plain_sobel_combined_{stem}.png"), sobel_combined)
        cv2.imwrite(os.path.join(out_dir, f"plain_canny_{stem}.png"), plain_canny)


if __name__ == "__main__":
    main()
