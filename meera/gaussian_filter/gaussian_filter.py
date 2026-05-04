"""Apply Gaussian filtering to every dataset image for Meera's Part 1."""

import os
import re
import sys
from typing import Dict, Iterable, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.image_loader import load_images_from_folder

DATASET_ROOT = os.path.join(_ROOT, "datasets")
DATASET_FOLDERS: Dict[str, Tuple[str, ...]] = {
    "Marine_Science": ("Marine Science", "MarineScience", "MarineBiology", "Marine Biology"),
    "Geology": ("Geology",),
    "Anthropology": ("Anthropology",),
}
OUTPUT_FOLDER = "Gaussian_Filter_Images"
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 1.5


def _safe_name(name: str) -> str:
    """Return a filesystem-safe name while keeping labels readable."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _first_existing_dataset(candidates: Iterable[str]) -> Optional[str]:
    for folder in candidates:
        path = os.path.join(DATASET_ROOT, folder)
        if os.path.isdir(path):
            return path
    return None


def _iter_dataset_images() -> Iterable[Tuple[str, str, np.ndarray]]:
    """Yield labeled images from all project dataset categories."""
    for label, candidates in DATASET_FOLDERS.items():
        dataset_dir = _first_existing_dataset(candidates)
        if dataset_dir is None:
            print(f"Skipping {label}: none of these folders exist: {', '.join(candidates)}")
            continue

        images = load_images_from_folder(dataset_dir)
        if not images:
            print(f"Skipping {label}: no .jpg, .jpeg, or .png images found in {dataset_dir}")
            continue

        for filename, image in images:
            yield label, filename, image


def _write_comparison(original: np.ndarray, filtered: np.ndarray, title: str, out_path: str) -> None:
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_rgb)
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    out_root = os.path.join(os.path.dirname(__file__), OUTPUT_FOLDER)
    os.makedirs(out_root, exist_ok=True)

    processed = 0
    for dataset_label, filename, bgr in _iter_dataset_images():
        dataset_out = os.path.join(out_root, dataset_label)
        os.makedirs(dataset_out, exist_ok=True)

        blurred = cv2.GaussianBlur(bgr, GAUSSIAN_KERNEL, sigmaX=GAUSSIAN_SIGMA)
        stem, _ = os.path.splitext(_safe_name(filename))
        filtered_path = os.path.join(dataset_out, f"{stem}_gaussian_filtered.png")
        comparison_path = os.path.join(dataset_out, f"{stem}_gaussian_comparison.png")

        cv2.imwrite(filtered_path, blurred)
        _write_comparison(
            bgr,
            blurred,
            f"Gaussian {GAUSSIAN_KERNEL}, sigma={GAUSSIAN_SIGMA}",
            comparison_path,
        )
        processed += 1

    if processed == 0:
        print("No images were processed. Add dataset images under the expected datasets folders.")
        return

    print(f"Gaussian filtering complete: saved {processed} image result(s) in {out_root}")


if __name__ == "__main__":
    main()
