"""Creative exploration sandbox for the CS136 term project."""

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
    # TODO: Prototype a novel visualization or hybrid pipeline and document intent.
    dataset_dir = os.path.join(_ROOT, "datasets", "MarineBiology")
    images = load_images_from_folder(dataset_dir)

    out_dir = os.path.join(os.path.dirname(__file__), "output_images")
    os.makedirs(out_dir, exist_ok=True)

    if not images:
        print(f"No images found in {dataset_dir}. Add .jpg or .png files to run the pipeline.")
        return

    for filename, bgr in images:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        boosted = cv2.merge([h, s, cv2.equalizeHist(v)])
        creative_bgr = cv2.cvtColor(boosted, cv2.COLOR_HSV2BGR)

        stem, _ = os.path.splitext(filename)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_creative.png"), creative_bgr)

        plt.figure(figsize=(6, 4))
        plt.imshow(cv2.cvtColor(creative_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Creative preprocessing (placeholder)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stem}_creative_preview.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
