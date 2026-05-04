"""Creative CV exploration: CLAHE+Canny, morphological edge cleanup, and DoG-style Sobel."""

import os
import sys

import cv2
import matplotlib.pyplot as plt
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
            all_items.append((f"{sub}/{name}", name, img))
    return all_items


def _morphological_skeleton(binary: np.ndarray) -> np.ndarray:
    """Zhang–Suen style iterative thinning using morphological hits (OpenCV)."""
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _sobel_mag_u8(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(gx, gy))


def main() -> None:
    images = _load_all_dataset_images()
    out_dir = os.path.join(os.path.dirname(__file__), "output_images")
    os.makedirs(out_dir, exist_ok=True)

    if not images:
        print("No images found under datasets/. Expected MarineBiology, Geology, Anthropology.")
        return

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for rel_key, filename, bgr in images:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        stem = os.path.splitext(rel_key.replace(os.sep, "_"))[0]

        # --- Technique 1: CLAHE + Canny vs plain Canny ---
        plain_canny = cv2.Canny(gray, 80, 160)
        clahe_gray = clahe.apply(gray)
        clahe_canny = cv2.Canny(clahe_gray, 80, 160)

        fig1, axes1 = plt.subplots(1, 3, figsize=(12, 4))
        axes1[0].imshow(gray, cmap="gray")
        axes1[0].set_title("Grayscale")
        axes1[1].imshow(plain_canny, cmap="gray")
        axes1[1].set_title("Plain Canny")
        axes1[2].imshow(clahe_canny, cmap="gray")
        axes1[2].set_title("CLAHE + Canny")
        for ax in axes1:
            ax.axis("off")
        plt.suptitle(f"Technique 1 — {rel_key}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tech1_clahe_canny_{stem}.png"), dpi=150)
        plt.close()

        cv2.imwrite(os.path.join(out_dir, f"tech1_plain_canny_{stem}.png"), plain_canny)
        cv2.imwrite(os.path.join(out_dir, f"tech1_clahe_canny_u8_{stem}.png"), clahe_canny)

        # --- Technique 2: Canny then dilate + skeleton ---
        canny_t2 = cv2.Canny(gray, 80, 160)
        dilated = cv2.dilate(canny_t2, kernel, iterations=1)
        skeleton = _morphological_skeleton(dilated)

        fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
        axes2[0].imshow(canny_t2, cmap="gray")
        axes2[0].set_title("Canny")
        axes2[1].imshow(dilated, cmap="gray")
        axes2[1].set_title("After dilation")
        axes2[2].imshow(skeleton, cmap="gray")
        axes2[2].set_title("Morphological skeleton")
        for ax in axes2:
            ax.axis("off")
        plt.suptitle(f"Technique 2 — {rel_key}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tech2_morph_edges_{stem}.png"), dpi=150)
        plt.close()

        cv2.imwrite(os.path.join(out_dir, f"tech2_canny_{stem}.png"), canny_t2)
        cv2.imwrite(os.path.join(out_dir, f"tech2_dilated_{stem}.png"), dilated)
        cv2.imwrite(os.path.join(out_dir, f"tech2_skeleton_{stem}.png"), skeleton)

        # --- Technique 3: Multi-scale Gaussian + Sobel (DoG-style) ---
        blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
        blur3 = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0, sigmaY=3.0)
        s1 = _sobel_mag_u8(blur1).astype(np.float32)
        s3 = _sobel_mag_u8(blur3).astype(np.float32)
        dog = cv2.normalize(np.abs(s1 - s3), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imwrite(os.path.join(out_dir, f"tech3_dog_sobel_{stem}.png"), dog)

        fig3, axes3 = plt.subplots(1, 3, figsize=(12, 4))
        axes3[0].imshow(s1.astype(np.uint8), cmap="gray")
        axes3[0].set_title("Sobel |∇| (σ=1 blur)")
        axes3[1].imshow(s3.astype(np.uint8), cmap="gray")
        axes3[1].set_title("Sobel |∇| (σ=3 blur)")
        axes3[2].imshow(dog, cmap="gray")
        axes3[2].set_title("|Sobel₁ − Sobel₃| (DoG-style)")
        for ax in axes3:
            ax.axis("off")
        plt.suptitle(f"Technique 3 — {rel_key}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tech3_multiscale_{stem}.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()

_CREATIVE_REPORT = """
=== Creative exploration report (parameters & expected outcomes) ===

Technique 1 — CLAHE + Canny:
  CLAHE (clipLimit=2.0, tileGridSize=8x8) boosts local contrast before Canny(80,160).
  Expected: finer texture and low-contrast boundaries become more visible versus plain
  Canny on raw grayscale; risk of amplifying sensor noise in very flat regions.

Technique 2 — Morphological edge cleanup:
  Canny edges are dilated (3x3 rect, 1 iter) to close small gaps, then thinned via
  iterative morphological skeletonization (cross structuring element).
  Expected: slightly thicker primitives first, then single-pixel-wide medial traces
  that emphasize topology of edge networks; small spurs may remain without pruning.

Technique 3 — Multi-scale Gaussian + Sobel (DoG-style):
  Gaussian blurs with σ=1 vs σ=3, Sobel magnitude on each, absolute difference,
  min-max normalized to 8-bit for saving/display.
  Expected: emphasizes edges/scales present at one smoothing level but not the other,
  similar intuition to band-pass / LoG-style highlighting of blob boundaries.
"""
