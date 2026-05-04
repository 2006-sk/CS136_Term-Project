"""Texture-based image segmentation for Meera's Part 1."""

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
OUTPUT_FOLDER = "Texture_Segmentation_Images"
SEGMENT_COUNT = 4
KMEANS_ATTEMPTS = 5


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _first_existing_dataset(candidates: Iterable[str]) -> Optional[str]:
    for folder in candidates:
        path = os.path.join(DATASET_ROOT, folder)
        if os.path.isdir(path):
            return path
    return None


def _iter_dataset_images() -> Iterable[Tuple[str, str, np.ndarray]]:
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


def _normalize_feature(feature: np.ndarray) -> np.ndarray:
    feature = feature.astype(np.float32)
    mean = float(feature.mean())
    std = float(feature.std())
    if std < 1e-6:
        return np.zeros_like(feature, dtype=np.float32)
    return (feature - mean) / std


def _local_texture_features(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    local_mean = cv2.GaussianBlur(gray_f, (9, 9), 0)
    local_sq_mean = cv2.GaussianBlur(gray_f * gray_f, (9, 9), 0)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0))
    laplacian = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
    edge_energy = cv2.GaussianBlur(np.abs(laplacian), (9, 9), 0)
    return np.dstack(
        [
            _normalize_feature(gray_f),
            _normalize_feature(local_mean),
            _normalize_feature(local_std),
            _normalize_feature(edge_energy),
        ]
    )


def _cluster_features(features: np.ndarray, segment_count: int = SEGMENT_COUNT) -> np.ndarray:
    rows, cols, channels = features.shape
    samples = features.reshape((-1, channels)).astype(np.float32)
    cluster_count = max(1, min(segment_count, samples.shape[0]))

    if cluster_count == 1:
        return np.zeros((rows, cols), dtype=np.uint8)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.2)
    _compactness, labels, _centers = cv2.kmeans(
        samples,
        cluster_count,
        None,
        criteria,
        KMEANS_ATTEMPTS,
        cv2.KMEANS_PP_CENTERS,
    )
    return labels.reshape((rows, cols)).astype(np.uint8)


def _labels_to_color(labels: np.ndarray) -> np.ndarray:
    labels_u8 = labels.astype(np.uint8)
    if int(labels_u8.max()) > 0:
        labels_u8 = np.round(labels_u8 * (255 / int(labels_u8.max()))).astype(np.uint8)
    return cv2.applyColorMap(labels_u8, cv2.COLORMAP_TURBO)


def _segment_with_average_color(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    segmented = np.zeros_like(image)
    for label in np.unique(labels):
        mask = labels == label
        segmented[mask] = image[mask].mean(axis=0)
    return segmented


def _write_preview(original: np.ndarray, gray_labels: np.ndarray, color_labels: np.ndarray, out_path: str) -> None:
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    gray_rgb = cv2.cvtColor(_labels_to_color(gray_labels), cv2.COLOR_BGR2RGB)
    color_rgb = cv2.cvtColor(_labels_to_color(color_labels), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 4))
    for index, (title, image) in enumerate(
        [("Original", original_rgb), ("Grayscale texture", gray_rgb), ("Texture + color", color_rgb)],
        start=1,
    ):
        plt.subplot(1, 3, index)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def segment_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return grayscale-texture labels, color-texture labels, and a color segmented image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_features = _local_texture_features(gray)
    gray_labels = _cluster_features(texture_features)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_features = np.dstack(
        [
            texture_features,
            _normalize_feature(lab[:, :, 0]),
            _normalize_feature(lab[:, :, 1]),
            _normalize_feature(lab[:, :, 2]),
        ]
    )
    color_labels = _cluster_features(color_features)
    color_segmented = _segment_with_average_color(image, color_labels)
    return gray_labels, color_labels, color_segmented


def main() -> None:
    out_root = os.path.join(os.path.dirname(__file__), OUTPUT_FOLDER)
    os.makedirs(out_root, exist_ok=True)

    processed = 0
    for dataset_label, filename, bgr in _iter_dataset_images():
        dataset_out = os.path.join(out_root, dataset_label)
        os.makedirs(dataset_out, exist_ok=True)

        stem, _ = os.path.splitext(_safe_name(filename))
        gray_labels, color_labels, color_segmented = segment_image(bgr)

        cv2.imwrite(os.path.join(dataset_out, f"{stem}_01_grayscale_texture_labels.png"), _labels_to_color(gray_labels))
        cv2.imwrite(os.path.join(dataset_out, f"{stem}_02_color_texture_labels.png"), _labels_to_color(color_labels))
        cv2.imwrite(os.path.join(dataset_out, f"{stem}_03_color_segmented.png"), color_segmented)
        _write_preview(
            bgr,
            gray_labels,
            color_labels,
            os.path.join(dataset_out, f"{stem}_texture_segmentation_preview.png"),
        )
        processed += 1

    if processed == 0:
        print("No images were processed. Add dataset images under the expected datasets folders.")
        return

    print(f"Texture segmentation complete: saved {processed} image result(s) in {out_root}")


if __name__ == "__main__":
    main()
