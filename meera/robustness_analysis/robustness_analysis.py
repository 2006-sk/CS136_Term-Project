"""Robustness analysis for Meera's Part 3."""

import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

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
OUTPUT_FOLDER = "Robustness_Analysis_Images"
SUBSET_PER_DATASET = 3
RANDOM_SEED = 136
SEGMENT_COUNT = 4


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _first_existing_dataset(candidates: Iterable[str]) -> Optional[str]:
    for folder in candidates:
        path = os.path.join(DATASET_ROOT, folder)
        if os.path.isdir(path):
            return path
    return None


def _iter_subset_images() -> Iterable[Tuple[str, str, np.ndarray]]:
    for label, candidates in DATASET_FOLDERS.items():
        dataset_dir = _first_existing_dataset(candidates)
        if dataset_dir is None:
            print(f"Skipping {label}: none of these folders exist: {', '.join(candidates)}")
            continue

        images = load_images_from_folder(dataset_dir)[:SUBSET_PER_DATASET]
        if not images:
            print(f"Skipping {label}: no .jpg, .jpeg, or .png images found in {dataset_dir}")
            continue

        for filename, image in images:
            yield label, filename, image


def _add_gaussian_noise(image: np.ndarray, rng: np.random.Generator, sigma: float = 22.0) -> np.ndarray:
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _lower_contrast(image: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    gray_midpoint = np.full_like(image, 128)
    return cv2.addWeighted(image, alpha, gray_midpoint, 1.0 - alpha, 0)


def _distortions(image: np.ndarray, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    return {
        "clean": image,
        "gaussian_noise": _add_gaussian_noise(image, rng),
        "blur": cv2.GaussianBlur(image, (9, 9), 2.0),
        "low_contrast": _lower_contrast(image),
    }


def _gaussian_preprocess(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 1.2)


def _sobel_edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude_u8 = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _threshold, edges = cv2.threshold(
        magnitude_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return edges


def _canny_edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    if lower == upper:
        lower, upper = 50, 150
    return cv2.Canny(gray, lower, upper)


def _edge_metrics(edge_map: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
    edge_bool = edge_map > 0
    baseline_bool = baseline > 0
    true_positive = float(np.logical_and(edge_bool, baseline_bool).sum())
    precision = true_positive / max(float(edge_bool.sum()), 1.0)
    recall = true_positive / max(float(baseline_bool.sum()), 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    density = float(edge_bool.mean())
    return {"density": density, "precision": precision, "recall": recall, "f1": f1}


def _normalize_feature(feature: np.ndarray) -> np.ndarray:
    feature = feature.astype(np.float32)
    mean = float(feature.mean())
    std = float(feature.std())
    if std < 1e-6:
        return np.zeros_like(feature, dtype=np.float32)
    return (feature - mean) / std


def _texture_features(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    local_mean = cv2.GaussianBlur(gray, (9, 9), 0)
    local_sq_mean = cv2.GaussianBlur(gray * gray, (9, 9), 0)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0))
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge_energy = cv2.GaussianBlur(np.abs(laplacian), (9, 9), 0)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.dstack(
        [
            _normalize_feature(gray),
            _normalize_feature(local_mean),
            _normalize_feature(local_std),
            _normalize_feature(edge_energy),
            _normalize_feature(lab[:, :, 0]),
            _normalize_feature(lab[:, :, 1]),
            _normalize_feature(lab[:, :, 2]),
        ]
    )


def _cluster_features(features: np.ndarray) -> np.ndarray:
    rows, cols, channels = features.shape
    samples = features.reshape((-1, channels)).astype(np.float32)
    cluster_count = max(1, min(SEGMENT_COUNT, samples.shape[0]))
    if cluster_count == 1:
        return np.zeros((rows, cols), dtype=np.uint8)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 70, 0.2)
    _compactness, labels, _centers = cv2.kmeans(
        samples,
        cluster_count,
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS,
    )
    return labels.reshape((rows, cols)).astype(np.uint8)


def _segmentation_boundaries(labels: np.ndarray) -> np.ndarray:
    boundaries = np.zeros(labels.shape, dtype=np.uint8)
    boundaries[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundaries[1:, :] |= labels[1:, :] != labels[:-1, :]
    return boundaries * 255


def _labels_to_color(labels: np.ndarray) -> np.ndarray:
    labels_u8 = labels.astype(np.uint8)
    if int(labels_u8.max()) > 0:
        labels_u8 = np.round(labels_u8 * (255 / int(labels_u8.max()))).astype(np.uint8)
    return cv2.applyColorMap(labels_u8, cv2.COLORMAP_TURBO)


def _write_figure(
    original: np.ndarray,
    distorted: np.ndarray,
    canny: np.ndarray,
    sobel: np.ndarray,
    labels: np.ndarray,
    out_path: str,
) -> None:
    plt.figure(figsize=(12, 7))
    panels = [
        ("Original", cv2.cvtColor(original, cv2.COLOR_BGR2RGB), None),
        ("Distorted", cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB), None),
        ("Canny", canny, "gray"),
        ("Sobel", sobel, "gray"),
        ("Texture segmentation", cv2.cvtColor(_labels_to_color(labels), cv2.COLOR_BGR2RGB), None),
    ]
    for index, (title, image, cmap) in enumerate(panels, start=1):
        plt.subplot(2, 3, index)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _summarize(records: List[Dict[str, object]]) -> str:
    if not records:
        return "No robustness records were generated because no dataset images were available.\n"

    lines = [
        "Robustness Analysis Summary",
        "===========================",
        "",
        "Pipeline: clean image -> distortion -> optional Gaussian preprocessing -> Canny/Sobel edges + texture segmentation.",
        "Scores compare each distorted output with the matching clean-image baseline. Higher F1 means better stability.",
        "",
        "Average edge stability by distortion and preprocessing:",
    ]

    for distortion in ("gaussian_noise", "blur", "low_contrast"):
        lines.append(f"\n{distortion}:")
        for preprocess in ("raw", "gaussian_filtered"):
            subset = [
                row
                for row in records
                if row["distortion"] == distortion and row["preprocess"] == preprocess
            ]
            if not subset:
                continue
            canny_f1 = float(np.mean([row["canny_f1"] for row in subset]))
            sobel_f1 = float(np.mean([row["sobel_f1"] for row in subset]))
            seg_f1 = float(np.mean([row["segmentation_boundary_f1"] for row in subset]))
            lines.append(
                f"- {preprocess}: Canny F1={canny_f1:.3f}, Sobel F1={sobel_f1:.3f}, "
                f"segmentation boundary F1={seg_f1:.3f}"
            )

    canny_noise = [
        row["canny_f1"]
        for row in records
        if row["distortion"] == "gaussian_noise" and row["preprocess"] == "raw"
    ]
    sobel_noise = [
        row["sobel_f1"]
        for row in records
        if row["distortion"] == "gaussian_noise" and row["preprocess"] == "raw"
    ]
    if canny_noise and sobel_noise:
        lines.extend(
            [
                "",
                "Short written analysis:",
                f"- Under Gaussian noise, Canny averaged {np.mean(canny_noise):.3f} F1 and Sobel averaged {np.mean(sobel_noise):.3f} F1 against the clean baseline.",
                "- Canny usually suppresses weak noisy gradients better because of non-maximum suppression and hysteresis thresholds.",
                "- Sobel is more sensitive to noise because every local intensity fluctuation contributes directly to gradient magnitude.",
                "- Gaussian filtering before edge detection generally helps noisy inputs by smoothing small fluctuations before gradients are computed.",
                "- Blur and low contrast reduce edge strength, so both edge detectors can lose fine boundaries; segmentation boundaries also become less stable when texture contrast disappears.",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    out_root = os.path.join(os.path.dirname(__file__), OUTPUT_FOLDER)
    os.makedirs(out_root, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    records: List[Dict[str, object]] = []

    for dataset_label, filename, bgr in _iter_subset_images():
        stem, _ = os.path.splitext(_safe_name(filename))
        image_out = os.path.join(out_root, dataset_label, stem)
        os.makedirs(image_out, exist_ok=True)

        distorted_images = _distortions(bgr, rng)
        baselines = {}
        for preprocess_name, preprocess_fn in {
            "raw": lambda image: image,
            "gaussian_filtered": _gaussian_preprocess,
        }.items():
            clean_processed = preprocess_fn(distorted_images["clean"])
            clean_labels = _cluster_features(_texture_features(clean_processed))
            baselines[preprocess_name] = {
                "canny": _canny_edges(clean_processed),
                "sobel": _sobel_edges(clean_processed),
                "segmentation": _segmentation_boundaries(clean_labels),
            }

        for distortion_name, distorted in distorted_images.items():
            cv2.imwrite(os.path.join(image_out, f"{distortion_name}.png"), distorted)

            for preprocess_name, preprocess_fn in {
                "raw": lambda image: image,
                "gaussian_filtered": _gaussian_preprocess,
            }.items():
                processed = preprocess_fn(distorted)
                labels = _cluster_features(_texture_features(processed))
                canny = _canny_edges(processed)
                sobel = _sobel_edges(processed)
                segmentation_edges = _segmentation_boundaries(labels)

                cv2.imwrite(
                    os.path.join(image_out, f"{distortion_name}_{preprocess_name}_canny.png"),
                    canny,
                )
                cv2.imwrite(
                    os.path.join(image_out, f"{distortion_name}_{preprocess_name}_sobel.png"),
                    sobel,
                )
                cv2.imwrite(
                    os.path.join(image_out, f"{distortion_name}_{preprocess_name}_segmentation.png"),
                    _labels_to_color(labels),
                )
                _write_figure(
                    bgr,
                    processed,
                    canny,
                    sobel,
                    labels,
                    os.path.join(image_out, f"{distortion_name}_{preprocess_name}_summary.png"),
                )

                baseline = baselines[preprocess_name]
                canny_metrics = _edge_metrics(canny, baseline["canny"])
                sobel_metrics = _edge_metrics(sobel, baseline["sobel"])
                segmentation_metrics = _edge_metrics(segmentation_edges, baseline["segmentation"])
                records.append(
                    {
                        "dataset": dataset_label,
                        "filename": filename,
                        "distortion": distortion_name,
                        "preprocess": preprocess_name,
                        "canny_f1": canny_metrics["f1"],
                        "sobel_f1": sobel_metrics["f1"],
                        "segmentation_boundary_f1": segmentation_metrics["f1"],
                        "canny_density": canny_metrics["density"],
                        "sobel_density": sobel_metrics["density"],
                    }
                )

    analysis_path = os.path.join(out_root, "robustness_analysis_summary.txt")
    with open(analysis_path, "w", encoding="utf-8") as analysis_file:
        analysis_file.write(_summarize(records))

    if not records:
        print("No images were processed. Add dataset images under the expected datasets folders.")
        print(f"Wrote empty analysis summary to {analysis_path}")
        return

    print(f"Robustness analysis complete: saved outputs and summary in {out_root}")


if __name__ == "__main__":
    main()
