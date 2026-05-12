"""Quantitative edge detector evaluation (Sobel vs Canny) for the CS136 term project.

Scans every image under the three dataset theme folders (on disk: MarineBiology,
Geology, Anthropology) using os.listdir per folder - counts are not fixed. Bar
chart PNG names (bar_marine_science.png, etc.) are fixed summary outputs written
to output_images/; per-image metrics come from whatever images were discovered.
"""

import os
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_DATASET_SUBDIRS = ("MarineBiology", "Geology", "Anthropology")
_DISPLAY_NAMES = {
    "MarineBiology": "marine_science",
    "Geology": "geology",
    "Anthropology": "anthropology",
}


def _is_image_filename(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".jpg", ".jpeg", ".png"))


def discover_all_dataset_images():
    """Walk each dataset subfolder with os.listdir; yield (folder_key, rel_key, filename, BGR)."""
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
            yield sub, f"{sub}/{fname}", fname, bgr


def _sobel_magnitude_float(gray: np.ndarray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.convertScaleAbs(mag)
    return gx, gy, mag, mag_u8


def _binary_sobel_edges(mag_u8: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _metrics_for_mask(mask: np.ndarray, mag_float: np.ndarray) -> dict:
    edge_pixels = mask > 0
    total = float(mask.size)
    density_pct = 100.0 * float(np.count_nonzero(edge_pixels)) / total if total else 0.0

    if np.any(edge_pixels):
        mean_grad = float(np.mean(mag_float[edge_pixels]))
    else:
        mean_grad = 0.0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    continuity = float(np.mean(areas)) if areas else 0.0

    return {
        "edge_density_pct": density_pct,
        "mean_edge_gradient": mean_grad,
        "avg_component_area": continuity,
    }


def main() -> None:
    images = list(discover_all_dataset_images())
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_images")
    os.makedirs(out_dir, exist_ok=True)

    if not images:
        print("No images found under datasets/. Expected MarineBiology, Geology, Anthropology.")
        return

    results = []
    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for folder, rel_key, filename, bgr in images:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, _, mag_f, mag_u8 = _sobel_magnitude_float(gray)
        sobel_mask = _binary_sobel_edges(mag_u8)
        canny_mask = cv2.Canny(gray, 100, 200)

        sobel_m = _metrics_for_mask(sobel_mask, mag_f)
        canny_m = _metrics_for_mask(canny_mask, mag_f)

        row = {
            "folder": folder,
            "rel_key": rel_key,
            "filename": filename,
            "sobel": sobel_m,
            "canny": canny_m,
        }
        results.append(row)

        for det in ("sobel", "canny"):
            for k, v in row[det].items():
                accum[folder][det][k].append(v)

    print("\n" + "=" * 100)
    print(f"{'Image':<50} | {'Det':<6} | {'Density %':>10} | {'Mean |grad|':>12} | {'Avg CC area':>12}")
    print("=" * 100)
    for row in results:
        for det in ("sobel", "canny"):
            m = row[det]
            print(
                f"{row['rel_key']:<50} | {det:<6} | {m['edge_density_pct']:10.3f} | "
                f"{m['mean_edge_gradient']:12.3f} | {m['avg_component_area']:12.1f}"
            )
    print("=" * 100 + "\n")

    metric_keys = ["edge_density_pct", "mean_edge_gradient", "avg_component_area"]
    metric_labels = ["Edge density (%)", "Mean |gradient| at edges", "Avg CC area (px)"]

    for folder in _DATASET_SUBDIRS:
        disp = _DISPLAY_NAMES.get(folder, folder)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, key, label in zip(axes, metric_keys, metric_labels):
            s_vals = accum[folder]["sobel"][key]
            c_vals = accum[folder]["canny"][key]
            if not s_vals or not c_vals:
                ax.set_visible(False)
                continue
            s_mean = float(np.mean(s_vals))
            c_mean = float(np.mean(c_vals))
            ax.bar([0, 1], [s_mean, c_mean], color=["#4C72B0", "#DD8452"])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Sobel", "Canny"])
            ax.set_ylabel(label)
            ax.set_title(label)
        fig.suptitle(f"Sobel vs Canny - {disp} (folder: {folder})")
        plt.tight_layout()
        safe = disp.replace(" ", "_")
        plt.savefig(os.path.join(out_dir, f"bar_{safe}.png"), dpi=150)
        plt.close()

    summary_parts = []
    for folder in _DATASET_SUBDIRS:
        disp = _DISPLAY_NAMES[folder]
        s_list = accum[folder]["sobel"]["edge_density_pct"]
        c_list = accum[folder]["canny"]["edge_density_pct"]
        if not s_list or not c_list:
            continue
        s_d = float(np.mean(s_list))
        c_d = float(np.mean(c_list))
        s_cc = float(np.mean(accum[folder]["sobel"]["avg_component_area"]))
        c_cc = float(np.mean(accum[folder]["canny"]["avg_component_area"]))
        if c_d < s_d - 0.05 and c_cc > s_cc + 0.5:
            rec = (
                f"For {disp}, Canny is sparser in edge density yet yields larger average components, "
                "favoring thin closed contours; Sobel-Otsu remains useful when richer gradient coverage is needed."
            )
        elif c_d > s_d + 0.05:
            rec = (
                f"For {disp}, Canny marks more edge pixels than Sobel-Otsu here; inspect for texture noise. "
                "Sobel-Otsu may give smoother continuity if Canny fragments thin structures."
            )
        else:
            rec = (
                f"For {disp}, densities are similar; prefer Canny for crisp hysteresis edges and Sobel-Otsu "
                "when tuning a single global threshold on gradient strength is easier than dual thresholds."
            )
        summary_parts.append(rec)

    paragraph = (
        "Across the three dataset themes (marine_science, geology, anthropology), Sobel with an Otsu "
        "threshold on gradient magnitude tends to mark more pixels as edges and can fragment structure, "
        "whereas Canny (100, 200) usually produces thinner, more localized boundaries with different "
        "connected-component statistics. "
        + " ".join(summary_parts)
    )
    print("Summary:")
    print(paragraph)
    print()


if __name__ == "__main__":
    main()
