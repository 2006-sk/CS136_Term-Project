"""Hough transform line and circle detection for the CS136 term project.

Scans every image under the three dataset theme folders (on disk: MarineBiology,
Geology, Anthropology). Output PNGs are generated per image at runtime in this
module's output_images/ directory (hough_circles_*, hough_lines_*); nothing is
hardcoded to a fixed count.
"""

import os

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_DATASET_SUBDIRS = ("MarineBiology", "Geology", "Anthropology")


def _is_image_filename(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".jpg", ".jpeg", ".png"))


def discover_all_dataset_images():
    """Walk each dataset subfolder with os.listdir; yield (rel_key, filename, BGR image)."""
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
            yield f"{sub}/{fname}", fname, bgr


def _resize_for_hough(bgr: np.ndarray, max_side: int = 800):
    """Downscale large images so Hough runs in reasonable time; return (bgr_small, sx, sy)."""
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr.copy(), 1.0, 1.0
    scale = max_side / float(m)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    sx, sy = w / float(nw), h / float(nh)
    return small, sx, sy


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_images")
    os.makedirs(out_dir, exist_ok=True)

    images = list(discover_all_dataset_images())
    if not images:
        print("No images found under datasets/. Expected MarineBiology, Geology, Anthropology.")
        return

    for rel_key, filename, bgr in images:
        small_bgr, sx, sy = _resize_for_hough(bgr, max_side=800)
        blurred = cv2.GaussianBlur(small_bgr, (9, 9), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        min_r = max(5, int(0.01 * min(h, w)))
        max_r = int(0.35 * min(h, w))

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=max(min_r * 2, 20),
            param1=80,
            param2=28,
            minRadius=min_r,
            maxRadius=max_r,
        )

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(30, int(0.12 * min(h, w))),
            minLineLength=int(0.04 * min(h, w)),
            maxLineGap=12,
        )

        sub, _sep, base_name = rel_key.partition("/")
        base, _ = os.path.splitext(base_name)
        stem = f"{sub}_{base}"

        circles_img = bgr.copy()
        n_circles = 0
        if circles is not None:
            circles_int = np.around(circles[0, :]).astype(np.float32)
            n_circles = len(circles_int)
            for (cx, cy, r) in circles_int:
                fx = int(round(cx * sx))
                fy = int(round(cy * sy))
                fr = int(round(r * (sx + sy) / 2.0))
                cv2.circle(circles_img, (fx, fy), max(fr, 1), (0, 255, 0), 2)

        lines_img = bgr.copy()
        n_lines = 0
        if lines is not None:
            n_lines = len(lines)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(
                    lines_img,
                    (int(round(x1 * sx)), int(round(y1 * sy))),
                    (int(round(x2 * sx)), int(round(y2 * sy))),
                    (0, 0, 255),
                    2,
                )

        cv2.imwrite(os.path.join(out_dir, f"hough_circles_{stem}.png"), circles_img)
        cv2.imwrite(os.path.join(out_dir, f"hough_lines_{stem}.png"), lines_img)

        print(f"{rel_key}: detected {n_circles} circles, {n_lines} line segments.")


if __name__ == "__main__":
    main()
