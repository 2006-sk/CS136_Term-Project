"""Hough transform line and circle detection for the CS136 term project."""

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
            all_items.append((f"{sub}/{name}", name, img))
    return all_items


def main() -> None:
    images = _load_all_dataset_images()
    out_dir = os.path.join(os.path.dirname(__file__), "output_images")
    os.makedirs(out_dir, exist_ok=True)

    if not images:
        print("No images found under datasets/. Expected MarineBiology, Geology, Anthropology.")
        return

    for rel_key, filename, bgr in images:
        blurred = cv2.GaussianBlur(bgr, (9, 9), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        min_r = max(5, int(0.01 * min(h, w)))
        max_r = int(0.35 * min(h, w))

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(min_r * 2, 20),
            param1=80,
            param2=22,
            minRadius=min_r,
            maxRadius=max_r,
        )

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(30, int(0.15 * min(h, w))),
            minLineLength=int(0.05 * min(h, w)),
            maxLineGap=12,
        )

        stem = os.path.splitext(rel_key.replace(os.sep, "_"))[0]

        circles_img = bgr.copy()
        n_circles = 0
        if circles is not None:
            circles_int = np.uint16(np.around(circles[0, :]))
            n_circles = len(circles_int)
            for (cx, cy, r) in circles_int:
                cv2.circle(circles_img, (int(cx), int(cy)), int(r), (0, 255, 0), 2)

        lines_img = bgr.copy()
        n_lines = 0
        if lines is not None:
            n_lines = len(lines)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite(os.path.join(out_dir, f"{stem}_hough_circles.png"), circles_img)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_hough_lines.png"), lines_img)

        print(f"{rel_key}: detected {n_circles} circles, {n_lines} line segments.")


if __name__ == "__main__":
    main()
