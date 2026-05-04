"""Reusable image loading helpers for the term project."""

import os
from typing import List, Tuple

import cv2
import numpy as np


def load_images_from_folder(folder_path: str) -> List[Tuple[str, np.ndarray]]:
    """Load all ``.jpg`` / ``.jpeg`` / ``.png`` images from a directory (any case).

    Parameters
    ----------
    folder_path:
        Absolute or relative path to a folder on disk.

    Returns
    -------
    list[tuple[str, numpy.ndarray]]
        ``(filename, image)`` pairs with BGR ``uint8`` images from OpenCV.
        Omits files that fail to decode. Returns an empty list if the folder
        is missing or contains no valid images.
    """
    if not os.path.isdir(folder_path):
        return []

    results: List[Tuple[str, np.ndarray]] = []
    for name in sorted(os.listdir(folder_path)):
        lower = name.lower()
        if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")):
            continue
        full_path = os.path.join(folder_path, name)
        image = cv2.imread(full_path, cv2.IMREAD_COLOR)
        if image is not None:
            results.append((name, image))
    return results
