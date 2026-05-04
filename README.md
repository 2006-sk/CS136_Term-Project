# CS136 Computer Vision Term Project

Team members: **Meera** and **Shresth**.

This repository contains coursework experiments for CS136, organized by contributor and algorithm. Shared dataset folders live under `datasets/`, and each module writes artifacts to its local `output_images/` directory.

## Repository layout

- `datasets/` — place `.jpg` / `.png` imagery under `marine_science/`, `geology/`, or `anthropology/` (tracked via `.gitkeep` until you add files).
- `meera/` — Gaussian filtering, texture segmentation, and robustness analysis pipelines.
- `shresth/` — edge detection, Hough transforms, edge evaluation, and a creative exploration script.
- `utils/` — shared helpers such as `image_loader.load_images_from_folder`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running a module

From the repository root (`term_project/`), run any script with Python using a path relative to this folder. Examples:

```bash
python meera/gaussian_filter/gaussian_filter.py
python meera/texture_segmentation/texture_segmentation.py
python meera/robustness_analysis/robustness_analysis.py
python shresth/edge_detection/sobel_canny.py
python shresth/hough_transformation/hough_transform.py
python shresth/edge_detector_evaluation/edge_evaluation.py
python shresth/creative_exploration/creative.py
```

Each script loads images from `datasets/` via the shared loader, prints a friendly message if no images are present, and saves previews or intermediate outputs next to the script under `output_images/`.
