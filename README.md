# CS136 Computer Vision Term Project

Repository: <https://github.com/2006-sk/CS136_Term-Project>

Team members: **Meera** and **Shresth**.

This project contains classical computer vision experiments for CS136. The code applies filtering, segmentation, edge detection, Hough transforms, robustness testing, and contrast-enhanced edge detection to scientific image datasets from marine biology, geology, and anthropology.

The repository is organized by contributor and algorithm. Meera's work focuses on Gaussian filtering, texture segmentation, and robustness analysis. Shresth's work focuses on Sobel/Canny edge detection, Hough line/circle detection, edge detector evaluation, and a creative CLAHE + Canny exploration.

## Project Goals

- Apply core computer vision techniques to real scientific imagery instead of only synthetic examples.
- Compare how preprocessing changes downstream edge and segmentation behavior.
- Generate visual outputs that make algorithm behavior easy to inspect.
- Quantify edge detector behavior using edge density, gradient strength, connected-component continuity, and robustness-style F1 comparisons.
- Keep shared data loading behavior simple and reusable across scripts.

## Repository Layout

```text
term_project/
|-- README.md
|-- requirements.txt
|-- datasets/
|   |-- Anthropology/
|   |-- Geology/
|   `-- MarineBiology/
|-- meera/
|   |-- gaussian_filter/
|   |   |-- README.md
|   |   |-- gaussian_filter.py
|   |   `-- output_images/
|   |-- texture_segmentation/
|   |   |-- README.md
|   |   |-- texture_segmentation.py
|   |   `-- output_images/
|   `-- robustness_analysis/
|       |-- README.md
|       |-- robustness_analysis.py
|       `-- output_images/
|-- shresth/
|   |-- Creative_Exploration_Images/
|   |   `-- creative.py
|   |-- Edge_Detection_Images/
|   |   `-- sobel_canny.py
|   |-- Edge_Detector_Evaluation_Images/
|   |   `-- edge_evaluation.py
|   `-- Hough_Transformation_Images/
|       `-- hough_transform.py
`-- utils/
    `-- image_loader.py
```

Important notes about the layout:

- The root folder for running commands is `term_project/`.
- `datasets/` contains the input images used by all scripts.
- `utils/image_loader.py` is the shared image loading helper used by Meera's scripts.
- Meera's scripts save generated files inside nested `output_images/` folders.
- Shresth's scripts save generated files directly into their corresponding `_Images` folders.
- `.gitkeep` files are present so empty output folders can remain tracked by Git.
- `.gitignore` excludes local Python caches, `.DS_Store`, `.pyc` files, and the local `.venv/`.

## Dataset Inventory

The repository currently includes 10 input images across three dataset themes.

### Marine Biology

Folder: `datasets/MarineBiology/`

- `32_T2_6_timepoint0.JPG` - 3532 x 3532
- `32_T2_6_timepoint1.JPG` - 3526 x 3526
- `33.5_T1_4_timepoint0.JPG` - 3636 x 3636
- `33.5_T1_4_timepoint1.JPG` - 3554 x 3554
- `35_T2_5_timepoint0.JPG` - 3532 x 3532
- `35_T2_5_timepoint1.JPG` - 3476 x 3476

### Geology

Folder: `datasets/Geology/`

- `coastline 1.JPG` - 1820 x 758
- `tahoe.JPG` - 1759 x 892

### Anthropology

Folder: `datasets/Anthropology/`

- `DW.jpg` - 930 x 727
- `T-101_DHEL-11__DHEL-12_20240802_BROOD.JPG` - 4000 x 3000

Supported image extensions are `.jpg`, `.jpeg`, and `.png`, case-insensitive. The scripts skip unreadable images rather than crashing.

## Installation

Use Python 3. The dependencies are intentionally small and are listed in `requirements.txt`:

- `opencv-python`
- `numpy`
- `matplotlib`
- `scipy`

Create and activate a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Running Everything

Run commands from `term_project/`, not from inside the individual module folders. The scripts use paths relative to the repository root.

```bash
python meera/gaussian_filter/gaussian_filter.py
python meera/texture_segmentation/texture_segmentation.py
python meera/robustness_analysis/robustness_analysis.py
python shresth/Edge_Detection_Images/sobel_canny.py
python shresth/Edge_Detector_Evaluation_Images/edge_evaluation.py
python shresth/Hough_Transformation_Images/hough_transform.py
python shresth/Creative_Exploration_Images/creative.py
```

If your machine's `python` command points to Python 2, use `python3` instead:

```bash
python3 meera/gaussian_filter/gaussian_filter.py
```

## Shared Image Loader

File: `utils/image_loader.py`

The shared helper `load_images_from_folder(folder_path)` returns a list of `(filename, image)` pairs.

Behavior:

- Accepts a folder path as input.
- Returns OpenCV BGR images as `uint8` NumPy arrays.
- Loads `.jpg`, `.jpeg`, and `.png` files in sorted filename order.
- Ignores non-image files.
- Skips images OpenCV cannot decode.
- Returns an empty list if the folder is missing or contains no valid images.

This keeps Meera's pipelines consistent and avoids duplicating image-loading logic.

## Meera's Modules

### Gaussian Filter

Files:

- `meera/gaussian_filter/README.md`
- `meera/gaussian_filter/gaussian_filter.py`

Purpose:

This module applies Gaussian smoothing to every image in the project datasets. Gaussian filtering is used to reduce high-frequency noise and demonstrate the visual effect of local weighted averaging before later processing steps.

Key parameters:

- Kernel size: `(5, 5)`
- Sigma: `1.5`
- Dataset labels: `Marine_Science`, `Geology`, `Anthropology`

Dataset folder behavior:

- For marine imagery, the script accepts several possible folder names: `Marine Science`, `MarineScience`, `MarineBiology`, and `Marine Biology`.
- For geology, it expects `datasets/Geology/`.
- For anthropology, it expects `datasets/Anthropology/`.

Run:

```bash
python meera/gaussian_filter/gaussian_filter.py
```

Outputs:

- Output root: `meera/gaussian_filter/output_images/`
- Filtered images: `<image_stem>_gaussian_filtered.png`
- Comparison panels: `<image_stem>_gaussian_comparison.png`

For each input image, the script writes:

- A standalone Gaussian-filtered result.
- A side-by-side Matplotlib preview comparing the original image to the filtered image.

Current generated output folders include:

- `meera/gaussian_filter/output_images/Anthropology/`
- `meera/gaussian_filter/output_images/Geology/`
- `meera/gaussian_filter/output_images/Marine_Science/`

### Texture Segmentation

Files:

- `meera/texture_segmentation/README.md`
- `meera/texture_segmentation/texture_segmentation.py`

Purpose:

This module performs texture-based segmentation using local grayscale, texture, edge-energy, and color features. It groups pixels into regions using k-means clustering and writes multiple visualizations for each input image.

Key parameters:

- Number of segments: `4`
- K-means attempts: `5`
- K-means termination: maximum 80 iterations or epsilon `0.2`
- Texture windowing: Gaussian blur with `(9, 9)` windows

Feature design:

- Grayscale intensity
- Local mean intensity
- Local standard deviation
- Laplacian edge energy
- LAB color channels for the color-aware segmentation pass

The script computes two label maps:

- Grayscale texture labels, based only on grayscale and local texture features.
- Color + texture labels, based on texture features plus LAB color features.

Run:

```bash
python meera/texture_segmentation/texture_segmentation.py
```

Outputs:

- Output root: `meera/texture_segmentation/output_images/`
- Grayscale texture labels: `<image_stem>_01_grayscale_texture_labels.png`
- Color + texture labels: `<image_stem>_02_color_texture_labels.png`
- Average-color segmented image: `<image_stem>_03_color_segmented.png`
- Three-panel preview: `<image_stem>_texture_segmentation_preview.png`

Current generated output folders include:

- `meera/texture_segmentation/output_images/Anthropology/`
- `meera/texture_segmentation/output_images/Geology/`
- `meera/texture_segmentation/output_images/Marine_Science/`

### Robustness Analysis

Files:

- `meera/robustness_analysis/README.md`
- `meera/robustness_analysis/robustness_analysis.py`
- `meera/robustness_analysis/output_images/robustness_analysis_summary.txt`

Purpose:

This module stress-tests edge detection and texture segmentation under common image distortions. It compares raw pipelines against Gaussian-preprocessed pipelines and measures stability against clean-image baselines.

Dataset scope:

- Uses up to the first 3 images per dataset category.
- Uses a deterministic random seed: `136`.

Distortions tested:

- `clean`
- `gaussian_noise`
- `blur`
- `low_contrast`

Pipelines compared:

- `raw`
- `gaussian_filtered`

Algorithms evaluated:

- Canny edge detection with automatic thresholds derived from grayscale median intensity.
- Sobel edge detection with Otsu thresholding on gradient magnitude.
- Texture segmentation boundaries from k-means segment labels.

Metrics:

- Edge density
- Precision against the matching clean baseline
- Recall against the matching clean baseline
- F1 score against the matching clean baseline
- Segmentation boundary F1 score

Run:

```bash
python meera/robustness_analysis/robustness_analysis.py
```

Outputs:

- Output root: `meera/robustness_analysis/output_images/`
- Distorted input images: `<distortion>.png`
- Canny maps: `<distortion>_<preprocess>_canny.png`
- Sobel maps: `<distortion>_<preprocess>_sobel.png`
- Segmentation maps: `<distortion>_<preprocess>_segmentation.png`
- Summary panels: `<distortion>_<preprocess>_summary.png`
- Written summary: `robustness_analysis_summary.txt`

Current summary results:

- Under Gaussian noise, raw Canny averaged `0.304` F1, raw Sobel averaged `0.578` F1, and raw segmentation boundaries averaged `0.220` F1.
- Under Gaussian noise, Gaussian preprocessing improved Canny to `0.684` F1, Sobel to `0.822` F1, and segmentation boundaries to `0.309` F1.
- Under blur, raw Canny averaged `0.171` F1 and raw Sobel averaged `0.595` F1.
- Under blur, Gaussian preprocessing improved Canny to `0.346` F1 and Sobel to `0.714` F1.
- Under low contrast, raw Sobel remained highly stable at `0.981` F1, while raw Canny averaged `0.483` F1.
- Overall, Gaussian preprocessing helped noisy and blurred inputs by smoothing small fluctuations before gradient-based detection.

## Shresth's Modules

### Sobel and Canny Edge Detection

File: `shresth/Edge_Detection_Images/sobel_canny.py`

Purpose:

This module applies Sobel and Canny edge detection to every image in the three dataset folders. It produces separate visual outputs for horizontal gradients, vertical gradients, combined gradient magnitude, and Canny edges.

Dataset folders:

- `datasets/MarineBiology/`
- `datasets/Geology/`
- `datasets/Anthropology/`

Algorithm details:

- Converts each input image to grayscale.
- Computes Sobel X with `cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)`.
- Computes Sobel Y with `cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)`.
- Computes combined magnitude with `cv2.magnitude(gx, gy)`.
- Runs Canny with thresholds `100` and `200`.

Run:

```bash
python shresth/Edge_Detection_Images/sobel_canny.py
```

Outputs:

- Output folder: `shresth/Edge_Detection_Images/`
- `plain_sobel_x_<dataset>_<image_stem>.png`
- `plain_sobel_y_<dataset>_<image_stem>.png`
- `plain_sobel_combined_<dataset>_<image_stem>.png`
- `plain_canny_<dataset>_<image_stem>.png`

### Edge Detector Evaluation

File: `shresth/Edge_Detector_Evaluation_Images/edge_evaluation.py`

Purpose:

This module quantitatively compares Sobel-Otsu and Canny edge detection. It prints a per-image metric table in the terminal and saves dataset-level bar charts.

Metrics:

- Edge density percentage
- Mean gradient magnitude at edge pixels
- Average connected-component area

Algorithm details:

- Sobel uses gradient magnitude followed by Otsu thresholding.
- Canny uses fixed thresholds `100` and `200`.
- Connected components are computed with 8-connectivity.
- Dataset-level means are plotted for each metric.

Run:

```bash
python shresth/Edge_Detector_Evaluation_Images/edge_evaluation.py
```

Outputs:

- Output folder: `shresth/Edge_Detector_Evaluation_Images/`
- `bar_marine_science.png`
- `bar_geology.png`
- `bar_anthropology.png`
- A printed terminal table containing per-image Sobel and Canny metrics.
- A printed written summary comparing detector behavior across the dataset themes.

Interpretation built into the script:

- Sobel-Otsu often marks more pixels as edges and can provide broader gradient coverage.
- Canny usually gives thinner, more localized boundaries because of non-maximum suppression and hysteresis thresholding.
- The better detector depends on whether the goal is dense gradient coverage or crisp contours.

### Hough Transformation

File: `shresth/Hough_Transformation_Images/hough_transform.py`

Purpose:

This module detects geometric primitives in the dataset images using Hough transforms. It detects circles with `cv2.HoughCircles` and line segments with probabilistic Hough lines.

Preprocessing:

- Large images are resized so their maximum side is at most `800` pixels.
- The resized image is blurred with a `(9, 9)` Gaussian kernel.
- Hough detections are computed on the resized image for speed.
- Detected circles and lines are scaled back onto copies of the original-resolution image.

Circle detection:

- Method: `cv2.HOUGH_GRADIENT`
- `dp=1.5`
- `param1=80`
- `param2=28`
- Minimum radius: `1%` of the smaller resized image dimension, at least `5` pixels
- Maximum radius: `35%` of the smaller resized image dimension

Line detection:

- Canny preprocessing thresholds: `50` and `150`
- Method: `cv2.HoughLinesP`
- `rho=1`
- `theta=np.pi / 180`
- Threshold: max of `30` and `12%` of the smaller resized image dimension
- Minimum line length: `4%` of the smaller resized image dimension
- Maximum line gap: `12`

Run:

```bash
python shresth/Hough_Transformation_Images/hough_transform.py
```

Outputs:

- Output folder: `shresth/Hough_Transformation_Images/`
- `hough_circles_<dataset>_<image_stem>.png`
- `hough_lines_<dataset>_<image_stem>.png`
- Terminal messages reporting the number of circles and line segments detected per image.

### Creative Exploration: CLAHE + Canny

File: `shresth/Creative_Exploration_Images/creative.py`

Purpose:

This module compares plain Canny edge detection against Canny after contrast-limited adaptive histogram equalization. CLAHE is used as a local contrast enhancement step before edge detection.

Why CLAHE:

CLAHE improves local contrast in small image regions instead of applying one global contrast transformation. This can reveal faint boundaries in low-light, unevenly illuminated, or low-contrast scientific images.

Parameters:

- CLAHE clip limit: `2.0`
- CLAHE tile grid size: `(8, 8)`
- Canny thresholds: `100` and `200`

Run:

```bash
python shresth/Creative_Exploration_Images/creative.py
```

Outputs:

- Output folder: `shresth/Creative_Exploration_Images/`
- `plain_canny_<dataset>_<image_stem>.png`
- `clahe_canny_<dataset>_<image_stem>.png`

Observations included in the script:

- Marine biology images can benefit when underwater regions are dark.
- Geology images can show clearer faint layer boundaries.
- Anthropology images can reveal low-contrast texture patterns more consistently.
- CLAHE can improve Canny output without changing the detector itself.

## Generated Outputs

Generated PNGs are already present in the repository. These outputs are useful for visual inspection, but they can also be regenerated by rerunning the scripts.

Current generated output summary:

- `meera/gaussian_filter/output_images/` contains Gaussian-filtered images and side-by-side comparisons.
- `meera/texture_segmentation/output_images/` contains grayscale texture labels, color texture labels, average-color segmentations, and preview panels.
- `meera/robustness_analysis/output_images/` contains distorted images, edge maps, segmentation maps, summary panels, and a written robustness summary.
- `shresth/Edge_Detection_Images/` contains Sobel X, Sobel Y, combined Sobel, and Canny outputs.
- `shresth/Edge_Detector_Evaluation_Images/` contains edge detector evaluation bar charts.
- `shresth/Hough_Transformation_Images/` contains Hough circle and Hough line outputs.
- `shresth/Creative_Exploration_Images/` contains plain Canny and CLAHE-enhanced Canny outputs.

## File Naming Conventions

The scripts sanitize names differently depending on contributor:

- Meera's scripts replace non-alphanumeric filename characters with underscores using a `_safe_name(...)` helper.
- Shresth's scripts preserve the dataset folder name and original image stem when building output names.
- Most generated files include both the dataset theme and the original image stem so outputs can be traced back to their source image.

Examples:

- `DW_gaussian_filtered.png`
- `DW_texture_segmentation_preview.png`
- `gaussian_noise_gaussian_filtered_summary.png`
- `plain_sobel_combined_Geology_tahoe.png`
- `hough_lines_MarineBiology_32_T2_6_timepoint0.png`
- `clahe_canny_Anthropology_DW.png`

## Troubleshooting

### No Images Were Processed

Make sure the dataset folders exist under `datasets/` and contain `.jpg`, `.jpeg`, or `.png` files.

Expected folders for Shresth's scripts:

```text
datasets/MarineBiology/
datasets/Geology/
datasets/Anthropology/
```

Meera's scripts also accept these marine folder aliases:

```text
datasets/Marine Science/
datasets/MarineScience/
datasets/MarineBiology/
datasets/Marine Biology/
```

### Import Errors

Activate the virtual environment and reinstall dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Wrong Python Version

On some macOS systems, `python` may point to Python 2. Use `python3` or activate the virtual environment before running scripts.

### Running From the Wrong Directory

Run scripts from the repository root:

```bash
cd term_project
python meera/gaussian_filter/gaussian_filter.py
```

Avoid running scripts from inside their subfolders unless you first update paths, because the code assumes the project root structure.

## Git Notes

This project is connected to GitHub at:

```text
https://github.com/2006-sk/CS136_Term-Project
```

The local repository uses `origin` for the CS136 project remote. There may also be other remotes configured locally, but `origin` is the remote for this project.

Useful commands:

```bash
git remote -v
git status
git pull --ff-only origin main
```

## Maintenance Notes

- Keep dependencies in `requirements.txt`.
- Keep generated folders tracked with `.gitkeep` if they are otherwise empty.
- Do not commit `.venv/`, `__pycache__/`, `.pyc`, or `.DS_Store` files.
- If dataset folder names change, update the folder constants in the scripts.
- If output naming changes, update this README so run instructions and generated artifact descriptions remain accurate.
