# Satellite-Object-Localization-using-SIFT
A lightweight computer vision tool built from scratch in Python to detect, locate, and outline a specific visual element (like a specific building or asset) inside a larger, high-resolution satellite or drone scene. 
Unlike basic template matching which breaks under zoom alterations, this project uses **Scale-Invariant Feature Transform (SIFT)** and **Homography** to successfully map objects even when they are heavily scaled, rotated, or slightly distorted.

## Features
* **Scale & Rotation Invariant:** Successfully locates targets regardless of resolution zoom or orientation adjustments.
* **Lowe's Ratio Testing:** Implements feature validation to drop false-positives caused by repetitive landscape elements (e.g., roads, uniform trees, grass).
* **Outlier Rejection:** Leverages the RANSAC algorithm to cleanly generate an exact bounding framework around the identified item.

## Tech Stack
* **Language:** Python 3
* **Libraries:** OpenCV (`opencv-python`), NumPy

## How it Works
1. **Feature Extraction:** SIFT discovers key architectural vectors, corners, and textures across both image datasets.
2. **K-Nearest Neighbors Matching:** Features are loosely mapped using a FLANN-based tree index.
3. **Filtering:** Lowe's Ratio Test discards ambiguous or multi-matching coordinate markers.
4. **Homography Warp:** `cv2.findHomography` maps the matching points mathematically to draw a precise, angled green border over the target destination image.
