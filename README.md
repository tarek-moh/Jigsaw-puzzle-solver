# Jigsaw Puzzle Preprocessing

This module implements the preprocessing pipeline required for extracting puzzle pieces and preparing their edges for later matching and assembly.

## Table of Contents
- [Overview](#overview)
- [Pipeline](#pipeline)
- [Techniques Used](#techniques-used)
- [Outputs](#outputs)
- [Future Work](#future-work)

---

## Overview

Given a complete puzzle image, the goal of this phase is to divide the image into equal-size pieces and extract normalized edge strips from each piece. These strips will later be used to compare borders, evaluate similarity, and reconstruct the puzzle in **Phase 2**.

> **Note:** This preprocessing is purely **classical computer vision**. No machine learning or heavy enhancement filters are used to avoid losing detail in cartoon-style puzzle pieces.

---

## Pipeline

The preprocessing pipeline consists of the following steps:

1.  **Image Reading:** Load the main puzzle image using OpenCV.
2.  **Image Division:** Split the full puzzle into a grid (2×2, 4×4, 8×8) based on the number of pieces.
3.  **Edge Strip Extraction:** For each piece, extract the top, right, bottom, and left borders.
4.  **Normalization:** Rotate and flip edge strips so all borders share a common vertical orientation.
5.  **Optional Feature Computation:**
    * Gradient extraction (Sobel)
    * Strip normalization (z-scores)
    * Baseline edge similarity metrics (SSD, correlation)

*No artificial image enhancement is applied because the provided images are already sharp and visually clean.*



---

## Techniques Used

### 1. Image Reading and Visualization
* Loaded using standard CV libraries.
* Verified visually with plotting tools.
* Ensures consistent format before division.

### 2. Puzzle Piece Division
* Custom logic ensures:
    * Input piece count is a perfect square.
    * Image dimensions match the expected grid.
    * Produces uniformly sized tiles.

### 3. Edge Strip Extraction
* Extracts borders of each piece.
* All strips are normalized to a **vertical orientation** for consistent array indexing.

### 4. Feature Preparation
* **Gradient computation (Sobel):** Used for structure-sensitive comparisons.
* **Metrics:** SSD, correlation, and gradient error metrics are prepared for Phase 2.
* **Constraint:** No image smoothing or CLAHE is used to avoid detail loss.

---

## Outputs

The preprocessing pipeline produces:

* A list of uniformly cropped puzzle pieces.
* Four normalized edge strips per piece.
* Optional gradient representations.
* Clean, reusable data structures for Phase 2 matching.

These outputs are directly compatible with SSD, correlation, and gradient-based edge similarity functions.

---

## Future Work

- [ ] **Automatic noise detection:** Apply enhancement only when needed instead of globally.
- [ ] **Piece rotation handling:** Extend preprocessing to account for rotated or flipped puzzle pieces.
