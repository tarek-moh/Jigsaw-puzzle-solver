# Jigsaw Puzzle Preprocessing and Assembly Framework

This project implements a classical image processing pipeline for solving grid-based jigsaw puzzles using edge extraction, similarity metrics, and greedy clustering strategies. The solution explicitly avoids machine learning and relies only on deterministic image processing techniques.

## Table of Contents
- [Overview](#overview)
- [Preprocessing Pipeline](#Preprocessing-Pipeline)
- [Edge Representation and Matching Metrics](#Edge-Representation-and-Matching-Metrics)
- [Assembly and Ordering Strategies](#Assembly-and-Ordering-Strategies)
- [Limitations](#Limitations)
- [Dataset Assumptions](#Dataset-Assumptions)
- [Web Interface](#Web-Interface)
- [References](#References)
- [Future Work](#future-work)

---

## Overview

The goal of this project is to reconstruct jigsaw puzzles (2×2, 4×4, and 8×8) composed of square cartoon pieces. The system is divided into two logical phases:
- Phase 1 – Preprocessing: Extract puzzle pieces and normalize their edges into a structured, comparable form.
- Phase 2 – Assembly: Match edges and assemble the puzzle using greedy and cluster-based ordering strategies.

> **Note:** This preprocessing is purely **classical image processing**. No machine learning or heavy enhancement filters are used to avoid losing detail in cartoon-style puzzle pieces.
This README primarily documents the preprocessing pipeline while also outlining the downstream assembly approaches enabled by it.

---

## Preprocessing Pipeline

The preprocessing pipeline consists of the following steps:

1.  **Image Reading:** Load the main puzzle image using OpenCV.
2.  **Image Division:** Split the full puzzle into a grid (2×2, 4×4, 8×8) based on the number of pieces.
3.  **Edge Strip Extraction:** For each piece, extract the top, right, bottom, and left borders.
4.  **Edge Normalization:** Rotate and flip edge strips so all borders share a common vertical orientation.
5.  **Optional (weighted later) Feature Computation:**
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

## Edge Representation and Matching Metrics

The preprocessing stage prepares data for multiple classical similarity metrics:
-Sum of Squared Differences (SSD):
Measures absolute color similarity.
-Normalized Cross-Correlation:
Captures texture similarity while being invariant to brightness shifts.
-Gradient Error (Sobel-based):
Evaluates structural continuity across edges.


---

## Assembly and Ordering Strategies

The preprocessing outputs enable multiple puzzle assembly strategies:

1- Cluster-Greedy Assembly with Best-Buddy Constraints
- Each piece starts as its own cluster.
- Only mutually best-matching edges (“best buddies”) are merged.
- Clusters grow incrementally while enforcing spatial consistency.
- Significantly more robust to local ambiguities.
2- Sequential Piece-Greedy Placement
- Starts from a seed piece.
- Adds one piece at a time based on the highest edge similarity.
- Fast but prone to early irreversible mistakes.

Justification:
The cluster-based approach delays commitment, reduces false matches on low-information edges, and produces more stable assemblies.
---

## Limitations

### Low-Entropy Edge Ambiguity
Edges with little visual variation (e.g., solid colors) often produce artificially high similarity scores. Although edge entropy was investigated as a weighting or filtering mechanism, it proved difficult to tune reliably across all puzzles within the project scope and was not fully integrated.

### No Backtracking
The greedy nature of the assembly algorithms means that incorrect early merges cannot always be undone. This is a known trade-off for maintaining reasonable runtime.

## Dataset Assumptions
- Square puzzles only
- Equal-sized pieces
- No rotated or flipped pieces

## Web Interface
A lightweight web interface is provided to visualize puzzle pieces, edge matches, and reconstruction results:
UI URL:
https://piecewise.streamlit.app/
The interface allows interactive inspection of preprocessing outputs and final assembly results.

## References
Gonzalez & Woods, Digital Image Processing

## Future work
- [ ] **Entropy based prioritizing:** Humans solve puzzles by matching edges with higher texture first then blank ones, we need to mimic this behaviour.
