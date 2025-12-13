# Jigsaw Solver Ordering Logic

Based on the computed metrics (SSD, Correlation, Gradient), One approach to assembling the puzzle is to use a **Greedy Best-First Strategy with "Best Buddy" Verification**. This is likely more robust than filling from top-left to bottom-right.

## 1. Metric Normalization and Combination
Since metrics vary in scale (e.g., SSD is large, Correlation is -1 to 1), normalize them before combining.

*   **Normalization**: Map all metrics to a [0, 1] range where 1 is the best match.
    *   *SSD & Gradient*: $Norm = 1 - (value / max\_observed\_value)$
    *   *Correlation*: $Norm = (value + 1) / 2$
*   **Weighted Score**: Combine them into a single confidence score.
    *   $Final\_Score = w_1 \cdot Norm_{SSD} + w_2 \cdot Norm_{Corr} + w_3 \cdot Norm_{Grad}$
    *   Start with equal weights or give higher weight to Gradient for edge alignment.

## 2. "Best Buddy" Constraint
Before accepting a match, verify it is mutual (symmetric best match). This drastically reduces false positives for generic pieces (like sky/grass).

*   **Condition**:
    1.  Piece A's *Right* edge best matches Piece B's *Left* edge.
    2.  **AND** Piece B's *Left* edge best matches Piece A's *Right* edge.
*   **Usage**: Only allow "Best Buddy" matches in the early stages of assembly.

## 3. The Algorithm: Priority-Based Merging
Instead of a grid search, view this as merging clusters of pieces.

1.  **Compute All Pairs**: Calculate the `Final_Score` for every possible connection ($N \times 4$ edges).
2.  **Initialize Clusters**: Each piece starts as its own cluster of size 1.
3.  **Priority Queue**: Put all valid connections into a list sorted by `Final_Score` (descending).
4.  **Assembly Loop**:
    *   Pop the highest-score connection (e.g., Piece A Right + Piece B Left).
    *   **Check Availability**:
        *   Are A and B already in the same cluster? (If so, check if the relative position is consistent. If consistent, skip; if conflict, reject).
        *   Do they belong to different clusters?
            *   **Merge**: Transform B's cluster coordinates to align with A.
            *   **Conflict Check**: Does merging cause any piece overlaps? If yes, reject.
            *   **Commit**: Combine the two clusters into one.
    *   Repeat until all pieces are in one cluster.

## Why this approach?
*   **Texture Agnostic**: It works well even if parts of the image are featureless, because it relies on the *relative* strength of matches (Best Buddies) rather than absolute thresholds.
*   **Error Correction**: By prioritizing the highest scores globally, you solve the "easiest" parts first (e.g., distinct faces or text), creating islands of correctness that eventually merge.
