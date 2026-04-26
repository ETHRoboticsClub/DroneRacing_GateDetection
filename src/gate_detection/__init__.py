"""Gate detection for autonomous drone racing.

Two-stage pipeline:
    1. GateNet  — U-Net segmentation (gate region → binary mask).
    2. QuAdGate — classical post-processing that extracts 8 sub-pixel
                  gate corners (4 outer + 4 inner) from the mask via
                  LSD line detection, descriptor matching, and RANSAC.
"""

__version__ = "0.1.0"
