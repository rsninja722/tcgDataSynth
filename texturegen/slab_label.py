"""
Procedural graded-slab label placeholder (bpy-FREE, Docker-testable).

Generates a grading-label-looking image: a colored header bar, a white body with
dark "text"-like marks, and a grade box. Not meant to be legible -- the detector
only cares about card corners (spec §3.2: a procedural placeholder is fine, user-
confirmed). Card-independent => pre-built asset, randomly selected per slab.
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def generate_slab_label(w: int = 680, h: int = 200, seed: int = 0) -> np.ndarray:
    """Return an (h, w, 3) uint8 BGR label image (BGR so cv2.imwrite is a no-op swap)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 238, np.uint8)  # off-white body

    # Header bar (random branded color).
    hdr_h = int(h * rng.uniform(0.30, 0.42))
    hdr_color = tuple(int(c) for c in rng.integers(30, 200, size=3))
    img[:hdr_h] = hdr_color
    # Light "title" marks on the header.
    for _ in range(int(rng.integers(2, 4))):
        x0 = int(rng.integers(10, w // 2))
        y0 = int(rng.integers(4, max(6, hdr_h - 14)))
        cv2.rectangle(img, (x0, y0), (x0 + int(rng.integers(40, 160)), y0 + 10),
                      (235, 235, 235), -1)

    # Body: rows of dark text-like marks (card/grade info).
    y = hdr_h + 12
    while y < h - 24:
        x = 14
        for _ in range(int(rng.integers(2, 6))):
            wlen = int(rng.integers(30, 130))
            cv2.rectangle(img, (x, y), (x + wlen, y + 8), (40, 40, 40), -1)
            x += wlen + int(rng.integers(10, 26))
            if x > w - 140:
                break
        y += int(rng.integers(16, 26))

    # Grade box (big number placeholder) on the right.
    bx0, by0 = w - 120, hdr_h + 14
    cv2.rectangle(img, (bx0, by0), (w - 16, h - 14), (60, 60, 60), 2)
    cv2.rectangle(img, (bx0 + 18, by0 + 14), (w - 34, h - 30), (35, 35, 35), -1)
    return img


def save_label(arr_bgr: np.ndarray, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cv2.imwrite(path, arr_bgr)
    return path
