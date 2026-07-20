"""
Plastic-warp normal map generator (bpy-FREE, Docker-testable).

Produces a subtle tangent-space normal map that simulates gentle, uneven plastic
undulation so sleeve/binder/display-case plastic reflects unevenly instead of
mirror-flat (spec §3.2, §3.5). It's a low-frequency smooth height field -> normal
map. Card-independent, so it's a pre-built asset (built once, reused everywhere).

Output convention: RGB PNG, R=+X, G=+Y, B=+Z (OpenGL-style), values [0,255] with
flat = (128,128,255). Load in Blender as Non-Color into a Normal Map node.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def _smooth_noise(h: int, w: int, cell: int, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency smooth field in [0,1] via cubic upsampling of a coarse grid."""
    ch = max(2, h // cell + 2)
    cw = max(2, w // cell + 2)
    coarse = rng.random((ch, cw)).astype(np.float32)
    return cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)


def height_field(
    w: int, h: int, seed: int,
    base_cell: int = 96, octaves: int = 2, persistence: float = 0.5,
) -> np.ndarray:
    """Multi-octave smooth height field in [0,1] (biggest undulation dominates)."""
    rng = np.random.default_rng(seed)
    field = np.zeros((h, w), np.float32)
    amp, cell, total = 1.0, base_cell, 0.0
    for _ in range(max(1, octaves)):
        field += amp * _smooth_noise(h, w, max(4, cell), rng)
        total += amp
        amp *= persistence
        cell = max(4, cell // 2)
    field /= total
    return field


def generate_plastic_warp_normal(
    w: int = 512, h: int = 512, seed: int = 0,
    strength: float = 1.0, base_cell: int = 320, octaves: int = 1,
    smooth_sigma: float = 10.0,
) -> np.ndarray:
    """Return an (h, w, 3) uint8 RGB normal map. `strength` scales the tilt; keep
    it modest so the plastic is subtly uneven, not visibly bumpy. Large `base_cell`
    + single octave => a FEW broad deflections rather than many small waves;
    `smooth_sigma` blurs the height field so undulations stay broad."""
    hf = height_field(w, h, seed, base_cell=base_cell, octaves=octaves, persistence=0.4)
    if smooth_sigma > 0:
        k = int(smooth_sigma * 4) | 1  # odd kernel
        hf = cv2.GaussianBlur(hf, (k, k), smooth_sigma)
    # Gradients (Sobel is smooth & fast). Scale so `strength` controls tilt size.
    gx = cv2.Sobel(hf, cv2.CV_32F, 1, 0, ksize=5) * strength
    gy = cv2.Sobel(hf, cv2.CV_32F, 0, 1, ksize=5) * strength
    nx, ny, nz = -gx, -gy, np.ones_like(hf)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    rgb = np.stack([nx, ny, nz], axis=-1)          # [-1,1]
    rgb = np.clip(rgb * 0.5 + 0.5, 0.0, 1.0)       # [0,1], flat -> (0.5,0.5,1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def save_normal_map(arr_rgb: np.ndarray, path: str) -> str:
    """Write an RGB normal-map array to PNG (cv2 wants BGR, so swap on write)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cv2.imwrite(path, arr_rgb[:, :, ::-1])
    return path


def get_or_build_warp_map(
    path: str, w: int = 512, h: int = 512, seed: int = 0, **kw
) -> str:
    """Cache helper: build the normal map at `path` once, reuse thereafter."""
    if not os.path.isfile(path):
        save_normal_map(generate_plastic_warp_normal(w, h, seed, **kw), path)
    return path
