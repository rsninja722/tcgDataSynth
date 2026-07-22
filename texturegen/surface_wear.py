"""
Surface wear map generator (bpy-FREE, Docker-testable).

Produces a grayscale "wear" map for used rigid plastic (toploaders): faint random
MICRO-SCRATCHES (thin lines) + DUST specks. Bright = worn/scratched/dusty, black =
clean. Used in Blender to modulate the clear plastic's roughness (worn areas go
rougher and catch the light), so a toploader looks handled, not pristine.

Card-independent => pre-built asset, reused across all toploaders.
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def generate_scratch_dust(
    w: int = 1024, h: int = 1024, seed: int = 0,
    n_scratches: int = 450, n_dust: int = 600,
    len_frac=(0.03, 0.45), blur: float = 0.6,
) -> np.ndarray:
    """Return (h, w) uint8 grayscale wear map (mostly black with faint scratches/dust).
    `len_frac` = (min, max) scratch length as a fraction of `w` (smaller => finer)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)

    # Micro-scratches: thin faint lines, varied length/angle.
    for _ in range(n_scratches):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        ang = float(rng.uniform(0.0, np.pi))
        length = float(rng.uniform(*len_frac)) * w
        x1 = int(x0 + np.cos(ang) * length)
        y1 = int(y0 + np.sin(ang) * length)
        inten = float(rng.uniform(0.06, 0.45))
        cv2.line(img, (x0, y0), (x1, y1), inten, 1, cv2.LINE_AA)

    # Dust specks: small bright dots.
    for _ in range(n_dust):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        rad = int(rng.integers(1, 3))
        inten = float(rng.uniform(0.12, 0.55))
        cv2.circle(img, (cx, cy), rad, inten, -1, cv2.LINE_AA)

    img = cv2.GaussianBlur(img, (0, 0), blur)
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def generate_case_cover_wear(
    w: int = 2048, h: int = 2048, seed: int = 0,
    n_scratches: int = 12000, n_dust: int = 6000,
) -> np.ndarray:
    """Wear map for a display-case COVER: a big, heavily-handled acrylic lid shows
    MANY more (but individually SMALL) scratches/smudges than a toploader. Higher
    resolution (2048 vs the toploader's 1024), far higher count, and SHORT scratches
    (len 0.4-3% of width) so they read as fine hairlines on the large lid, not gouges."""
    return generate_scratch_dust(w=w, h=h, seed=seed, n_scratches=n_scratches,
                                 n_dust=n_dust, len_frac=(0.004, 0.03))


def save_gray(arr: np.ndarray, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cv2.imwrite(path, arr)
    return path


if __name__ == "__main__":
    # Bake the display-case cover wear set (6 variants, 2048, denser). Toploader maps
    # (toploader_wear_*.png) are already present and are NOT touched here.
    _here = os.path.dirname(os.path.abspath(__file__))
    _assets = os.path.join(_here, "..", "assets")
    for _i in range(6):
        _p = save_gray(generate_case_cover_wear(seed=100 + _i),
                       os.path.join(_assets, f"case_cover_wear_{_i}.png"))
        print(f"baked {os.path.basename(_p)} (2048x2048)")
