"""
Holo region masks + patterns (bpy-FREE, Docker-testable). Spec §3.4.

- Region mask: which part of the card is holo — 'entire', 'picture' (the art box),
  or 'reverse' (everything EXCEPT the art box). Picture region comes per-card as
  normalized top-down coords (x0,y0,x1,y1); NOT hardcoded.
- Holo pattern within the region: 'none' (smooth foil), 'cosmos' (cloudy sparkle),
  or 'horizontal_lines'. Returned as grayscale maps that drive the Blender holo
  shader (roughness/normal/emission modulation).
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

HOLO_REGIONS = ("entire", "picture", "reverse")
HOLO_PATTERNS = ("none", "cosmos", "horizontal_lines", "water_web")


def _smooth_noise(h, w, cell, rng):
    coarse = rng.random((max(2, h // cell), max(2, w // cell))).astype(np.float32)
    n = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    return (n - n.min()) / (np.ptp(n) + 1e-6)


def region_mask(w: int, h: int, picture_region: Tuple[float, float, float, float],
                mode: str) -> np.ndarray:
    """Return (h, w) uint8 mask (255 = holo, 0 = not). `picture_region` is
    (x0,y0,x1,y1) normalized, y measured top-down."""
    if mode == "entire":
        return np.full((h, w), 255, np.uint8)
    x0, y0, x1, y1 = picture_region
    px0, px1 = int(round(x0 * w)), int(round(x1 * w))
    py0, py1 = int(round(y0 * h)), int(round(y1 * h))
    box = np.zeros((h, w), np.uint8)
    box[max(0, py0):min(h, py1), max(0, px0):min(w, px1)] = 255
    if mode == "picture":
        return box
    if mode == "reverse":
        return 255 - box
    raise ValueError(f"bad holo region mode {mode!r}; expected {HOLO_REGIONS}")


def pattern_smooth(w: int, h: int) -> np.ndarray:
    """Uniform mid-grey: smooth foil, no pattern."""
    return np.full((h, w), 128, np.uint8)


def pattern_horizontal_lines(w: int, h: int, period_px: int = 6) -> np.ndarray:
    """Horizontal stripes (sinusoidal so they read as fine foil lines)."""
    yy = np.arange(h, dtype=np.float32).reshape(-1, 1)
    line = (0.5 + 0.5 * np.sin(2.0 * np.pi * yy / max(2, period_px)))
    return (np.tile(line, (1, w)) * 255.0).astype(np.uint8)


def pattern_cosmos(w: int, h: int, seed: int = 0) -> np.ndarray:
    """'Cosmos' holo: many different-sized soft circles + additional smaller
    PIXELATED (aliased) circles over a faint cloudy base."""
    rng = np.random.default_rng(seed)
    img = 0.25 + 0.2 * _smooth_noise(h, w, 24, rng)     # faint cloud base
    area = w * h
    # Big soft circles, varied sizes.
    for _ in range(area // 2200):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(6, 30))
        cv2.circle(img, (cx, cy), r, float(rng.uniform(0.35, 0.9)), -1, cv2.LINE_AA)
    # Smaller PIXELATED circles (no anti-aliasing -> blocky).
    for _ in range(area // 700):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(1, 6))
        cv2.circle(img, (cx, cy), r, float(rng.uniform(0.5, 1.0)), -1)  # no LINE_AA
    return (np.clip(img, 0, 1) * 255.0).astype(np.uint8)


def pattern_water_web(w: int, h: int, seed: int = 0) -> np.ndarray:
    """'Water web' holo: subtle wavy interconnected web lines (the level-set ridges
    of a low-frequency field, domain-warped for waviness)."""
    rng = np.random.default_rng(seed)
    base = _smooth_noise(h, w, 46, rng)
    warp = (_smooth_noise(h, w, 70, rng) - 0.5) * 12.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = cv2.remap(base, (xx + warp).astype(np.float32), (yy + warp).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    web = 1.0 - np.abs(2.0 * base - 1.0)                 # ridges at the mid level set
    web = np.clip((web - 0.72) / 0.28, 0.0, 1.0)          # thin web lines
    web = cv2.GaussianBlur(web, (0, 0), 0.8)
    out = 0.35 + 0.65 * web                               # subtle, lifted floor
    return (np.clip(out, 0, 1) * 255.0).astype(np.uint8)


def holo_pattern(w: int, h: int, pattern: str, seed: int = 0) -> np.ndarray:
    if pattern == "none":
        return pattern_smooth(w, h)
    if pattern == "horizontal_lines":
        return pattern_horizontal_lines(w, h)
    if pattern == "cosmos":
        return pattern_cosmos(w, h, seed)
    if pattern == "water_web":
        return pattern_water_web(w, h, seed)
    raise ValueError(f"bad holo pattern {pattern!r}; expected {HOLO_PATTERNS}")


def pattern_normal(pattern_gray: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """Normal map from a grayscale pattern (so the etched texture bends the holo)."""
    from texturegen.physical_texture import height_to_normal
    return height_to_normal(pattern_gray.astype(np.float32) / 255.0, strength=strength)


def masked_pattern(w: int, h: int, picture_region, mode: str, pattern: str,
                   seed: int = 0) -> np.ndarray:
    """Pattern within the holo region, 0 elsewhere (grayscale (h,w) uint8)."""
    m = region_mask(w, h, picture_region, mode)
    p = holo_pattern(w, h, pattern, seed)
    out = p.copy()
    out[m == 0] = 0
    return out
