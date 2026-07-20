"""
Card damage overlays (bpy-FREE, Docker-testable). Spec §3.3.

Each is independently on/off at random and composited onto the card FACE texture:
  - dirt:    patchy grime/smudges.
  - scratches: fine random scratch lines.
  - surface: white-ish paper-texture whitening blobs, STRONGLY biased to edges/corners
             (where cards actually wear).

Returned as RGBA (h,w,4) uint8 overlays (alpha = strength); composite onto the base
card image with composite_overlays().

TRAINING DIVERSITY: unlike the (deterministic) physical-texture normal map and holo
masks — which may be cached per card ID — damage MUST vary per INSTANCE. The Blender
generation loop passes a fresh per-instance seed (derived from the scene seed) to
dirt()/scratches()/surface_damage() every time a card is placed, so the same card ID
shows different wear on each appearance and the model can't overfit to fixed damage.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def _smooth_noise(h, w, cell, rng):
    coarse = rng.random((max(2, h // cell), max(2, w // cell))).astype(np.float32)
    n = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    return (n - n.min()) / (np.ptp(n) + 1e-6)


def _edge_corner_bias(h, w, power: float = 2.0) -> np.ndarray:
    """Weight map ~0 in the center, ->1 at edges, strongest at corners."""
    ys = np.linspace(-1, 1, h, dtype=np.float32).reshape(-1, 1)
    xs = np.linspace(-1, 1, w, dtype=np.float32).reshape(1, -1)
    dx = np.abs(xs)
    dy = np.abs(ys)
    # Chebyshev-ish distance emphasizes edges; product term boosts corners.
    edge = np.maximum(dx, dy) ** power
    corner = (dx * dy)
    return np.clip(0.7 * edge + 0.6 * corner, 0.0, 1.0)


def dirt(w: int, h: int, seed: int = 0, strength: float = 0.5) -> np.ndarray:
    """Patchy grime overlay (brownish, semi-transparent). A LARGE-scale mask leaves
    whole regions clean so it's not a uniform layer of dirt."""
    rng = np.random.default_rng(seed)
    patches = _smooth_noise(h, w, 20, rng) * _smooth_noise(h, w, 7, rng)
    # Large-scale exclusion: entire areas get little/no dirt.
    large = _smooth_noise(h, w, max(8, min(h, w) // 4), rng)
    large_mask = np.clip((large - 0.4) / 0.35, 0.0, 1.0)
    alpha = np.clip((patches - 0.25) * 2.2, 0.0, 1.0) * large_mask * strength
    color = np.array([60, 50, 40], np.float32)  # BGR brownish
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    return rgba


def scratches(w: int, h: int, seed: int = 0, n: int = 60, strength: float = 0.6) -> np.ndarray:
    """Fine random scratch lines (bright, like abraded foil)."""
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w), np.float32)
    for _ in range(n):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        ang = float(rng.uniform(0, np.pi))
        length = float(rng.uniform(0.05, 0.5)) * w
        x1, y1 = int(x0 + np.cos(ang) * length), int(y0 + np.sin(ang) * length)
        cv2.line(a, (x0, y0), (x1, y1), float(rng.uniform(0.3, 1.0)), 1, cv2.LINE_AA)
    a = cv2.GaussianBlur(a, (0, 0), 0.5) * strength
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = 225
    rgba[..., 3] = np.clip(a * 255, 0, 255).astype(np.uint8)
    return rgba


def _edge_rip_triangle(rng, w, h, edge):
    """A triangular rip: base ON an edge, apex pointing inward."""
    base = int(rng.integers(8, 26))
    depth = int(rng.integers(16, 52))
    jitter = int(rng.integers(-8, 9))
    if edge == 0:      # top
        cx = int(rng.integers(0, w))
        return [(cx - base // 2, 0), (cx + base // 2, 0), (cx + jitter, depth)]
    if edge == 1:      # bottom
        cx = int(rng.integers(0, w))
        return [(cx - base // 2, h - 1), (cx + base // 2, h - 1), (cx + jitter, h - 1 - depth)]
    if edge == 2:      # left
        cy = int(rng.integers(0, h))
        return [(0, cy - base // 2), (0, cy + base // 2), (depth, cy + jitter)]
    cy = int(rng.integers(0, h))  # right
    return [(w - 1, cy - base // 2), (w - 1, cy + base // 2), (w - 1 - depth, cy + jitter)]


def surface_damage(w: int, h: int, seed: int = 0, n_blobs: int = 90,
                   strength: float = 0.8, edge_fraction: float = 1.0 / 3.0,
                   n_rips: int = 3) -> np.ndarray:
    """White-ish paper whitening (spec §3.3). `edge_fraction` of the blobs sit ON the
    nearest edge (edge whitening actually starts at the border); the rest are spread
    across the card. Plus a few triangular RIPS anchored to an edge. Vary `seed` per
    instance for training diversity (see module note)."""
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w), np.float32)
    n_edge = int(round(n_blobs * edge_fraction))
    n_uniform = n_blobs - n_edge

    # Blobs spread across the card.
    for _ in range(n_uniform):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(a, (x, y), int(rng.integers(3, 14)), float(rng.uniform(0.4, 1.0)),
                   -1, cv2.LINE_AA)

    # Edge blobs: centre on the nearest edge line (inset < r) so the blob TOUCHES it.
    for _ in range(n_edge):
        r = int(rng.integers(4, 16))
        edge = int(rng.integers(0, 4))
        inset = int(rng.uniform(0, r * 0.5))
        if edge == 0:      # top
            x, y = int(rng.integers(0, w)), inset
        elif edge == 1:    # bottom
            x, y = int(rng.integers(0, w)), h - 1 - inset
        elif edge == 2:    # left
            x, y = inset, int(rng.integers(0, h))
        else:              # right
            x, y = w - 1 - inset, int(rng.integers(0, h))
        cv2.circle(a, (x, y), r, float(rng.uniform(0.5, 1.0)), -1, cv2.LINE_AA)

    # Triangular rips anchored to a random edge.
    for _ in range(n_rips):
        tri = _edge_rip_triangle(rng, w, h, int(rng.integers(0, 4)))
        cv2.fillPoly(a, [np.array(tri, np.int32)], float(rng.uniform(0.6, 1.0)),
                     cv2.LINE_AA)

    # Paper texture within the whitening.
    a *= (0.6 + 0.4 * _smooth_noise(h, w, 4, rng))
    a = cv2.GaussianBlur(a, (0, 0), 1.0) * strength
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = 235
    rgba[..., 3] = np.clip(a * 255, 0, 255).astype(np.uint8)
    return rgba


def composite_overlays(base_bgr: np.ndarray, overlays: Sequence[np.ndarray]) -> np.ndarray:
    """Alpha-composite RGBA overlays onto a BGR base image (uint8)."""
    out = base_bgr.astype(np.float32)
    for ov in overlays:
        a = (ov[..., 3:4].astype(np.float32) / 255.0)
        out = out * (1.0 - a) + ov[..., :3].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)
