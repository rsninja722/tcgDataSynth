"""Deterministic two-scale simplex masks for lighting shadow planes."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


GRID_FACES = 50
COARSE_FREQUENCY = 2.0
FINE_FREQUENCY = 12.0
COARSE_WEIGHT = 0.65
FINE_WEIGHT = 0.35
REMOVE_THRESHOLD = 0.50

_F2 = (math.sqrt(3.0) - 1.0) / 2.0
_G2 = (3.0 - math.sqrt(3.0)) / 6.0
_MASK_64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_GRADIENTS = np.asarray(
    (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (math.sqrt(0.5), math.sqrt(0.5)),
        (-math.sqrt(0.5), math.sqrt(0.5)),
        (math.sqrt(0.5), -math.sqrt(0.5)),
        (-math.sqrt(0.5), -math.sqrt(0.5)),
    ),
    dtype=np.float64,
)


def _lattice_hash(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Stable integer hash for simplex gradient selection."""
    ux = np.asarray(x, dtype=np.int64).astype(np.uint64)
    uy = np.asarray(y, dtype=np.int64).astype(np.uint64)
    value = (ux * np.uint64(0x9E3779B185EBCA87)) & _MASK_64
    value ^= (uy * np.uint64(0xC2B2AE3D27D4EB4F)) & _MASK_64
    value ^= np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)
    value ^= value >> np.uint64(30)
    value = (value * np.uint64(0xBF58476D1CE4E5B9)) & _MASK_64
    value ^= value >> np.uint64(27)
    value = (value * np.uint64(0x94D049BB133111EB)) & _MASK_64
    value ^= value >> np.uint64(31)
    return value


def _corner_contribution(
    x: np.ndarray,
    y: np.ndarray,
    lattice_x: np.ndarray,
    lattice_y: np.ndarray,
    seed: int,
) -> np.ndarray:
    gradient_index = (_lattice_hash(lattice_x, lattice_y, seed) & np.uint64(7)).astype(
        np.intp
    )
    gradients = _GRADIENTS[gradient_index]
    attenuation = 0.5 - x * x - y * y
    contribution = np.zeros_like(x, dtype=np.float64)
    active = attenuation > 0.0
    attenuation_sq = attenuation[active] * attenuation[active]
    dot = gradients[..., 0][active] * x[active] + gradients[..., 1][active] * y[active]
    contribution[active] = attenuation_sq * attenuation_sq * dot
    return contribution


def simplex_noise_2d(
    u: np.ndarray,
    v: np.ndarray,
    frequency: float,
    seed: int,
) -> np.ndarray:
    """Return seeded 2D simplex noise, approximately normalized to ``[-1, 1]``."""
    x = np.asarray(u, dtype=np.float64) * float(frequency)
    y = np.asarray(v, dtype=np.float64) * float(frequency)
    if x.shape != y.shape:
        raise ValueError("u and v must have matching shapes")

    skew = (x + y) * _F2
    lattice_x = np.floor(x + skew).astype(np.int64)
    lattice_y = np.floor(y + skew).astype(np.int64)
    unskew = (lattice_x + lattice_y) * _G2
    x0 = x - (lattice_x - unskew)
    y0 = y - (lattice_y - unskew)

    first_x = (x0 > y0).astype(np.int64)
    first_y = 1 - first_x
    x1 = x0 - first_x + _G2
    y1 = y0 - first_y + _G2
    x2 = x0 - 1.0 + 2.0 * _G2
    y2 = y0 - 1.0 + 2.0 * _G2

    noise = _corner_contribution(x0, y0, lattice_x, lattice_y, seed)
    noise += _corner_contribution(
        x1, y1, lattice_x + first_x, lattice_y + first_y, seed
    )
    noise += _corner_contribution(x2, y2, lattice_x + 1, lattice_y + 1, seed)
    return np.clip(noise * 70.0, -1.0, 1.0)


def face_sample_coordinates(grid_faces: int = GRID_FACES) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalized face-center coordinates with a top-left origin."""
    if int(grid_faces) < 1:
        raise ValueError("grid_faces must be positive")
    centers = (np.arange(int(grid_faces), dtype=np.float64) + 0.5) / int(grid_faces)
    return np.meshgrid(centers, centers, indexing="xy")


def shadow_brightness(seed: int, grid_faces: int = GRID_FACES) -> np.ndarray:
    """Combine broad and fine simplex layers into normalized mask brightness."""
    u, v = face_sample_coordinates(grid_faces)
    coarse = simplex_noise_2d(u, v, COARSE_FREQUENCY, int(seed)) * 0.5 + 0.5
    fine_seed = int(seed) ^ 0x6A09E667F3BCC909
    fine = simplex_noise_2d(u, v, FINE_FREQUENCY, fine_seed) * 0.5 + 0.5
    return np.clip(COARSE_WEIGHT * coarse + FINE_WEIGHT * fine, 0.0, 1.0)


def retained_faces(seed: int, grid_faces: int = GRID_FACES) -> np.ndarray:
    """True where an opaque face remains; samples above 50% are holes."""
    return shadow_brightness(seed, grid_faces) <= REMOVE_THRESHOLD


def unit_grid_geometry(retained: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """Build a unit-square grid and +Z-wound quads for retained mask cells."""
    retained = np.asarray(retained, dtype=bool)
    if retained.ndim != 2 or retained.shape[0] != retained.shape[1]:
        raise ValueError("retained must be a square 2D array")
    grid_faces = retained.shape[0]
    axis = np.linspace(0.0, 1.0, grid_faces + 1, dtype=np.float64)
    u, v = np.meshgrid(axis, axis, indexing="xy")
    vertices = np.column_stack((u.ravel() - 0.5, 0.5 - v.ravel(), np.zeros(u.size)))

    stride = grid_faces + 1
    faces: List[Tuple[int, int, int, int]] = []
    for row, col in np.argwhere(retained):
        top_left = int(row) * stride + int(col)
        top_right = top_left + 1
        bottom_left = top_left + stride
        bottom_right = bottom_left + 1
        faces.append((top_left, bottom_left, bottom_right, top_right))
    return vertices, faces
