"""
Physical-texture emulation (bpy-FREE, Docker-testable). Spec §3.4.

Etched-foil effect: local contours of the card art set the local direction of the
texture lines, and lines extend parallel to neighbouring lines until the card is
filled. Implemented as: structure-tensor flow field (direction ALONG edges,
smoothed so it propagates into flat regions) -> Line Integral Convolution of noise
along that field (the parallel etched lines) -> grayscale height -> normal map.

Output: a normal map (feeds the holo/foil shader's Normal in Blender) plus the raw
line pattern for inspection. End-to-end Docker-testable from a card image.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def flow_field(gray: np.ndarray, sigma: float = 6.0,
               contrast_percentile: float = 80.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fx, fy) unit flow along image contours via the structure tensor.
    Only HIGH-CONTRAST edges (gradient magnitude above `contrast_percentile`) drive
    the direction; weak color changes are zeroed and instead inherit their
    orientation from nearby strong edges through the tensor smoothing (so lines
    stay parallel and follow real contours, not every faint gradient)."""
    g = cv2.GaussianBlur(gray.astype(np.float32) / 255.0, (0, 0), 1.0)  # denoise input
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, contrast_percentile)
    keep = (mag >= thr).astype(np.float32)           # strong edges only
    gx *= keep
    gy *= keep
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)   # major eigenvector (across edge)
    fx = -np.sin(theta)                              # rotate 90deg -> along edge
    fy = np.cos(theta)
    return fx.astype(np.float32), fy.astype(np.float32)


def lic(fx: np.ndarray, fy: np.ndarray, noise: np.ndarray,
        length: int = 12, step: float = 1.0) -> np.ndarray:
    """Line Integral Convolution: average `noise` along the flow streamlines,
    forward and backward, producing streaks aligned with the flow."""
    h, w = noise.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    accum = noise.copy().astype(np.float32)
    count = np.ones_like(accum)
    for sign in (1.0, -1.0):
        px, py = xx.copy(), yy.copy()
        for _ in range(length):
            vx = cv2.remap(fx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            vy = cv2.remap(fy, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            px = px + sign * vx * step
            py = py + sign * vy * step
            accum += cv2.remap(noise, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            count += 1.0
    out = accum / count
    return (out - out.min()) / (np.ptp(out) + 1e-6)


def height_to_normal(height: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """(h,w) float[0,1] height -> (h,w,3) uint8 RGB tangent-space normal map."""
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) * strength
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) * strength
    nx, ny, nz = -gx, -gy, np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    rgb = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    return (np.clip(rgb * 0.5 + 0.5, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def traced_lines(fx: np.ndarray, fy: np.ndarray, spacing: float = 3.0,
                 step: float = 0.6, seed: int = 0, max_len: int = 0) -> np.ndarray:
    """Evenly-spaced streamlines of the flow field (Jobard-Lefebvre style): clean
    1px anti-aliased lines separated by `spacing` px (default 3 => 1px line + 2px gap),
    NOT noisy. Each line propagates until it exits the card or meets another line (so
    the card fills with flowing lines and no gaps need vertical fill). Returns (h,w)
    float32 in [0,1]."""
    h, w = fx.shape
    out = np.zeros((h, w), np.float32)
    cell = max(1.0, spacing)
    # Large enough that a line can cross the whole card; it still stops at the border
    # or another line first. Only a backstop against closed-loop flows.
    if max_len <= 0:
        max_len = int((w + h) * 3.0 / step)
    gh, gw = int(np.ceil(h / cell)), int(np.ceil(w / cell))
    occ = np.zeros((gh, gw), np.uint8)   # separation grid (previous lines only)
    rng = np.random.default_rng(seed)

    def sample(field, x, y):
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        if x0 < 0 or y0 < 0 or x0 >= w - 1 or y0 >= h - 1:
            return None
        tx, ty = x - x0, y - y0
        return (field[y0, x0] * (1 - tx) * (1 - ty) + field[y0, x0 + 1] * tx * (1 - ty)
                + field[y0 + 1, x0] * (1 - tx) * ty + field[y0 + 1, x0 + 1] * tx * ty)

    def occ_at(x, y):
        gx, gy = int(x / cell), int(y / cell)
        return not (0 <= gx < gw and 0 <= gy < gh) or occ[gy, gx]

    def trace(sx, sy, direction):
        pts, x, y = [], sx, sy
        pvx = pvy = None
        for _ in range(max_len):
            vx, vy = sample(fx, x, y), sample(fy, x, y)
            if vx is None:
                break
            if pvx is not None and (vx * pvx + vy * pvy) < 0:   # line field: fix sign flips
                vx, vy = -vx, -vy
            pvx, pvy = vx, vy
            x += direction * vx * step
            y += direction * vy * step
            if x < 0 or y < 0 or x >= w or y >= h or occ_at(x, y):
                break
            pts.append((x, y))
        return pts

    cells = [(gx, gy) for gy in range(gh) for gx in range(gw)]
    rng.shuffle(cells)
    for (gx, gy) in cells:
        sx, sy = (gx + 0.5) * cell, (gy + 0.5) * cell
        if occ_at(sx, sy):
            continue
        line = list(reversed(trace(sx, sy, -1.0))) + [(sx, sy)] + trace(sx, sy, 1.0)
        if len(line) < 2:
            continue
        cv2.polylines(out, [np.round(line).astype(np.int32)], False, 1.0, 1, cv2.LINE_AA)
        for (x, y) in line:
            ggx, ggy = int(x / cell), int(y / cell)
            if 0 <= ggx < gw and 0 <= ggy < gh:
                occ[ggy, ggx] = 1
    return out


def generate_physical_texture(card_bgr: np.ndarray, seed: int = 0,
                              spacing: float = 3.0, normal_strength: float = 1.4
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """From a BGR card image, return (line_pattern_gray_uint8, normal_map_rgb_uint8).
    Clean evenly-spaced etched lines (1px AA, `spacing`px apart) that follow the art's
    high-contrast contours."""
    gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    fx, fy = flow_field(gray)
    pattern = traced_lines(fx, fy, spacing=spacing, seed=seed)
    normal = height_to_normal(pattern, strength=normal_strength)
    return (pattern * 255.0 + 0.5).astype(np.uint8), normal


def save_png(arr: np.ndarray, path: str) -> str:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if arr.ndim == 3:
        cv2.imwrite(path, arr[:, :, ::-1])  # RGB -> BGR for cv2
    else:
        cv2.imwrite(path, arr)
    return path
