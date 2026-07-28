"""Deterministic, bpy-free post-processing for rendered BGR uint8 images.

All randomness needed by an effect is stored in ``rules.combinations.PostFxConfig``.
This module therefore never reads global random state and never changes label geometry.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from rules.combinations import PostFxConfig


def _require_bgr_u8(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("postfx expects a uint8 NumPy image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("postfx expects a BGR image with exactly three channels")
    return image


def _u8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def _white_balance(image: np.ndarray, temperature_shift_k: float) -> np.ndarray:
    """Approximate a subtle Kelvin shift; positive values make the image cooler."""
    strength = float(np.clip(temperature_shift_k / 3000.0, -0.30, 0.30))
    gains = np.array([1.0 + strength, 1.0, 1.0 - strength], dtype=np.float32)
    return np.clip(image * gains, 0.0, 1.0)


def _tint(image: np.ndarray, green_magenta_shift: float) -> np.ndarray:
    """Positive values shift toward green; negative values shift toward magenta."""
    shift = float(green_magenta_shift)
    gains = np.array([1.0 - shift * 0.5, 1.0 + shift, 1.0 - shift * 0.5], dtype=np.float32)
    return np.clip(image * gains, 0.0, 1.0)


def _haze(image: np.ndarray, strength: float, brightness: float) -> np.ndarray:
    haze_color = np.full((1, 1, 3), brightness, dtype=np.float32)
    return image * (1.0 - strength) + haze_color * strength


def _chromatic_aberration(image: np.ndarray, shift_px: float, angle_deg: float) -> np.ndarray:
    if shift_px <= 0.0:
        return image
    height, width = image.shape[:2]
    angle = np.deg2rad(angle_deg)
    dx, dy = shift_px * np.cos(angle), shift_px * np.sin(angle)
    forward = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    reverse = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
    blue = cv2.warpAffine(image[:, :, 0], forward, (width, height),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    red = cv2.warpAffine(image[:, :, 2], reverse, (width, height),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return np.dstack((blue, image[:, :, 1], red))


def _pixel_melt(image: np.ndarray, blur_sigma: float, scale: float) -> np.ndarray:
    out = image
    if blur_sigma > 0.0:
        out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
    height, width = out.shape[:2]
    small_w, small_h = max(1, round(width * scale)), max(1, round(height * scale))
    if (small_w, small_h) != (width, height):
        out = cv2.resize(out, (small_w, small_h), interpolation=cv2.INTER_AREA)
        out = cv2.resize(out, (width, height), interpolation=cv2.INTER_LINEAR)
    return out


def _motion_blur_kernel(direction_index: int, strength: float) -> np.ndarray:
    """Build a center-weighted 7x7 trail kernel in one of 16 directions."""
    if not isinstance(direction_index, (int, np.integer)) or not 0 <= direction_index < 16:
        raise ValueError("motion-blur direction_index must be in [0, 15]")
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("motion-blur strength must be in [0, 1]")
    angle = 2.0 * np.pi * direction_index / 16.0
    kernel = np.zeros((7, 7), dtype=np.float32)
    kernel[3, 3] = 1.0 - strength
    for distance in range(1, 4):
        x = int(np.rint(distance * np.cos(angle)))
        y = int(np.rint(distance * np.sin(angle)))
        kernel[3 + y, 3 + x] += strength / 3.0
    return kernel


def _motion_blur(image: np.ndarray, direction_index: int, strength: float) -> np.ndarray:
    kernel = _motion_blur_kernel(direction_index, strength)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT101)


def _sensor_noise(image: np.ndarray, luma_sigma: float, chroma_sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    luma = rng.normal(0.0, luma_sigma, size=(height, width, 1)).astype(np.float32)
    chroma = rng.normal(0.0, chroma_sigma, size=image.shape).astype(np.float32)
    return np.clip(image + luma + chroma, 0.0, 1.0)


def _jpeg_cycles(image: np.ndarray, quality: int, cycles: int) -> np.ndarray:
    out = _u8(image)
    for _ in range(cycles):
        ok, encoded = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("OpenCV could not encode post-effect JPEG")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("OpenCV could not decode post-effect JPEG")
        out = decoded
    return out.astype(np.float32) / 255.0


def apply_postfx(image: np.ndarray, settings: "PostFxConfig") -> np.ndarray:
    """Apply sampled post effects in a fixed camera-like order and return BGR uint8.

    ``settings`` is a fully sampled scene value. It contains no probabilities or
    ranges, so repeat application with the same input and settings is byte-identical.
    """
    source = _require_bgr_u8(image)
    out = source.astype(np.float32) / 255.0
    if settings.white_balance is not None:
        out = _white_balance(out, settings.white_balance.temperature_shift_k)
    if settings.tint is not None:
        out = _tint(out, settings.tint.green_magenta_shift)
    if settings.haze is not None:
        out = _haze(out, settings.haze.strength, settings.haze.brightness)
    if settings.contrast is not None:
        out = np.clip((out - 0.5) * (1.0 - settings.contrast.reduction) + 0.5, 0.0, 1.0)
    if settings.chromatic_aberration is not None:
        out = _chromatic_aberration(out, settings.chromatic_aberration.shift_px,
                                     settings.chromatic_aberration.angle_deg)
    if settings.motion_blur is not None:
        out = _motion_blur(out, settings.motion_blur.direction_index,
                           settings.motion_blur.strength)
    if settings.pixel_melt is not None:
        out = _pixel_melt(out, settings.pixel_melt.blur_sigma, settings.pixel_melt.scale)
    if settings.sensor_noise is not None:
        out = _sensor_noise(out, settings.sensor_noise.luma_sigma,
                            settings.sensor_noise.chroma_sigma, settings.sensor_noise.seed)
    if settings.compression is not None:
        out = _jpeg_cycles(out, settings.compression.jpeg_quality, settings.compression.cycles)
    return _u8(out)


def apply_postfx_file(input_path: str, output_path: str, settings: "PostFxConfig") -> str:
    """Read a render, apply image-space effects, and write its processed copy."""
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read render: {input_path}")
    if not cv2.imwrite(output_path, apply_postfx(image, settings)):
        raise RuntimeError(f"Could not write processed render: {output_path}")
    return output_path
