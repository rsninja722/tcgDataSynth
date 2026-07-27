"""Docker unit tests for deterministic, bpy-free Phase 6 post effects."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import cv2
import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from postfx.effects import apply_postfx, apply_postfx_file  # noqa: E402
from rules import combinations as C  # noqa: E402


def _image() -> np.ndarray:
    y, x = np.indices((96, 128), dtype=np.int16)
    return np.dstack(((2 * x + y) % 256, (3 * y + x) % 256,
                      (5 * x + 2 * y) % 256)).astype(np.uint8)


def _all_effects() -> C.PostFxConfig:
    return C.PostFxConfig(
        sensor_noise=C.SensorNoiseConfig(0.012, 0.006, 123456),
        compression=C.CompressionConfig(25, 2),
        pixel_melt=C.PixelMeltConfig(1.1, 0.60),
        white_balance=C.WhiteBalanceConfig(400.0),
        tint=C.TintConfig(0.04),
        chromatic_aberration=C.ChromaticAberrationConfig(1.5, 35.0),
        contrast=C.ContrastConfig(0.5),
        haze=C.HazeConfig(0.18, 0.9),
    )


def test_noop_preserves_pixels_shape_and_type():
    image = _image()
    result = apply_postfx(image, C.PostFxConfig())
    assert result.dtype == np.uint8 and result.shape == image.shape
    assert np.array_equal(result, image)


def test_all_effects_are_repeatable_and_visible():
    image = _image()
    settings = _all_effects()
    first = apply_postfx(image, settings)
    second = apply_postfx(image, settings)
    assert np.array_equal(first, second)
    assert first.shape == image.shape and first.dtype == np.uint8
    assert not np.array_equal(first, image)


def test_each_effect_changes_a_colored_image():
    image = _image()
    all_effects = _all_effects()
    for name in C.POST_EFFECTS:
        settings = C.PostFxConfig(**{name: getattr(all_effects, name)})
        result = apply_postfx(image, settings)
        assert not np.array_equal(result, image), name


def test_postfx_sampling_honors_config_ranges_and_effect_toggles():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.json")
        tuning = {
            "sensor_noise": {"probability": 1.0, "luma_sigma_range": [0.01, 0.01],
                             "chroma_sigma_range": [0.005, 0.005]},
            "compression": {"probability": 1.0, "jpeg_quality_range": [30, 30],
                            "cycles_range": [2, 2]},
            "pixel_melt": {"probability": 1.0, "blur_sigma_range": [0.8, 0.8],
                           "scale_range": [0.7, 0.7]},
            "white_balance": {"probability": 1.0, "temperature_shift_k_range": [100, 100]},
            "tint": {"probability": 1.0, "green_magenta_shift_range": [0.02, 0.02]},
            "chromatic_aberration": {"probability": 1.0, "shift_px_range": [1.0, 1.0]},
            "contrast": {"probability": 1.0, "reduction_range": [0.25, 0.25]},
            "haze": {"probability": 1.0, "strength_range": [0.1, 0.1],
                     "brightness_range": [0.8, 0.8]},
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"postfx": tuning}, handle)
        cfg = C.sample_scene_config({"layouts": ["table"], "post_effects": list(C.POST_EFFECTS)},
                                    44, config_path=path)
        assert all(getattr(cfg.postfx, name) is not None for name in C.POST_EFFECTS)
        assert cfg.postfx.sensor_noise.luma_sigma == 0.01
        assert cfg.postfx.compression.jpeg_quality == 30
        assert cfg.postfx.pixel_melt.scale == 0.7
        assert cfg.postfx.white_balance.temperature_shift_k == 100.0
        assert cfg.postfx.tint.green_magenta_shift == 0.02
        assert cfg.postfx.chromatic_aberration.shift_px == 1.0
        assert cfg.postfx.contrast.reduction == 0.25
        assert cfg.postfx.haze.brightness == 0.8

        disabled = C.sample_scene_config({"layouts": ["table"], "post_effects": []},
                                         44, config_path=path)
        assert all(getattr(disabled.postfx, name) is None for name in C.POST_EFFECTS)


def test_postfx_file_round_trip():
    with tempfile.TemporaryDirectory() as d:
        source = os.path.join(d, "source.png")
        output = os.path.join(d, "output.png")
        assert cv2.imwrite(source, _image())
        assert apply_postfx_file(source, output, _all_effects()) == output
        written = cv2.imread(output, cv2.IMREAD_COLOR)
        assert written is not None and written.shape == _image().shape


def _run_all():
    fns = [value for key, value in sorted(globals().items())
           if key.startswith("test_") and callable(value)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
