"""
Docker unit tests for texturegen/plastic_warp.py. Also writes a sample normal map
and a shaded preview to out/ for eyeball review.

Run:  python3 tests/unit/test_plastic_warp.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import cv2  # noqa: E402

import config  # noqa: E402
from texturegen import plastic_warp as pw  # noqa: E402


def test_shape_dtype_and_flatness():
    # Use the real asset resolution (1024): large base_cell needs a big enough image
    # to average several deflections out to ~flat; at 256 it's one big tilt.
    nm = pw.generate_plastic_warp_normal(1024, 1024, seed=1)
    assert nm.shape == (1024, 1024, 3) and nm.dtype == np.uint8
    # Mostly-flat plastic => B (nz) high, R/G (nx,ny) centered near 128.
    assert nm[:, :, 2].mean() > 190, nm[:, :, 2].mean()
    assert 100 < nm[:, :, 0].mean() < 156, nm[:, :, 0].mean()
    assert 100 < nm[:, :, 1].mean() < 156, nm[:, :, 1].mean()


def test_unit_length_normals():
    nm = pw.generate_plastic_warp_normal(128, 128, seed=2).astype(np.float32)
    vec = nm / 255.0 * 2.0 - 1.0
    lens = np.sqrt((vec ** 2).sum(axis=-1))
    assert abs(lens.mean() - 1.0) < 0.02, lens.mean()  # ~unit length


def test_determinism():
    a = pw.generate_plastic_warp_normal(96, 96, seed=5)
    b = pw.generate_plastic_warp_normal(96, 96, seed=5)
    c = pw.generate_plastic_warp_normal(96, 96, seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_strength_increases_tilt():
    weak = pw.generate_plastic_warp_normal(128, 128, seed=3, strength=0.5)
    strong = pw.generate_plastic_warp_normal(128, 128, seed=3, strength=3.0)
    # Stronger warp => nx/ny deviate further from the flat 128.
    assert np.abs(strong[:, :, 0].astype(int) - 128).mean() > \
           np.abs(weak[:, :, 0].astype(int) - 128).mean()


def _write_previews():
    out = config.OUTPUT.root
    os.makedirs(out, exist_ok=True)
    nm = pw.generate_plastic_warp_normal(512, 512, seed=7, strength=1.6)
    pw.save_normal_map(nm, os.path.join(out, "unit_plastic_warp_normal.png"))
    # Cheap shaded preview: dot(normal, light) to visualize the undulation.
    vec = nm.astype(np.float32) / 255.0 * 2.0 - 1.0
    light = np.array([0.3, 0.3, 0.9], np.float32)
    light /= np.linalg.norm(light)
    shade = np.clip((vec * light).sum(-1), 0, 1)
    cv2.imwrite(os.path.join(out, "unit_plastic_warp_shaded.png"),
                (shade * 255).astype(np.uint8))
    print(f"    (wrote {out}/unit_plastic_warp_normal.png + _shaded.png for eyeball)")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    _write_previews()
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
