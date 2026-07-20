"""Docker unit tests for texturegen/surface_wear.py. Writes a preview to out/."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import config  # noqa: E402
from texturegen import surface_wear as sw  # noqa: E402


def test_shape_dtype_mostly_clean():
    m = sw.generate_scratch_dust(512, 512, seed=1)
    assert m.shape == (512, 512) and m.dtype == np.uint8
    # Mostly clean: the median pixel is near-black, but some wear exists.
    assert np.median(m) < 20, np.median(m)
    assert m.max() > 60, m.max()
    assert (m > 20).mean() < 0.5, (m > 20).mean()  # wear covers < half the surface


def test_determinism():
    a = sw.generate_scratch_dust(256, 256, seed=3)
    b = sw.generate_scratch_dust(256, 256, seed=3)
    c = sw.generate_scratch_dust(256, 256, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def _write_preview():
    os.makedirs(config.OUTPUT.root, exist_ok=True)
    m = sw.generate_scratch_dust(1024, 1024, seed=0)
    sw.save_gray(m, os.path.join(config.OUTPUT.root, "unit_surface_wear.png"))
    print(f"    (wrote {config.OUTPUT.root}/unit_surface_wear.png for eyeball)")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    _write_preview()
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
