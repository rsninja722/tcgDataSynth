"""Docker unit tests for texturegen/holo.py. Writes previews to out/."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import config  # noqa: E402
from texturegen import holo  # noqa: E402

REGION = config.DEFAULT_PICTURE_REGION


def test_region_modes():
    w = h = 200
    entire = holo.region_mask(w, h, REGION, "entire")
    pic = holo.region_mask(w, h, REGION, "picture")
    rev = holo.region_mask(w, h, REGION, "reverse")
    assert (entire == 255).all()
    # picture + reverse partition the image
    assert np.array_equal(pic + rev, np.full((h, w), 255, np.uint8))
    # picture region is a nonempty box strictly inside the default region
    assert pic.max() == 255 and pic.min() == 0
    # top rows (above y0=0.098) are NOT in the picture region
    assert (pic[0, :] == 0).all()


def test_patterns_shapes_and_character():
    w = h = 128
    smooth = holo.holo_pattern(w, h, "none")
    lines = holo.holo_pattern(w, h, "horizontal_lines")
    cosmos = holo.holo_pattern(w, h, "cosmos", seed=1)
    for p in (smooth, lines, cosmos):
        assert p.shape == (h, w) and p.dtype == np.uint8
    assert smooth.std() < 1.0                      # uniform
    # horizontal lines vary along y, (almost) constant along x
    assert lines.std(axis=0).mean() > lines.std(axis=1).mean()
    assert cosmos.std() > 5.0                       # textured


def test_masked_pattern_zero_outside():
    w = h = 160
    out = holo.masked_pattern(w, h, REGION, "picture", "cosmos", seed=2)
    m = holo.region_mask(w, h, REGION, "picture")
    assert (out[m == 0] == 0).all()
    assert out[m == 255].max() > 0


def test_determinism():
    a = holo.holo_pattern(64, 64, "cosmos", seed=5)
    b = holo.holo_pattern(64, 64, "cosmos", seed=5)
    assert np.array_equal(a, b)


def test_all_patterns_and_normals():
    for pat in holo.HOLO_PATTERNS:
        g = holo.holo_pattern(96, 128, pat, seed=1)
        assert g.shape == (128, 96) and g.dtype == np.uint8
        nm = holo.pattern_normal(g)
        assert nm.shape == (128, 96, 3) and nm[:, :, 2].mean() > 150  # mostly +Z


def _write_previews():
    out = config.OUTPUT.root
    os.makedirs(out, exist_ok=True)
    w, h = 420, 588  # card aspect
    for mode in holo.HOLO_REGIONS:
        cv2.imwrite(os.path.join(out, f"unit_holo_mask_{mode}.png"),
                    holo.region_mask(w, h, REGION, mode))
    for pat in holo.HOLO_PATTERNS:
        g = holo.holo_pattern(w, h, pat, seed=0)
        cv2.imwrite(os.path.join(out, f"unit_holo_pattern_{pat}.png"), g)
        cv2.imwrite(os.path.join(out, f"unit_holo_normal_{pat}.png"),
                    holo.pattern_normal(g)[:, :, ::-1])
    cv2.imwrite(os.path.join(out, "unit_holo_reverse_cosmos.png"),
                holo.masked_pattern(w, h, REGION, "reverse", "cosmos", seed=0))
    print(f"    (wrote {out}/unit_holo_*.png for eyeball)")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    _write_previews()
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
