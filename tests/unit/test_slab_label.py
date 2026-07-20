"""Docker unit tests for texturegen/slab_label.py. Writes a preview to out/."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import config  # noqa: E402
from texturegen import slab_label as sl  # noqa: E402


def test_shape_and_determinism():
    a = sl.generate_slab_label(680, 200, seed=2)
    b = sl.generate_slab_label(680, 200, seed=2)
    c = sl.generate_slab_label(680, 200, seed=3)
    assert a.shape == (200, 680, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_has_header_and_body_contrast():
    m = sl.generate_slab_label(680, 200, seed=1)
    # Body region should be mostly light (off-white) with some dark marks.
    body = m[80:, :, :]
    assert body.mean() > 150, body.mean()
    assert (body.mean(axis=2) < 80).any()  # some dark text marks exist


def _write_previews():
    os.makedirs(config.OUTPUT.root, exist_ok=True)
    for s in range(3):
        sl.save_label(sl.generate_slab_label(680, 200, seed=s),
                      os.path.join(config.OUTPUT.root, f"unit_slab_label_{s}.png"))
    print(f"    (wrote {config.OUTPUT.root}/unit_slab_label_0..2.png for eyeball)")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    _write_previews()
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
