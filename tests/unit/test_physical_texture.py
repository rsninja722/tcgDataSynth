"""Docker unit tests for texturegen/physical_texture.py. Writes previews to out/."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import config  # noqa: E402
from texturegen import physical_texture as pt  # noqa: E402


def _fake_card(w=360, h=504):
    """Synthetic art with clear curved contours so the flow field has structure."""
    img = np.full((h, w, 3), 90, np.uint8)
    cv2.circle(img, (w // 2, h // 3), min(w, h) // 4, (220, 180, 60), -1)
    cv2.ellipse(img, (w // 2, int(h * 0.7)), (w // 3, h // 6), 20, 0, 360, (60, 120, 220), -1)
    cv2.line(img, (0, h // 2), (w, int(h * 0.55)), (240, 240, 240), 6)
    return img


def test_flow_field_unit_ish():
    g = cv2.cvtColor(_fake_card(120, 160), cv2.COLOR_BGR2GRAY)
    fx, fy = pt.flow_field(g)
    mag = np.sqrt(fx ** 2 + fy ** 2)
    assert fx.shape == g.shape
    assert abs(mag.mean() - 1.0) < 0.05   # ~unit vectors


def test_lic_and_normal_shapes():
    card = _fake_card(120, 160)
    pattern, normal = pt.generate_physical_texture(card, seed=1, line_length=8)
    assert pattern.shape == (160, 120) and pattern.dtype == np.uint8
    assert normal.shape == (160, 120, 3) and normal.dtype == np.uint8
    # normal map is mostly +Z (B channel high)
    assert normal[:, :, 2].mean() > 170
    # LIC of noise along a coherent field has more structure than the raw noise mean
    assert pattern.std() > 10


def test_determinism():
    card = _fake_card(100, 100)
    a, an = pt.generate_physical_texture(card, seed=3, line_length=6)
    b, bn = pt.generate_physical_texture(card, seed=3, line_length=6)
    assert np.array_equal(a, b) and np.array_equal(an, bn)


def _write_previews():
    out = config.OUTPUT.root
    os.makedirs(out, exist_ok=True)
    card = _fake_card()
    cv2.imwrite(os.path.join(out, "unit_physical_card.png"), card)
    pattern, normal = pt.generate_physical_texture(card, seed=0)
    pt.save_png(pattern, os.path.join(out, "unit_physical_lines.png"))
    pt.save_png(normal, os.path.join(out, "unit_physical_normal.png"))
    print(f"    (wrote {out}/unit_physical_*.png for eyeball)")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    _write_previews()
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
