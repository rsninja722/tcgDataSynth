"""Docker unit tests for texturegen/damage.py. Writes previews to out/."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import config  # noqa: E402
from texturegen import damage  # noqa: E402


def _fake_card(w=420, h=588):
    """A synthetic colorful card face for compositing tests."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = np.linspace(40, 220, w).astype(np.uint8)
    img[:, :, 1] = np.linspace(200, 60, h).reshape(-1, 1).astype(np.uint8)
    img[:, :, 2] = 120
    cv2.rectangle(img, (int(w * 0.08), int(h * 0.10)), (int(w * 0.92), int(h * 0.47)),
                  (240, 240, 240), -1)
    return img


def test_overlays_shape_and_alpha():
    for fn in (damage.dirt, damage.scratches, damage.surface_damage):
        ov = fn(200, 280, seed=1)
        assert ov.shape == (280, 200, 4) and ov.dtype == np.uint8
        assert ov[..., 3].max() > 0  # some effect present


def test_surface_damage_edge_biased():
    w = h = 240
    ov = damage.surface_damage(w, h, seed=2, n_blobs=200)
    a = ov[..., 3].astype(np.float32)
    m = 40  # border width
    border = a.copy()
    border[m:-m, m:-m] = 0
    center = np.zeros_like(a)
    center[m:-m, m:-m] = a[m:-m, m:-m]
    # More whitening mass near the border than the center (edge/corner bias).
    assert border.sum() > center.sum(), (border.sum(), center.sum())


def test_composite_changes_image():
    base = _fake_card()
    h, w = base.shape[:2]
    out = damage.composite_overlays(base, [
        damage.dirt(w, h, 1), damage.scratches(w, h, 2), damage.surface_damage(w, h, 3),
    ])
    assert out.shape == base.shape
    assert not np.array_equal(out, base)


def test_determinism():
    a = damage.surface_damage(100, 100, seed=7)
    b = damage.surface_damage(100, 100, seed=7)
    assert np.array_equal(a, b)


def _write_previews():
    out = config.OUTPUT.root
    os.makedirs(out, exist_ok=True)
    base = _fake_card()
    h, w = base.shape[:2]
    cv2.imwrite(os.path.join(out, "unit_damage_base.png"), base)
    cv2.imwrite(os.path.join(out, "unit_damage_all.png"),
                damage.composite_overlays(base, [
                    damage.dirt(w, h, 1), damage.scratches(w, h, 2),
                    damage.surface_damage(w, h, 3)]))
    cv2.imwrite(os.path.join(out, "unit_damage_surface_alpha.png"),
                damage.surface_damage(w, h, 3)[..., 3])
    print(f"    (wrote {out}/unit_damage_*.png for eyeball)")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    _write_previews()
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
