"""Docker unit tests for texturegen/cardprep.py (uses a synthetic card image)."""
from __future__ import annotations

import os
import sys
import tempfile

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from texturegen import cardprep  # noqa: E402


def _write_fake_card(path, w=360, h=504):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = np.linspace(40, 220, w).astype(np.uint8)
    img[:, :, 1] = np.linspace(200, 60, h).reshape(-1, 1).astype(np.uint8)
    img[:, :, 2] = 120
    cv2.rectangle(img, (30, 40), (w - 30, h // 2), (240, 240, 240), -1)
    cv2.imwrite(path, img)


def test_physical_normal_cached_and_normal_like():
    with tempfile.TemporaryDirectory() as d:
        card = os.path.join(d, "abc123.png")
        _write_fake_card(card)
        cache = os.path.join(d, "cache")
        p = cardprep.physical_normal_path(card, cache)
        assert p and os.path.isfile(p)
        nm = cv2.imread(p)  # BGR
        assert nm.shape[2] == 3 and nm[:, :, 0].mean() > 150   # B(=+Z after BGR read) high
        # cached: second call returns the same path without regenerating
        assert cardprep.physical_normal_path(card, cache) == p


def test_damage_no_flags_returns_original():
    with tempfile.TemporaryDirectory() as d:
        card = os.path.join(d, "c.png")
        _write_fake_card(card)
        assert cardprep.damaged_card_path(card, os.path.join(d, "cache"), 0) == card


def test_damage_composites_and_varies_by_seed():
    with tempfile.TemporaryDirectory() as d:
        card = os.path.join(d, "c.png")
        _write_fake_card(card)
        cache = os.path.join(d, "cache")
        p1 = cardprep.damaged_card_path(card, cache, seed=1, dirt=True, scratches=True, surface=True)
        assert p1 != card and os.path.isfile(p1)
        base = cv2.imread(card)
        dmgd = cv2.imread(p1)
        assert not np.array_equal(base, dmgd)     # damage changed the image
        # different instance seed -> different wear
        p2 = cardprep.damaged_card_path(card, cache, seed=2, dirt=True, scratches=True, surface=True)
        assert not np.array_equal(cv2.imread(p1), cv2.imread(p2))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
