"""
Docker unit tests for labeltools/frustum.py — the bpy-free label DECISION logic.

Validates (without Blender): Y-flip to top-left origin, keypoint semantic ordering
(TL/TR/BR/BL land in the right image quadrants), frustum containment, behind-camera
rejection, and the front-face-visible rule. A tiny synthetic pinhole stands in for
world_to_camera_view to produce plausible NDC.

Run:  python3 tests/unit/test_frustum.py
"""
from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from labeltools import frustum  # noqa: E402

W, H = config.CARD_W_M, config.CARD_H_M

# Ideal corners in KEYPOINT_ORDER TL,TR,BR,BL (card local, front face z=0).
LOCAL = [(-W / 2, +H / 2), (+W / 2, +H / 2), (+W / 2, -H / 2), (-W / 2, -H / 2)]


def pinhole_ndc(X, Y, cam_z, focal_mm=35.0, sensor=36.0):
    """Blender-convention NDC for a head-on camera at (0,0,cam_z) looking -Z.
    (0.5,0.5)=center, +y=up, z=depth (positive in front). Mirrors world_to_camera_view."""
    depth = cam_z  # card at Z=0
    fov = 2.0 * math.atan((sensor / 2.0) / focal_mm)
    k = 1.0 / (2.0 * math.tan(fov / 2.0))
    ndc_x = 0.5 + (X / depth) * k
    ndc_y = 0.5 + (Y / depth) * k
    return (ndc_x, ndc_y, depth)


def card_ndc(offset_x=0.0, cam_z=0.35, focal=35.0):
    return [pinhole_ndc(lx + offset_x, ly, cam_z, focal) for (lx, ly) in LOCAL]


def test_headon_labeled_with_correct_quadrants():
    ndc = card_ndc()
    label, reason = frustum.classify("charizard", ndc, front_visible=True)
    assert reason == "labeled" and label is not None
    tl, tr, br, bl = label.corners  # yolo top-left origin
    # TL -> top-left quadrant, TR -> top-right, BR -> bottom-right, BL -> bottom-left
    assert tl[0] < 0.5 and tl[1] < 0.5, tl
    assert tr[0] > 0.5 and tr[1] < 0.5, tr
    assert br[0] > 0.5 and br[1] > 0.5, br
    assert bl[0] < 0.5 and bl[1] > 0.5, bl
    assert label.card_id == "charizard"


def test_yflip_is_applied():
    # A corner high in Blender NDC (y=0.8, near top) must map to small YOLO y (top).
    ndc = [(0.5, 0.8, 0.3)] * 4
    label, _ = frustum.classify("x", ndc, front_visible=True)
    assert abs(label.corners[0][1] - 0.2) < 1e-9


def test_back_facing_not_labeled():
    ndc = card_ndc()  # perfectly in frame...
    label, reason = frustum.classify("x", ndc, front_visible=False)  # ...but back shows
    assert label is None and reason == "back-facing"


def test_corner_out_of_frame_not_labeled():
    ndc = card_ndc(offset_x=0.2)  # shove right until a corner exits x>1
    xs = [c[0] for c in ndc]
    assert max(xs) > 1.0, f"test setup: expected a corner past the right edge, got {xs}"
    label, reason = frustum.classify("x", ndc, front_visible=True)
    assert label is None and reason == "corner-out-of-frustum"


def test_behind_camera_not_labeled():
    ndc = [(0.5, 0.5, -0.1)] * 4  # z<0 => behind camera
    label, reason = frustum.classify("x", ndc, front_visible=True)
    assert label is None and reason == "corner-out-of-frustum"


def test_corner_in_frustum_bounds():
    assert frustum.corner_in_frustum(0.0, 0.0, 0.1)
    assert frustum.corner_in_frustum(1.0, 1.0, 0.1)
    assert not frustum.corner_in_frustum(-1e-6, 0.5, 0.1)
    assert not frustum.corner_in_frustum(0.5, 1.0001, 0.1)
    assert not frustum.corner_in_frustum(0.5, 0.5, 0.0)  # exactly at camera plane


def test_is_front_visible():
    # Front normal +Z, camera in front (+Z) => visible.
    assert frustum.is_front_visible((0, 0, 1), cam_pos=(0, 0, 0.3), card_pos=(0, 0, 0))
    # Card rotated 180 (normal -Z), camera still in front => back shown, not visible.
    assert not frustum.is_front_visible((0, 0, -1), cam_pos=(0, 0, 0.3), card_pos=(0, 0, 0))
    # Grazing/edge-on => dot 0 => treated as not visible.
    assert not frustum.is_front_visible((1, 0, 0), cam_pos=(0, 0, 0.3), card_pos=(0, 0, 0))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
