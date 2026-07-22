"""
Docker unit tests for labeltools/occlusion.py (occlusion second pass) and the custom
PolyLabel format. Pure-python; uses shapely (occlusion carving is skipped without it,
so the carving tests are guarded).

Run:  python3 tests/unit/test_occlusion.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from labeltools import occlusion as occ  # noqa: E402
from labeltools import yolo_pose as yp  # noqa: E402

# A card fully inside the frustum: TL,TR,BR,BL in Blender-NDC (bottom-left origin), z>0.
FULL = [(0.2, 0.8, 0.5), (0.8, 0.8, 0.5), (0.8, 0.2, 0.5), (0.2, 0.2, 0.5)]


def _flags(tagged):
    return [f for (_x, _y, f) in tagged]


def test_full_card_four_corner_flags_in_cw_order():
    pts, cls, reason = occ.compute_bound("c", FULL, front_visible=True)
    assert cls == 0 and reason == "labeled"
    # 4 corners, flags TL=1,TR=2,BR=4,BL=3 in clockwise order starting at TL.
    assert _flags(pts) == [1, 2, 4, 3], _flags(pts)
    # starts at TL (yolo top-left ~ (0.2,0.2))
    assert abs(pts[0][0] - 0.2) < 1e-6 and abs(pts[0][1] - 0.2) < 1e-6


def test_back_facing_and_fully_out():
    assert occ.compute_bound("c", FULL, front_visible=False)[2] == "back-facing"
    out = [(1.2, 1.2, 0.5), (1.8, 1.2, 0.5), (1.8, 0.6, 0.5), (1.2, 0.6, 0.5)]
    pts, cls, reason = occ.compute_bound("c", out, front_visible=True)
    assert pts is None and reason == "fully-out-of-frustum"


def test_partial_has_created_points():
    # Shove right so TR/BR exit the frustum -> class 1 with flag-5 crossings.
    part = [(0.6, 0.8, 0.5), (1.4, 0.8, 0.5), (1.4, 0.2, 0.5), (0.6, 0.2, 0.5)]
    pts, cls, reason = occ.compute_bound("c", part, front_visible=True)
    assert cls == 1 and reason == "labeled-partial"
    assert 5 in _flags(pts) and 1 in _flags(pts)          # created + a surviving corner
    on_edge = [p for p in pts if min(p[0], 1 - p[0]) <= 1e-6 or min(p[1], 1 - p[1]) <= 1e-6]
    assert len(on_edge) >= 2


def test_occluder_below_threshold_no_change():
    if not occ._HAVE_SHAPELY:
        return
    # Small occluder covering ~11% (<25%) of the card -> unchanged 4 corners.
    small = ([(0.2, 0.6, 0.5), (0.4, 0.6, 0.5), (0.4, 0.8, 0.5), (0.2, 0.8, 0.5)][:4])
    small_xy = [(x, y) for (x, y, _z) in [(0.2, 0.6, 0), (0.4, 0.6, 0), (0.4, 0.8, 0), (0.2, 0.8, 0)]]
    pts, cls, _ = occ.compute_bound("c", FULL, True, occluders=[(small_xy, 0.3)], card_depth=0.5)
    assert _flags(pts) == [1, 2, 4, 3]


def test_occluder_carves_and_removes_corners():
    if not occ._HAVE_SHAPELY:
        return
    # Occluder covers the RIGHT ~half (>25%), nearer than the card -> TR/BR removed,
    # replaced by two created (flag-5) points on the cut line; TL/BL survive.
    occ_xy = [(0.5, 0.1), (0.95, 0.1), (0.95, 0.9), (0.5, 0.9)]
    pts, cls, reason = occ.compute_bound("c", FULL, True, occluders=[(occ_xy, 0.3)], card_depth=0.5)
    flags = _flags(pts)
    assert 2 not in flags and 4 not in flags, flags        # TR, BR carved away
    assert 1 in flags and 3 in flags, flags                # TL, BL survive
    assert flags.count(5) >= 2                              # >=2 new cut vertices
    # No bound point lies to the right of the cut (x>0.5) beyond tolerance.
    assert max(p[0] for p in pts) <= 0.5 + 1e-6


def test_occluder_behind_card_ignored():
    if not occ._HAVE_SHAPELY:
        return
    occ_xy = [(0.5, 0.1), (0.95, 0.1), (0.95, 0.9), (0.5, 0.9)]
    # depth 0.7 > card 0.5 => occluder is BEHIND the card => ignored.
    pts, cls, _ = occ.compute_bound("c", FULL, True, occluders=[(occ_xy, 0.7)], card_depth=0.5)
    assert _flags(pts) == [1, 2, 4, 3]


def test_multiple_occluders_carve_in_turn():
    if not occ._HAVE_SHAPELY:
        return
    right = [(0.55, 0.1), (0.95, 0.1), (0.95, 0.9), (0.55, 0.9)]
    top = [(0.1, 0.55), (0.9, 0.55), (0.9, 0.95), (0.1, 0.95)]
    pts, cls, _ = occ.compute_bound("c", FULL, True,
                                    occluders=[(right, 0.3), (top, 0.2)], card_depth=0.5)
    # Both the right and top strips removed -> everything kept is x<=0.55 and (ndc) y<=0.55.
    assert max(p[0] for p in pts) <= 0.55 + 1e-6           # x in yolo == x in ndc
    assert len(pts) >= 4


def test_interior_occluder_hole_is_bridged():
    if not occ._HAVE_SHAPELY:
        return
    # Occluder fully INSIDE the card (>25%) -> difference has a hole -> bridged into a
    # single perimeter ring (keyhole). Must stay a valid single ring, no crash.
    inner = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
    pts, cls, reason = occ.compute_bound("c", FULL, True, occluders=[(inner, 0.3)], card_depth=0.5)
    assert reason in ("labeled", "labeled-partial") and pts is not None
    assert len(pts) >= 8                                   # 4 outer + hole ring + slit
    # All 4 original corners survive (only the interior was carved).
    assert set(_flags(pts)) >= {1, 2, 3, 4}


def test_polylabel_line_roundtrip():
    pts, cls, _ = occ.compute_bound("charizard", FULL, True)
    lab = yp.PolyLabel("charizard", tuple(pts), holo_tag="reverse", class_id=cls)
    line = lab.to_line(include_id=True)
    segs = line.split("|")
    assert segs[1].strip() == "charizard" and segs[2].strip() == "reverse"
    core = segs[0].split()
    assert core[0] == "0"                                   # class
    assert len(core) == 5 + 4 * 3                           # 2-corner bbox (4) + class + 4 pts*3
    parsed = yp.parse_poly_line(line)
    assert parsed.class_id == 0 and len(parsed.points) == 4
    assert [p[2] for p in parsed.points] == [1, 2, 4, 3]
    (bx0, by0), (bx1, by1) = parsed.bbox2
    assert bx0 < bx1 and by0 < by1
    assert "|" not in lab.to_line(include_id=False)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed. (shapely={'yes' if occ._HAVE_SHAPELY else 'NO'})")


if __name__ == "__main__":
    _run_all()
