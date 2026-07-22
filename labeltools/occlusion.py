"""
Occlusion-aware label bound computation (bpy-FREE, Docker-testable). SECOND PASS.

First pass (labeltools.frustum) gives each card the outline of its VISIBLE (in-frustum)
region. This module runs AFTER that: given a card's 4 projected corners and a list of
OCCLUDER RECTANGLES (other cards' physical extents projected to 2D, each with a camera
depth), it carves the parts of the card's bound that are hidden by a nearer rectangle
that covers > `area_frac` (default 25%) of the card's CURRENT bound. The result may be
CONCAVE and a bound may be carved by several occluders in turn.

Occluder rectangles (built on the Blender side, see blender.labeling):
  - bare card   -> the card rectangle
  - sleeved     -> the sleeve's outer rectangle
  - toploader   -> TWO rectangles (front + back plastic layer)
  - slab        -> TWO rectangles (front + back face)

Output vertices are tagged with a flag (see FLAG_*):
  1=TL corner, 2=TR corner, 3=BL corner, 4=BR corner, 5=a NON-corner bound point
  (a frustum-edge crossing, a covered frame corner, or an occlusion-carved vertex).
Corners removed by carving — or never added because they were outside the frustum —
simply do not appear. Points are emitted in clockwise perimeter order starting at the
lowest-ranked present corner (TL, then TR, BR, BL), with flag-5 points slotted in where
they fall on the perimeter.

Geometry uses shapely when available (robust concave/holed boolean ops). Without shapely
the frustum clip still runs (via labeltools.frustum) but occlusion carving is skipped —
so gen-time needs shapely in Blender's Python (like cv2), else labels omit occlusion.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from labeltools.frustum import corner_in_frustum, ndc_to_yolo, _clip_to_unit_square

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union  # noqa: F401  (kept for future union needs)
    _HAVE_SHAPELY = True
except Exception:  # noqa: BLE001  (shapely optional at gen time)
    _HAVE_SHAPELY = False

Ndc = Tuple[float, float, float]
Pt = Tuple[float, float]
TaggedPt = Tuple[float, float, int]

# User flag per KEYPOINT_ORDER index (TL,TR,BR,BL) -> (TL=1, TR=2, BR=4, BL=3).
_FLAG_BY_KP_INDEX = {0: 1, 1: 2, 2: 4, 3: 3}
FLAG_CREATED = 5
# Clockwise perimeter rank of each KEYPOINT_ORDER index (TL,TR,BR,BL already CW).
_PERIM_RANK = {0: 0, 1: 1, 2: 2, 3: 3}
_MATCH_TOL = 1e-6


def _visible_ring_no_shapely(ndc_corners: Sequence[Ndc]) -> List[Pt]:
    """Fallback (no shapely): frustum-clipped visible outline, occlusion skipped."""
    return _clip_to_unit_square([(x, y) for (x, y, _z) in ndc_corners])


def _signed_area(pts: Sequence[Pt]) -> float:
    s = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def _bridge_holes(poly: "Polygon") -> List[Pt]:
    """Return a single exterior ring for `poly`, splicing any interior rings (holes)
    into the exterior via a zero-width keyhole so a concave-with-hole result is still
    one perimeter (the format has no hole support). Holes are rare (a fully-interior
    occluder); the slit keeps the carved area excluded."""
    ext = list(poly.exterior.coords)[:-1]
    holes = [list(r.coords)[:-1] for r in poly.interiors]
    for hole in holes:
        # Connect the closest exterior/hole vertex pair with a doubled edge (keyhole).
        best = None
        for ei, e in enumerate(ext):
            for hi, h in enumerate(hole):
                d = (e[0] - h[0]) ** 2 + (e[1] - h[1]) ** 2
                if best is None or d < best[0]:
                    best = (d, ei, hi)
        _d, ei, hi = best
        hole_seq = hole[hi:] + hole[:hi] + [hole[hi]]     # hole loop back to entry
        ext = ext[:ei + 1] + hole_seq + [ext[ei]] + ext[ei + 1:]
    return ext


def _largest_ring(geom) -> Optional[List[Pt]]:
    """Exterior ring (holes bridged) of the largest polygon in a (Multi)Polygon."""
    if geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        poly = geom
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        polys = [g for g in geom.geoms if getattr(g, "geom_type", "") == "Polygon" and g.area > 0]
        if not polys:
            return None
        poly = max(polys, key=lambda p: p.area)
    else:
        return None
    if poly.area <= 0:
        return None
    return _bridge_holes(poly)


def _occlude_shapely(card_quad: List[Pt], occluders: Sequence[Tuple[List[Pt], float]],
                     card_depth: float, area_frac: float) -> Optional[List[Pt]]:
    """Clip the card quad to the frustum, then carve nearer occluders covering >
    area_frac of the current bound. Returns the exterior ring, or None if nothing
    visible remains."""
    frustum = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    bound = Polygon(card_quad).buffer(0)         # buffer(0) fixes tiny self-touching
    bound = bound.intersection(frustum)
    if bound.is_empty or bound.area <= 0:
        return None
    # Nearer occluders first (a closer rectangle carves before a farther one).
    for quad, depth in sorted(occluders, key=lambda o: o[1]):
        if depth >= card_depth - 1e-9:            # not in front of this card
            continue
        if bound.is_empty or bound.area <= 0:
            break
        occ = Polygon(quad).buffer(0)
        if occ.is_empty:
            continue
        inter = bound.intersection(occ)
        if inter.area > area_frac * bound.area:
            bound = bound.difference(occ)
    return _largest_ring(bound)


def _tag_and_order(ring: Sequence[Pt], ndc_corners: Sequence[Ndc]) -> List[TaggedPt]:
    """Tag each ring vertex (match to an in-frustum original corner -> its flag, else
    FLAG_CREATED), enforce the canonical (card-quad) winding, and rotate to start at
    the lowest-rank present corner. Returns YOLO-space (top-left origin) tagged points."""
    # Original in-frustum corners as (yolo_xy, flag, perim_rank), keyed for matching.
    originals = []
    for i, (x, y, z) in enumerate(ndc_corners):
        if corner_in_frustum(x, y, z):
            originals.append((ndc_to_yolo(x, y), _FLAG_BY_KP_INDEX[i], _PERIM_RANK[i]))

    def match(pt: Pt):
        for (oxy, flag, rank) in originals:
            if abs(pt[0] - oxy[0]) <= _MATCH_TOL and abs(pt[1] - oxy[1]) <= _MATCH_TOL:
                return flag, rank
        return FLAG_CREATED, None

    yolo = [ndc_to_yolo(px, py) for (px, py) in ring]
    tagged = [(p[0], p[1], match(p)[0]) for p in yolo]
    ranks = [match(p)[1] for p in yolo]

    # Canonical winding = same orientation as the projected card quad TL,TR,BR,BL.
    card_quad = [ndc_to_yolo(x, y) for (x, y, _z) in ndc_corners]
    want = _signed_area(card_quad)
    have = _signed_area(yolo)
    if want != 0.0 and have != 0.0 and (want > 0) != (have > 0):
        tagged.reverse()
        ranks.reverse()

    # Rotate to start at the present corner with the smallest perimeter rank.
    start = 0
    best_rank = None
    for idx, r in enumerate(ranks):
        if r is not None and (best_rank is None or r < best_rank):
            best_rank, start = r, idx
    return tagged[start:] + tagged[:start]


def compute_bound(
    card_id: str,
    ndc_corners: Sequence[Ndc],
    front_visible: bool,
    occluders: Sequence[Tuple[List[Pt], float]] = (),
    card_depth: Optional[float] = None,
    area_frac: float = 0.25,
) -> Tuple[Optional[List[TaggedPt]], int, str]:
    """Second-pass label bound for one card.

    Returns (tagged_points | None, class_id, reason). class_id is 0 'card' (all 4
    corners in frustum) or 1 'partial_card' (some out). reason ∈
    {'labeled','labeled-partial','back-facing','fully-out-of-frustum'}.
    `occluders` are (quad_xy, depth) in the SAME 2D (Blender-NDC) space as the card
    corners; only those nearer than `card_depth` and covering > area_frac carve.
    """
    if not front_visible:
        return None, 0, "back-facing"
    inside = [corner_in_frustum(x, y, z) for (x, y, z) in ndc_corners]
    n_in = sum(inside)
    class_id = 0 if n_in == len(ndc_corners) else 1
    if n_in == 0 and any(z <= 1e-6 for (_x, _y, z) in ndc_corners):
        return None, class_id, "fully-out-of-frustum"

    if _HAVE_SHAPELY:
        card_2d = [(x, y) for (x, y, _z) in ndc_corners]
        cd = card_depth if card_depth is not None else float("inf")
        ring = _occlude_shapely(card_2d, occluders, cd, area_frac)
    else:
        ring = _visible_ring_no_shapely(ndc_corners)   # occlusion skipped

    if not ring or len(ring) < 3:
        return None, class_id, "fully-out-of-frustum"
    tagged = _tag_and_order(ring, ndc_corners)
    if len(tagged) < 3:
        return None, class_id, "fully-out-of-frustum"
    return tagged, class_id, ("labeled" if class_id == 0 else "labeled-partial")
