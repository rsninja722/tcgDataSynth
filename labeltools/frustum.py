"""
Frustum containment + label decision (bpy-FREE, Docker-testable).

The Blender side computes, per card, the four ideal corners' camera-view NDC via
world_to_camera_view (Blender convention: (0,0)=bottom-left, (1,1)=top-right,
z = depth along view axis, NEGATIVE means behind camera) and a front-face-visible
boolean. Everything after that — the Y-flip to YOLO's top-left origin, the
frustum test, and the label/skip decision — is pure arithmetic and lives here so
it can be verified without Blender.

Spec §3.9 + user rules:
  - Front face must be visible (back-facing cards are NEVER labeled).
  - ALL four ideal corners inside the frustum -> class 0 'card', 4 keypoints.
  - Card only PARTLY in frame -> class 1 'partial_card': keypoints trace the exact
    outline of the VISIBLE region = polygon (card quad) ∩ (frustum square). Those
    vertices are the in-frustum card corners, the card-edge/frustum-edge crossings,
    AND any frustum CORNER the card covers (a point interior to the card that lands on
    the frame corner). Variable count (up to 8). Occlusion is irrelevant.
  - Card does not overlap the frame at all -> not labeled (fully outside the frustum).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

try:
    import config
except ImportError:  # pragma: no cover
    from .. import config  # type: ignore

from labeltools.yolo_pose import CardLabel

Ndc = Tuple[float, float, float]  # Blender camera-view coords (x, y, z)

_Z_EPS = 1e-6


def corner_in_frustum(x: float, y: float, z: float) -> bool:
    """True if a single NDC corner is within the image rect AND in front of cam."""
    return (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0) and (z > _Z_EPS)


def ndc_to_yolo(x: float, y: float) -> Tuple[float, float]:
    """Blender NDC (bottom-left origin) -> YOLO normalized (top-left origin)."""
    return (x, 1.0 - y)


def is_front_visible(
    world_front_normal: Sequence[float],
    cam_pos: Sequence[float],
    card_pos: Sequence[float],
) -> bool:
    """Front face visible iff its outward normal points toward the camera.

    dot(front_normal, card->camera direction) > 0. Pure vector math (no bpy) so
    the rule is testable; the Blender side passes in world-space vectors.
    """
    dx = cam_pos[0] - card_pos[0]
    dy = cam_pos[1] - card_pos[1]
    dz = cam_pos[2] - card_pos[2]
    dot = world_front_normal[0] * dx + world_front_normal[1] * dy + world_front_normal[2] * dz
    return dot > 0.0


def _dedup_ring(pts: List[Tuple[float, float]], tol: float = 1e-9) -> List[Tuple[float, float]]:
    """Drop consecutive duplicate vertices and a closing vertex equal to the first."""
    out: List[Tuple[float, float]] = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) > tol or abs(p[1] - out[-1][1]) > tol:
            out.append(p)
    if len(out) >= 2 and abs(out[0][0] - out[-1][0]) <= tol and abs(out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


def _clip_to_unit_square(poly: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sutherland-Hodgman clip of a polygon against the unit square [0,1]^2.

    Returns the intersection polygon's vertices in order (the visible-region outline).
    They include surviving card corners, card-edge/square-edge crossings, and any
    square CORNER the polygon covers. Boundary vertices have a component == 0 or 1.
    """
    def _clip(pts, inside, cut):
        out = []
        n = len(pts)
        for i in range(n):
            cur, prev = pts[i], pts[i - 1]
            c_in, p_in = inside(cur), inside(prev)
            if c_in:
                if not p_in:
                    out.append(cut(prev, cur))
                out.append(cur)
            elif p_in:
                out.append(cut(prev, cur))
        return out

    def _cut_x(p, q, xb):
        t = (xb - p[0]) / (q[0] - p[0])
        return (xb, p[1] + (q[1] - p[1]) * t)

    def _cut_y(p, q, yb):
        t = (yb - p[1]) / (q[1] - p[1])
        return (p[0] + (q[0] - p[0]) * t, yb)

    pts = list(poly)
    pts = _clip(pts, lambda p: p[0] >= 0.0, lambda p, q: _cut_x(p, q, 0.0))
    if pts:
        pts = _clip(pts, lambda p: p[0] <= 1.0, lambda p, q: _cut_x(p, q, 1.0))
    if pts:
        pts = _clip(pts, lambda p: p[1] >= 0.0, lambda p, q: _cut_y(p, q, 0.0))
    if pts:
        pts = _clip(pts, lambda p: p[1] <= 1.0, lambda p, q: _cut_y(p, q, 1.0))
    return _dedup_ring(pts)


def classify(
    card_id: str,
    ndc_corners: Sequence[Ndc],
    front_visible: bool,
    holo_tag: str = "none",
) -> Tuple[Optional[CardLabel], str]:
    """Decide whether/how to label. ndc_corners are the 4 ideal corners in
    KEYPOINT_ORDER (TL,TR,BR,BL) as Blender camera-view coords.

    Returns (CardLabel or None, reason) where reason is one of
    'labeled' | 'labeled-partial' | 'back-facing' | 'fully-out-of-frustum'.
    """
    if not front_visible:
        return None, "back-facing"
    inside = [corner_in_frustum(x, y, z) for (x, y, z) in ndc_corners]
    n_in = sum(inside)
    if n_in == len(ndc_corners):
        corners_yolo = [ndc_to_yolo(x, y) for (x, y, _z) in ndc_corners]
        return CardLabel(card_id, tuple(corners_yolo), holo_tag=holo_tag,
                         class_id=config.YOLO_CLASS_ID), "labeled"
    # A corner at/behind the camera makes its projected 2D polygon invalid. Properly
    # supporting this case requires clipping in 3D against the near plane first.
    # Until then, reject it rather than emitting a corrupt partial polygon.
    if any(z <= _Z_EPS for (_x, _y, z) in ndc_corners):
        return None, "fully-out-of-frustum"
    poly = _clip_to_unit_square([(x, y) for (x, y, _z) in ndc_corners])
    if len(poly) < 3:                         # no real overlap -> fully outside
        return None, "fully-out-of-frustum"
    corners_yolo = [ndc_to_yolo(x, y) for (x, y) in poly]
    return CardLabel(card_id, tuple(corners_yolo), holo_tag=holo_tag,
                     class_id=config.PARTIAL_CLASS_ID), "labeled-partial"
