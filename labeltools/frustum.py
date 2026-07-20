"""
Frustum containment + label decision (bpy-FREE, Docker-testable).

The Blender side computes, per card, the four ideal corners' camera-view NDC via
world_to_camera_view (Blender convention: (0,0)=bottom-left, (1,1)=top-right,
z = depth along view axis, NEGATIVE means behind camera) and a front-face-visible
boolean. Everything after that — the Y-flip to YOLO's top-left origin, the
frustum test, and the label/skip decision — is pure arithmetic and lives here so
it can be verified without Blender.

Spec §3.9 + user rule:
  - A card is labeled iff ALL four ideal corners are inside the frustum
    (0<=x<=1, 0<=y<=1, in front of camera) AND its front face is visible.
  - Corners occluded by fingers/glare/sleeves are still labeled — only frustum
    containment matters. Back-facing cards are NOT labeled.
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


def classify(
    card_id: str,
    ndc_corners: Sequence[Ndc],
    front_visible: bool,
) -> Tuple[Optional[CardLabel], str]:
    """Decide whether to label. ndc_corners are the 4 ideal corners in
    KEYPOINT_ORDER (TL,TR,BR,BL) as Blender camera-view coords.

    Returns (CardLabel or None, reason) where reason is one of
    'labeled' | 'back-facing' | 'corner-out-of-frustum'.
    """
    if not front_visible:
        return None, "back-facing"
    if not all(corner_in_frustum(x, y, z) for (x, y, z) in ndc_corners):
        return None, "corner-out-of-frustum"
    corners_yolo: List[Tuple[float, float]] = [ndc_to_yolo(x, y) for (x, y, _z) in ndc_corners]
    return CardLabel(card_id, tuple(corners_yolo)), "labeled"
