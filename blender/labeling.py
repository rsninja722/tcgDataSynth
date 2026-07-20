"""
Card corner labeling in Blender (bpy REQUIRED). Promoted from the Phase-1 test
script after the labeling math was validated against real renders.

Projects a card's four ideal (un-rounded, sharp) corner points with
world_to_camera_view and defers the label/skip DECISION to the Docker-tested
labeltools.frustum.classify. This is the single labeling path used by every
layout from Phase 4 on.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

import config
from labeltools.frustum import classify, is_front_visible
from labeltools.yolo_pose import CardLabel


def ideal_corners_local(
    width: float = config.CARD_W_M,
    height: float = config.CARD_H_M,
    z: float = config.CARD_T_M / 2.0,
) -> List[Vector]:
    """The 4 ideal corners in KEYPOINT_ORDER TL,TR,BR,BL, on the front face (z=+T/2)."""
    return [
        Vector((-width / 2, +height / 2, z)),  # TL
        Vector((+width / 2, +height / 2, z)),  # TR
        Vector((+width / 2, -height / 2, z)),  # BR
        Vector((-width / 2, -height / 2, z)),  # BL
    ]


def front_normal_world(obj) -> Vector:
    """World-space outward normal of the card's front (+Z local) face."""
    return (obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()


def label_card(
    scene,
    cam,
    obj,
    card_id: str,
    corners_local: Optional[Sequence[Vector]] = None,
    holo_tag: str = "none",
) -> Tuple[Optional[CardLabel], str, List[Tuple[int, float, float, float, bool]]]:
    """Return (CardLabel|None, reason, debug_rows) for one card object.

    reason ∈ {'labeled','back-facing','corner-out-of-frustum'}. debug_rows are
    (corner_index, ndc_x, ndc_y, ndc_z, in_frustum) for logging/inspection.
    holo_tag (none|full|holo|reverse) is written into the label.
    """
    corners_local = corners_local or ideal_corners_local()
    mw = obj.matrix_world
    front_visible = is_front_visible(
        front_normal_world(obj), cam.matrix_world.translation, mw.translation
    )
    ndc_corners: List[Tuple[float, float, float]] = []
    debug_rows = []
    for i, local in enumerate(corners_local):
        v = world_to_camera_view(scene, cam, mw @ local)
        ndc_corners.append((v.x, v.y, v.z))
        in_f = (0.0 <= v.x <= 1.0) and (0.0 <= v.y <= 1.0) and (v.z > 1e-6)
        debug_rows.append((i + 1, v.x, v.y, v.z, in_f))
    label, reason = classify(card_id, ndc_corners, front_visible, holo_tag=holo_tag)
    return label, reason, debug_rows
