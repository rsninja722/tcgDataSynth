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
from labeltools.yolo_pose import CardLabel, PolyLabel
from labeltools.occlusion import compute_bound, require_shapely
from blender import scene_builder as sb
from blender import protection as prot


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

    reason ∈ {'labeled','labeled-partial','back-facing','fully-out-of-frustum'}.
    A 'labeled-partial' result carries class 1 (partial_card) with 3-8 keypoints.
    debug_rows are (corner_index, ndc_x, ndc_y, ndc_z, in_frustum) for inspection.
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


# --------------------------------------------------------------------------- #
# Occlusion second pass (spec: 2026-07-22 user). Each card is also an OCCLUDER:
# its physical rectangle(s) hide parts of cards behind it. label_scene runs the
# frustum pass then carves each card's bound where a nearer rectangle covers >25%.
# --------------------------------------------------------------------------- #
def occluder_quads_world(inst):
    """World-space occluder rectangle(s) for a card instance, per protection (user):
    bare -> the card rect; sleeve -> the sleeve's outer rect; toploader/semi-rigid/slab
    -> BOTH plastic layers (front + back). Each quad is 4 world Vectors (TL,TR,BR,BL)."""
    pcfg = getattr(inst, "protection", None)
    kind = pcfg.kind if pcfg is not None else "none"
    mw = inst.root.matrix_world                       # protection is centered on the root
    fw, fh = sb.protection_footprint(pcfg) if pcfg is not None else (config.CARD_W_M, config.CARD_H_M)
    hw, hh = fw / 2.0, fh / 2.0

    def quad(zc):
        return [mw @ Vector((-hw, +hh, zc)), mw @ Vector((+hw, +hh, zc)),
                mw @ Vector((+hw, -hh, zc)), mw @ Vector((-hw, -hh, zc))]

    if kind in ("toploader", "semi_rigid"):
        s = prot.TOPLOADER_GAP / 2.0 + config.CARD_T_M
        return [quad(+s), quad(-s)]
    if kind == "slab":
        s = prot.SLAB_T / 2.0
        return [quad(+s), quad(-s)]
    return [quad(config.CARD_T_M / 2.0)]              # bare card / sleeve: one rect


def _project_quads(scene, cam, quads):
    """Project each world quad to (quad_2d, mean_depth); drop quads with any corner
    behind the camera (their 2D projection is unreliable)."""
    out = []
    for qw in quads:
        proj = [world_to_camera_view(scene, cam, p) for p in qw]
        if any(pv.z <= 1e-6 for pv in proj):
            continue
        out.append(([(pv.x, pv.y) for pv in proj], sum(pv.z for pv in proj) / len(proj)))
    return out


def label_scene(scene, cam, instances, area_frac: float = 0.25):
    """Two-pass, occlusion-aware labeling for a whole scene.

    Returns a list of (instance, PolyLabel|None, reason). Each card's visible-region
    bound is carved by every OTHER card's nearer rectangle(s) that cover > area_frac
    of its current bound. Shapely is mandatory; missing it is a setup error.
    """
    require_shapely()
    projected = []   # per instance: (ndc_corners, front_visible, card_depth)
    occ_quads = []   # per instance: list[(quad_2d, depth)]
    for inst in instances:
        mw = inst.card.matrix_world
        ndc = []
        for lc in ideal_corners_local():
            v = world_to_camera_view(scene, cam, mw @ lc)
            ndc.append((v.x, v.y, v.z))
        fv = is_front_visible(front_normal_world(inst.card),
                              cam.matrix_world.translation, mw.translation)
        depth = sum(z for (_x, _y, z) in ndc) / 4.0
        projected.append((ndc, fv, depth))
        occ_quads.append(_project_quads(scene, cam, occluder_quads_world(inst)))

    results = []
    for i, inst in enumerate(instances):
        ndc, fv, depth = projected[i]
        occluders = [q for j, ql in enumerate(occ_quads) if j != i for q in ql]
        pts, cls, reason = compute_bound(inst.card_id, ndc, fv, occluders=occluders,
                                         card_depth=depth, area_frac=area_frac)
        label = None
        if pts:
            label = PolyLabel(inst.card_id, tuple(pts), holo_tag=inst.holo_tag, class_id=cls)
        results.append((inst, label, reason))
    return results
