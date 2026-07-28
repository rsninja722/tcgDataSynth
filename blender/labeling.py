"""
Card corner labeling in Blender (bpy REQUIRED). Promoted from the Phase-1 test
script after the labeling math was validated against real renders.

Projects a card's four ideal (un-rounded, sharp) corner points with
an idealized refractive ray where marked bulk acrylic is present, then defers the
label/skip decision to bpy-free labeltools geometry. This is the single labeling
path used by every layout from Phase 4 on.
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
from labeltools.refraction import (BOUNDS_MAX_PROPERTY, BOUNDS_MIN_PROPERTY,
                                   IOR_PROPERTY, RefractiveBox, RefractionError,
                                   solve_camera_ray)


class CornerProjectionError(RuntimeError):
    """One ideal card corner has no solved apparent camera ray."""


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
    boxes = _refractive_boxes(scene)
    linear = mw.to_3x3()
    card_x = (linear @ Vector((1.0, 0.0, 0.0))).normalized()
    card_y = (linear @ Vector((0.0, 1.0, 0.0))).normalized()
    ndc_corners: List[Tuple[float, float, float]] = []
    debug_rows = []
    for i, local in enumerate(corners_local):
        x, y, z = _project_apparent_corner(scene, cam, mw @ local, card_x, card_y,
                                            boxes, card_id)
        ndc_corners.append((x, y, z))
        in_f = (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0) and (z > 1e-6)
        debug_rows.append((i + 1, x, y, z, in_f))
    label, reason = classify(card_id, ndc_corners, front_visible, holo_tag=holo_tag)
    return label, reason, debug_rows


# --------------------------------------------------------------------------- #
# Apparent projection + occlusion second pass. Each marked refractive object is
# approximated as a finite homogeneous box. Every card polygon is projected once and
# reused as its opaque occluder so carving and labels share the same optical model.
# --------------------------------------------------------------------------- #
def _refractive_boxes(scene):
    boxes = []
    for obj in scene.objects:
        if IOR_PROPERTY not in obj:
            continue
        if BOUNDS_MIN_PROPERTY not in obj or BOUNDS_MAX_PROPERTY not in obj:
            raise RuntimeError(f"Refractor {obj.name!r} has no local optical bounds")
        bmin = Vector(tuple(float(v) for v in obj[BOUNDS_MIN_PROPERTY]))
        bmax = Vector(tuple(float(v) for v in obj[BOUNDS_MAX_PROPERTY]))
        center_local = (bmin + bmax) * 0.5
        half_local = (bmax - bmin) * 0.5
        matrix = obj.matrix_world
        linear = matrix.to_3x3()
        center = matrix @ center_local
        half_vectors = [linear @ Vector((half_local.x, 0.0, 0.0)),
                        linear @ Vector((0.0, half_local.y, 0.0)),
                        linear @ Vector((0.0, 0.0, half_local.z))]
        half_sizes = [vec.length for vec in half_vectors]
        if min(half_sizes) <= 1e-9:
            raise RuntimeError(f"Refractor {obj.name!r} has a degenerate world transform")
        boxes.append(RefractiveBox(
            center=tuple(center),
            axes=[tuple(vec.normalized()) for vec in half_vectors],
            half_sizes=half_sizes,
            ior=float(obj[IOR_PROPERTY]),
            name=obj.name,
        ))
    return boxes


def _project_apparent_corner(scene, cam, world_corner, card_x, card_y, boxes, card_name):
    direct = world_to_camera_view(scene, cam, world_corner)
    if direct.z <= 1e-6 or not boxes:
        return direct.x, direct.y, direct.z
    camera_pos = cam.matrix_world.translation
    try:
        ray = solve_camera_ray(tuple(camera_pos), tuple(world_corner),
                               tuple(card_x), tuple(card_y), boxes)
    except RefractionError as exc:
        raise CornerProjectionError(
            f"Could not project refracted corner for {card_name!r}: {exc}"
        ) from exc
    # Perspective image coordinates depend on ray direction, not the distance of this
    # helper point. Keep the direct target depth for physical near/far ordering.
    along_ray = camera_pos + Vector(tuple(float(v) for v in ray))
    apparent = world_to_camera_view(scene, cam, along_ray)
    return apparent.x, apparent.y, direct.z


def label_scene(scene, cam, instances, area_frac: float = 0.25,
                refraction_failures: Optional[List[dict]] = None):
    """Two-pass, occlusion-aware labeling for a whole scene.

    Returns a list of (instance, PolyLabel|None, reason). Each card's visible-region
    bound is carved by every OTHER card's nearer rectangle(s) that cover > area_frac
    of its current bound. Non-card scene geometry is not an occluder. Shapely is
    mandatory; missing it is a setup error.
    """
    require_shapely()
    boxes = _refractive_boxes(scene)
    projected = []   # per instance: (ndc_corners, front_visible, card_depth)
    occ_quads = []   # per instance: list[(quad_2d, depth)]
    for inst in instances:
        mw = inst.card.matrix_world
        linear = mw.to_3x3()
        card_x = (linear @ Vector((1.0, 0.0, 0.0))).normalized()
        card_y = (linear @ Vector((0.0, 1.0, 0.0))).normalized()
        local_corners = ideal_corners_local()
        ndc = []
        for corner_index, local_corner in enumerate(local_corners):
            try:
                projected_corner = _project_apparent_corner(
                    scene, cam, mw @ local_corner, card_x, card_y, boxes, inst.card_id)
            except CornerProjectionError as exc:
                if refraction_failures is not None:
                    refraction_failures.append({
                        "card_id": inst.card_id,
                        "instance_name": inst.root.name,
                        "protection_kind": inst.protection.kind,
                        "corner_index": corner_index + 1,
                        "corner_name": config.KEYPOINT_ORDER[corner_index],
                        "error": str(exc.__cause__ or exc),
                        "fallback": "direct-card-polygon",
                    })
                ndc = []
                for direct_corner in local_corners:
                    direct = world_to_camera_view(scene, cam, mw @ direct_corner)
                    ndc.append((direct.x, direct.y, direct.z))
                break
            ndc.append(projected_corner)
        fv = is_front_visible(front_normal_world(inst.card),
                              cam.matrix_world.translation, mw.translation)
        depth = sum(z for (_x, _y, z) in ndc) / 4.0
        projected.append((ndc, fv, depth))
        occ_quads.append([] if any(z <= 1e-6 for _x, _y, z in ndc)
                         else [([(x, y) for x, y, _z in ndc], depth)])

    results = []
    for i, inst in enumerate(instances):
        ndc, fv, depth = projected[i]
        occluders = [q for j, ql in enumerate(occ_quads) if j != i for q in ql]
        pts, cls, reason = compute_bound(
            inst.card_id, ndc, fv, occluders=occluders,
            card_depth=depth, area_frac=area_frac)
        label = None
        if pts:
            label = PolyLabel(inst.card_id, tuple(pts), holo_tag=inst.holo_tag, class_id=cls)
        results.append((inst, label, reason))
    return results
