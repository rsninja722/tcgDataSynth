"""Phase 5 perspective camera placement and DoF focus (bpy REQUIRED)."""
from __future__ import annotations

import math

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

import config
from blender import scene_builder as sb


def subject_target_and_extent(instances):
    """Return a world-space target and conservative diameter for card instances."""
    if not instances:
        raise ValueError("Camera framing requires at least one card instance")
    centers = [instance.root.matrix_world.translation.copy() for instance in instances]
    target = sum(centers, Vector((0.0, 0.0, 0.0))) / len(centers)
    radius = 0.0
    for instance, center in zip(instances, centers):
        width, height = sb.protection_footprint(instance.protection)
        half_depth = sb.protection_half_thickness(instance.protection)
        unit_radius = math.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2
                                + half_depth ** 2)
        radius = max(radius, (center - target).length + unit_radius)
    return target, max(2.0 * radius, config.CARD_H_M)


def subject_extent_from_target(instances, target) -> float:
    """Conservative subject diameter around an explicit world-space camera target."""
    target = Vector(target)
    radius = 0.0
    for instance in instances:
        center = instance.root.matrix_world.translation
        width, height = sb.protection_footprint(instance.protection)
        half_depth = sb.protection_half_thickness(instance.protection)
        unit_radius = math.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2
                                + half_depth ** 2)
        radius = max(radius, (center - target).length + unit_radius)
    return max(2.0 * radius, config.CARD_H_M)


def random_focus_target(instances, rng):
    """Choose a front-facing configured card and return its center and instance."""
    candidates = [instance for instance in instances
                  if not instance.back_to_camera and instance.label_enabled]
    if not candidates:
        raise RuntimeError("Camera focus requires a front-facing card instance")
    instance = candidates[int(rng.integers(0, len(candidates)))]
    return instance.card.matrix_world.translation.copy(), instance


def frame_distance(focal_mm: float, frame_extent_m: float,
                   target_fraction: float = 0.78) -> float:
    """Distance in meters that fits a subject diameter in the square frame."""
    if focal_mm <= 0.0 or frame_extent_m <= 0.0:
        raise ValueError("Focal length and frame extent must be positive")
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must be between zero and one")
    half_fov = math.atan(18.0 / float(focal_mm))  # 36mm horizontal sensor
    # The input is a bounding-sphere diameter, not a flat plane at target depth.
    # Convert desired image-plane fill to angular radius, then use the sphere tangent.
    framed_angle = math.atan(target_fraction * math.tan(half_fov))
    return (frame_extent_m / 2.0) / math.sin(framed_angle)


def build_camera(camera_cfg, target, frame_extent_m: float,
                 target_fraction: float = 0.78):
    """Build a configured camera on the sampled off-axis/orbit cone."""
    target = Vector(target)
    offaxis = math.radians(float(camera_cfg.offaxis_deg))
    orbit = math.radians(float(camera_cfg.orbit_deg))
    direction = Vector((math.sin(offaxis) * math.cos(orbit),
                        math.sin(offaxis) * math.sin(orbit),
                        math.cos(offaxis)))
    distance = frame_distance(camera_cfg.focal_mm, frame_extent_m, target_fraction)

    data = bpy.data.cameras.new("Phase5Camera")
    data.type = "PERSP"
    data.lens = float(camera_cfg.focal_mm)
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.shift_x = 0.0
    data.shift_y = 0.0
    data.clip_start = 0.001
    data.clip_end = 100.0
    data.dof.use_dof = bool(camera_cfg.dof_enabled)
    data.dof.aperture_fstop = float(camera_cfg.aperture_fstop)
    data.dof.focus_distance = distance

    camera = bpy.data.objects.new("Phase5Camera", data)
    camera.location = target + direction * distance
    camera.rotation_euler = (target - camera.location).normalized().to_track_quat(
        "-Z", "Y").to_euler()
    camera["tcg_offaxis_deg"] = float(camera_cfg.offaxis_deg)
    camera["tcg_orbit_deg"] = float(camera_cfg.orbit_deg)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    # matrix_world is consumed immediately by the camera-relative light builder.
    bpy.context.view_layer.update()
    return camera


def focus_on_labeled_card(camera, camera_cfg, label_results):
    """Focus on the emitted label nearest frame center and return its instance."""
    candidates = []
    for index, (instance, label, _reason) in enumerate(label_results):
        if label is None or not label.points:
            continue
        center_x = sum(point[0] for point in label.points) / len(label.points)
        center_y = sum(point[1] for point in label.points) / len(label.points)
        distance_sq = (center_x - 0.5) ** 2 + (center_y - 0.5) ** 2
        candidates.append((distance_sq, index, instance))
    if not candidates:
        raise RuntimeError("Camera framing produced no labeled card to focus on")

    instance = min(candidates, key=lambda item: (item[0], item[1]))[2]
    camera.data.dof.use_dof = bool(camera_cfg.dof_enabled)
    camera.data.dof.aperture_fstop = float(camera_cfg.aperture_fstop)
    camera.data.dof.focus_object = instance.card
    camera.data.dof.focus_distance = (
        camera.matrix_world.translation - instance.card.matrix_world.translation).length
    return instance


def focus_on_card(camera, camera_cfg, instance):
    """Set DoF to the card deliberately intersected by the center camera ray."""
    camera.data.dof.use_dof = bool(camera_cfg.dof_enabled)
    camera.data.dof.aperture_fstop = float(camera_cfg.aperture_fstop)
    camera.data.dof.focus_object = instance.card
    camera.data.dof.focus_distance = (
        camera.matrix_world.translation - instance.card.matrix_world.translation).length
    camera["tcg_focus_card_id"] = instance.card_id
    return instance


def _card_fully_contained(scene, camera, instance) -> bool:
    half_w = config.CARD_W_M / 2.0
    half_h = config.CARD_H_M / 2.0
    z = config.CARD_T_M / 2.0
    corners = ((-half_w, +half_h, z), (+half_w, +half_h, z),
               (+half_w, -half_h, z), (-half_w, -half_h, z))
    for corner in corners:
        projected = world_to_camera_view(scene, camera, instance.card.matrix_world @ Vector(corner))
        if projected.z <= 1e-6 or not (0.0 <= projected.x <= 1.0) \
                or not (0.0 <= projected.y <= 1.0):
            return False
    return True


def all_cards_fully_contained(scene, camera, instances) -> bool:
    """Whether every card's ideal front rectangle is inside the camera frustum."""
    return bool(instances) and all(
        _card_fully_contained(scene, camera, instance) for instance in instances)


def zoom_to_card_boundary(scene, camera, instances, rng,
                          increment_mm: float = 4.0, max_increments: int = 10000) -> float:
    """Increase lens zoom to the first card/frustum crossing, then roll back 0-2 steps.

    A scene that already contains a partial or out-of-view card is left unchanged.
    """
    if increment_mm <= 0.0 or max_increments < 1:
        raise ValueError("Camera zoom increments must be positive")
    bpy.context.view_layer.update()
    initial_lens = float(camera.data.lens)
    camera["tcg_zoom_initial_mm"] = initial_lens
    camera["tcg_zoom_increment_mm"] = float(increment_mm)
    if not all_cards_fully_contained(scene, camera, instances):
        camera["tcg_zoom_adjusted"] = False
        camera["tcg_zoom_rollback"] = -1
        return initial_lens

    for _step in range(max_increments):
        camera.data.lens = float(camera.data.lens) + increment_mm
        bpy.context.view_layer.update()
        if not all_cards_fully_contained(scene, camera, instances):
            crossing_lens = float(camera.data.lens)
            chance_no_rollback = int(rng.integers(0, 3))
            rollback = int(rng.integers(2, 5)) if chance_no_rollback > 0 else -2
            camera.data.lens = max(crossing_lens - rollback * increment_mm, initial_lens)
            camera["tcg_zoom_adjusted"] = True
            camera["tcg_zoom_crossing_mm"] = crossing_lens
            camera["tcg_zoom_rollback"] = rollback
            camera["tcg_zoom_final_mm"] = float(camera.data.lens)
            bpy.context.view_layer.update()
            return float(camera.data.lens)
    raise RuntimeError("Camera zoom did not reach a card boundary")


def remove_camera(camera) -> None:
    """Remove one generated camera and its datablock."""
    if camera is None:
        return
    data = camera.data
    if bpy.context.scene.camera is camera:
        bpy.context.scene.camera = None
    bpy.data.objects.remove(camera, do_unlink=True)
    if data.users == 0:
        bpy.data.cameras.remove(data)
