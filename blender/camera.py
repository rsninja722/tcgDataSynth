"""Phase 5 perspective camera placement and DoF focus (bpy REQUIRED)."""
from __future__ import annotations

import math

import bpy
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
