"""Phase 5 camera-relative lighting rig (bpy REQUIRED)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import bpy
import numpy as np
from mathutils import Matrix, Vector

from texturegen import shadow_mask


@dataclass
class LightRig:
    objects: List[object]
    sun: object
    spotlight: Optional[object]
    points: List[object]
    occluders: List[object]
    materials: List[object] = field(default_factory=list)


_SHADOW_SIZE_MARGIN = 1.06
_CAMERA_PLANE_EPSILON_M = 0.002
SHADOW_OPACITY = 0.95
_SPOT_SOFT_SIZE_UNMASKED_M = 0.012
_SPOT_SOFT_SIZE_MASKED_M = (0.012 + 0.025) / 2.0


def _camera_basis(camera, target):
    target = Vector(target)
    front = (camera.matrix_world.translation - target).normalized()
    rotation = camera.matrix_world.to_3x3()
    right = (rotation @ Vector((1.0, 0.0, 0.0))).normalized()
    up = (rotation @ Vector((0.0, 1.0, 0.0))).normalized()
    return right, up, front


def _aim_minus_z(obj, target):
    direction = (Vector(target) - obj.location).normalized()
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _temperature_to_rgb(kelvin: float):
    """Approximate black-body RGB for Blender light colors."""
    temperature = max(1000.0, min(40000.0, float(kelvin))) / 100.0
    if temperature <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(temperature) - 161.1195681661
        blue = (0.0 if temperature <= 19.0 else
                138.5177312231 * math.log(temperature - 10.0) - 305.0447927307)
    else:
        red = 329.698727446 * ((temperature - 60.0) ** -0.1332047592)
        green = 288.1221695283 * ((temperature - 60.0) ** -0.0755148492)
        blue = 255.0
    return tuple(max(0.0, min(255.0, value)) / 255.0
                 for value in (red, green, blue))


def _new_light(name: str, kind: str, energy: float, color):
    data = bpy.data.lights.new(name, type=kind)
    data.energy = float(energy)
    data.color = tuple(float(value) for value in color)
    obj = bpy.data.objects.new(name, data)
    obj["tcg_phase5_environment"] = True
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _shadow_material():
    material = bpy.data.materials.new("Phase5ShadowMaskMat")
    material.diffuse_color = (0.035, 0.04, 0.05, SHADOW_OPACITY)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.035, 0.04, 0.05, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.8
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = SHADOW_OPACITY
    output = nodes.get("Material Output")
    for link in list(output.inputs["Surface"].links):
        links.remove(link)
    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    material["tcg_shadow_opacity"] = SHADOW_OPACITY
    return material


def _plane_axes(normal, camera_right, camera_up):
    """Camera-stable local X/Y axes perpendicular to the source direction."""
    normal = Vector(normal).normalized()
    axis_x = Vector(camera_right) - normal * Vector(camera_right).dot(normal)
    if axis_x.length < 1e-7:
        axis_x = Vector(camera_up) - normal * Vector(camera_up).dot(normal)
    if axis_x.length < 1e-7:
        fallback = Vector((1.0, 0.0, 0.0))
        if abs(fallback.dot(normal)) > 0.9:
            fallback = Vector((0.0, 1.0, 0.0))
        axis_x = fallback - normal * fallback.dot(normal)
    axis_x.normalize()
    axis_y = normal.cross(axis_x).normalized()
    return axis_x, axis_y


def _visible_scene_radius(camera, target, subject_extent_m: float) -> float:
    """Conservative camera-frame radius over the subject's depth interval."""
    distance = (camera.matrix_world.translation - Vector(target)).length
    far_distance = distance + max(float(subject_extent_m) / 2.0, 0.001)
    half_width = far_distance * math.tan(float(camera.data.angle_x) / 2.0)
    half_height = far_distance * math.tan(float(camera.data.angle_y) / 2.0)
    return math.hypot(half_width, half_height) * _SHADOW_SIZE_MARGIN


def _finite_shadow_transform(source, camera, target, visible_radius,
                             camera_right, camera_up, source_radius: float):
    target = Vector(target)
    source = Vector(source)
    source_vector = source - target
    source_distance = source_vector.length
    if source_distance <= 1e-6:
        raise RuntimeError("Shadow-mask light source coincides with the scene target")
    normal = source_vector / source_distance
    axis_x, axis_y = _plane_axes(normal, camera_right, camera_up)
    camera_front = (camera.matrix_world.translation - target).normalized()
    camera_distance = (camera.matrix_world.translation - target).length
    front_dot = normal.dot(camera_front)
    corner_factor = abs(axis_x.dot(camera_front)) + abs(axis_y.dot(camera_front))
    source_radius = max(0.0, float(source_radius))

    # The blocker shrinks toward a point source but must approach the radius of a
    # finite emitter. This both softens grid-shaped edges and keeps the no-hole control
    # large enough to cover the visible frame.
    current_denominator = front_dot + (
        (visible_radius - source_radius) * corner_factor / source_distance)
    current_required_distance = math.inf
    if current_denominator > 1e-9:
        current_required_distance = (
            camera_distance + visible_radius * corner_factor
            + _CAMERA_PLANE_EPSILON_M
        ) / current_denominator
    source_clearance = min(
        source_distance * 0.20, max(0.002, source_radius * 1.1))
    current_maximum_distance = source_distance - source_clearance
    current_axial_distance = (
        (current_required_distance + current_maximum_distance) / 2.0
        if current_required_distance < current_maximum_distance
        else source_distance * 0.35)

    # Midpoint the finite-emitter placement with the earlier point-source placement.
    old_denominator = front_dot + visible_radius * corner_factor / source_distance
    old_required_distance = math.inf
    if old_denominator > 1e-9:
        old_required_distance = (
            camera_distance + visible_radius * corner_factor
            + _CAMERA_PLANE_EPSILON_M
        ) / old_denominator
    old_maximum_distance = source_distance * 0.92
    old_axial_distance = (
        (old_required_distance + old_maximum_distance) / 2.0
        if old_required_distance < old_maximum_distance
        else source_distance * 0.35)
    axial_distance = (old_axial_distance + current_axial_distance) / 2.0

    source_fraction = axial_distance / source_distance
    half_size = (visible_radius * (1.0 - source_fraction)
                 + source_radius * source_fraction)
    camera_plane_min = (
        axial_distance * front_dot - camera_distance
        - half_size * corner_factor)
    placement = ("behind_camera" if camera_plane_min > _CAMERA_PLANE_EPSILON_M
                 else "camera_invisible_fallback")
    center = target + normal * axial_distance
    return (center, normal, axis_x, axis_y, half_size, placement,
            source_fraction)


def _new_shadow_plane(name: str, seed: int, source_kind: str, source_index: int,
                      transform, material, solid: bool = False):
    center, normal, axis_x, axis_y, half_size, placement, source_fraction = transform
    retained = (np.ones((shadow_mask.GRID_FACES, shadow_mask.GRID_FACES), dtype=bool)
                if solid else shadow_mask.retained_faces(seed))
    vertices, faces = shadow_mask.unit_grid_geometry(retained)
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices.tolist(), [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    matrix = Matrix((axis_x, axis_y, normal)).transposed().to_4x4()
    matrix.translation = center
    obj.matrix_world = matrix
    obj.scale = (2.0 * half_size, 2.0 * half_size, 1.0)
    obj.data.materials.append(material)
    obj["tcg_phase5_environment"] = True
    obj["tcg_shadow_source"] = source_kind
    obj["tcg_shadow_source_index"] = int(source_index)
    obj["tcg_shadow_mask_seed"] = int(seed)
    obj["tcg_shadow_grid_faces"] = int(shadow_mask.GRID_FACES)
    obj["tcg_shadow_retained_faces"] = int(np.count_nonzero(retained))
    obj["tcg_shadow_candidate_faces"] = int(retained.size)
    obj["tcg_shadow_half_size_m"] = float(half_size)
    obj["tcg_shadow_placement"] = placement
    obj["tcg_shadow_source_fraction"] = float(source_fraction)
    obj["tcg_shadow_solid_control"] = bool(solid)
    obj["tcg_shadow_opacity"] = SHADOW_OPACITY

    if not hasattr(obj, "visible_camera") or not hasattr(obj, "visible_shadow"):
        raise RuntimeError("Blender object ray visibility is required for shadow masks")
    obj.visible_camera = False
    obj.visible_shadow = True
    for attribute, value in (
        ("visible_diffuse", False),
        ("visible_glossy", False),
        ("visible_transmission", False),
        ("visible_volume_scatter", False),
    ):
        if hasattr(obj, attribute):
            setattr(obj, attribute, value)
    return obj


def build_lighting(lighting_cfg, camera, target, subject_extent_m: float,
                   solid_shadow_masks: bool = False) -> LightRig:
    """Build configured lights and deterministic per-source shadow masks."""
    bpy.context.view_layer.update()
    target = Vector(target)
    right, up, front = _camera_basis(camera, target)
    objects = []

    elevation = math.radians(float(lighting_cfg.sun_angle_deg[0]))
    azimuth = math.radians(float(lighting_cfg.sun_angle_deg[1]))
    source_direction = (
        right * (math.cos(elevation) * math.sin(azimuth))
        + up * (math.cos(elevation) * math.cos(azimuth))
        + front * math.sin(elevation)
    ).normalized()
    sun = _new_light("Phase5Sun", "SUN", lighting_cfg.sun_energy, (1.0, 0.97, 0.92))
    sun.location = target + source_direction * max(subject_extent_m, 0.2)
    sun.rotation_euler = (-source_direction).to_track_quat("-Z", "Y").to_euler()
    front_dot = float(source_direction.dot(front))
    if front_dot <= 0.0:
        raise RuntimeError("Sampled sun resolved behind the camera-visible hemisphere")
    sun["tcg_front_dot"] = front_dot
    objects.append(sun)

    spotlight = None
    if lighting_cfg.spotlight_beside_camera:
        spotlight = _new_light("Phase5PhoneFlash", "SPOT", 14.0625, (0.78, 0.88, 1.0))
        spotlight.data.spot_size = math.radians(68.0)
        spotlight.data.spot_blend = 0.28
        spotlight.data.shadow_soft_size = (
            _SPOT_SOFT_SIZE_MASKED_M
            if lighting_cfg.spotlight_shadow_mask_seed is not None
            else _SPOT_SOFT_SIZE_UNMASKED_M)
        spotlight.location = camera.matrix_world.translation + right * 0.035 + up * 0.015
        _aim_minus_z(spotlight, target)
        objects.append(spotlight)

    points = []
    for index, point_cfg in enumerate(lighting_cfg.point_lights):
        point = _new_light(f"Phase5Point{index}", "POINT", point_cfg.intensity,
                           _temperature_to_rgb(point_cfg.color_temp))
        x, y, z = (float(value) for value in point_cfg.position)
        point.location = target + right * x + up * y + front * z
        if point_cfg.shadow_mask_seed is not None:
            old_soft_size = max(0.005, subject_extent_m * 0.025)
            current_soft_size = max(0.012, subject_extent_m * 0.05)
            point.data.shadow_soft_size = (old_soft_size + current_soft_size) / 2.0
        else:
            point.data.shadow_soft_size = max(0.005, subject_extent_m * 0.025)
        point["tcg_color_temp"] = float(point_cfg.color_temp)
        point["tcg_front_dot"] = float((point.location - target).dot(front))
        points.append(point)
        objects.append(point)

    occluders = []
    materials = []
    mask_specs = []
    if lighting_cfg.spotlight_shadow_mask_seed is not None:
        mask_specs.append(("spotlight", -1, lighting_cfg.spotlight_shadow_mask_seed,
                           spotlight.location, spotlight.data.shadow_soft_size))
    for index, (point, point_cfg) in enumerate(zip(points, lighting_cfg.point_lights)):
        if point_cfg.shadow_mask_seed is not None:
            mask_specs.append(("point", index, point_cfg.shadow_mask_seed,
                               point.location, point.data.shadow_soft_size))

    if mask_specs:
        material = _shadow_material()
        materials.append(material)
        visible_radius = _visible_scene_radius(camera, target, subject_extent_m)
        for source_kind, source_index, seed, source_location, source_radius in mask_specs:
            transform = _finite_shadow_transform(
                source_location, camera, target, visible_radius, right, up,
                source_radius)
            occluder = _new_shadow_plane(
                f"Phase5ShadowMask_{source_kind}_{source_index}", seed,
                source_kind, source_index, transform, material,
                solid=solid_shadow_masks)
            occluders.append(occluder)
            objects.append(occluder)

    return LightRig(objects=objects, sun=sun, spotlight=spotlight, points=points,
                    occluders=occluders, materials=materials)


def remove_light_rig(rig: Optional[LightRig]) -> None:
    """Remove a generated rig without touching cards or layout objects."""
    if rig is None:
        return
    for obj in reversed(rig.objects):
        if obj.name not in bpy.data.objects:
            continue
        data = obj.data
        obj_type = obj.type
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            if obj_type == "LIGHT":
                bpy.data.lights.remove(data)
            elif obj_type == "MESH":
                bpy.data.meshes.remove(data)
    for material in rig.materials:
        if material.users == 0:
            bpy.data.materials.remove(material)
