"""CC0 rigged-hand loading, posing, placement, and skin materials (bpy REQUIRED).

The source is the validated Blender 2.79 ``Hands + armature.blend`` asset. Runtime
loading keeps only one selected mesh/armature pair, limits Multires to level 1, and
places the pair through a single root transform. The source file is never modified.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List

import bpy
from mathutils import Matrix, Vector

import config

_OBJECTS = {
    "left": ("Hand.L", "Hand_Left"),
    "right": ("Hand.R", "Hand_Right"),
}
_CONTROLS = ("index control", "Major control", "Ring control", "Pinky control")
_FINGER_TIPS = {
    "index": "Bone.002",
    "middle": "Bone.005",
    "ring": "Bone.008",
    "pinky": "Bone.011",
    "thumb": "Bone.019",
}

# Curated plausible skin colors, sampled and interpolated by the scene RNG.
_SKIN_TONES = (
    (0.78, 0.52, 0.36),
    (0.66, 0.38, 0.23),
    (0.52, 0.27, 0.14),
    (0.39, 0.18, 0.085),
    (0.27, 0.105, 0.050),
    (0.16, 0.060, 0.030),
)


@dataclass
class HandInstance:
    root: object
    mesh: object
    armature: object
    objects: List[object]

def _skin_color(rng):
    index = int(rng.integers(0, len(_SKIN_TONES)))
    other = min(index + 1, len(_SKIN_TONES) - 1)
    amount = float(rng.uniform(0.0, 0.65)) if other != index else 0.0
    return tuple((1.0 - amount) * _SKIN_TONES[index][i]
                 + amount * _SKIN_TONES[other][i] for i in range(3))


def make_skin_material(name: str, rng):
    """Seeded procedural material with broad tone coverage and subtle skin detail."""
    base = _skin_color(rng)
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        raise RuntimeError("Principled BSDF is unavailable for the hand material")
    bsdf.inputs["Roughness"].default_value = float(rng.uniform(0.42, 0.58))
    if bsdf.inputs.get("IOR") is not None:
        bsdf.inputs["IOR"].default_value = 1.4
    if bsdf.inputs.get("Subsurface Weight") is not None:
        bsdf.inputs["Subsurface Weight"].default_value = float(rng.uniform(0.035, 0.07))

    coordinate = nodes.new("ShaderNodeTexCoord")
    mottling = nodes.new("ShaderNodeTexNoise")
    mottling.inputs["Scale"].default_value = float(rng.uniform(3.0, 6.0))
    mottling.inputs["Detail"].default_value = float(rng.uniform(2.0, 4.0))
    mottling.inputs["Roughness"].default_value = 0.55
    mottling.inputs["Distortion"].default_value = float(rng.uniform(0.08, 0.2))
    ramp = nodes.new("ShaderNodeValToRGB")
    darker = tuple(max(0.0, channel * float(rng.uniform(0.74, 0.84))) for channel in base)
    lighter = tuple(min(1.0, channel * float(rng.uniform(1.04, 1.11)) + 0.008)
                    for channel in base)
    ramp.color_ramp.elements[0].color = (*darker, 1.0)
    ramp.color_ramp.elements[1].color = (*lighter, 1.0)
    ramp.color_ramp.elements[0].position = 0.26
    ramp.color_ramp.elements[1].position = 0.74

    pores = nodes.new("ShaderNodeTexNoise")
    pores.inputs["Scale"].default_value = float(rng.uniform(80.0, 120.0))
    pores.inputs["Detail"].default_value = 2.0
    pores.inputs["Roughness"].default_value = 0.7
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = float(rng.uniform(0.045, 0.09))
    bump.inputs["Distance"].default_value = 0.0003

    links.new(coordinate.outputs["Generated"], mottling.inputs["Vector"])
    links.new(mottling.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(coordinate.outputs["Generated"], pores.inputs["Vector"])
    links.new(pores.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return material


def _load_pair(name: str, handedness: str):
    if handedness not in _OBJECTS:
        raise ValueError(f"Unsupported handedness: {handedness!r}")
    path = config.hand_asset_path()
    mesh_name, armature_name = _OBJECTS[handedness]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Hand asset not found: {path!r}. Set TCG_HAND_ASSET to the CC0 .blend file."
        )

    with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
        missing = [item for item in (mesh_name, armature_name)
                   if item not in data_from.objects]
        if missing:
            raise RuntimeError(f"Hand asset is missing objects: {missing}")
        data_to.objects = [mesh_name, armature_name]
    loaded = [obj for obj in data_to.objects if obj is not None]

    for obj in loaded:
        if not obj.users_collection:
            bpy.context.scene.collection.objects.link(obj)
    mesh = next((obj for obj in loaded if obj.type == "MESH"), None)
    armature = next((obj for obj in loaded if obj.type == "ARMATURE"), None)
    if mesh is None or armature is None:
        raise RuntimeError("Hand asset did not resolve to one mesh and one armature")

    armature_modifiers = [modifier for modifier in mesh.modifiers
                          if modifier.type == "ARMATURE"]
    if len(armature_modifiers) != 1:
        raise RuntimeError(f"{mesh.name!r} needs exactly one Armature modifier")
    armature_modifiers[0].object = armature
    if mesh.parent is not armature:
        world = mesh.matrix_world.copy()
        mesh.parent = armature
        mesh.matrix_world = world
    for modifier in mesh.modifiers:
        if modifier.type == "MULTIRES":
            total = int(getattr(modifier, "total_levels", 1))
            modifier.levels = min(1, total)
            modifier.render_levels = min(1, total)
    missing_controls = [control for control in _CONTROLS
                        if control not in armature.pose.bones]
    if missing_controls:
        raise RuntimeError(f"Hand armature is missing controls: {missing_controls}")
    return mesh, armature


def _rotate_pose_bone(armature, bone_name, axis, degrees):
    pose_bone = armature.pose.bones.get(bone_name)
    if pose_bone is None:
        raise RuntimeError(f"Hand armature is missing pose bone {bone_name!r}")
    pose_bone.matrix_basis = pose_bone.matrix_basis @ Matrix.Rotation(
        math.radians(float(degrees)), 4, axis)


def _pose_grip(armature, handedness: str, grip: str, rng):
    variation = float(rng.uniform(0.9, 1.1))
    if grip == "pinch":
        curls = (28.0, 18.0, 22.0, 26.0)
        thumb_curls = (18.0, 22.0, 12.0)
        thumb_splay = 14.0 if handedness == "left" else -14.0
    elif grip == "side":
        curls = (32.0, 36.0, 40.0, 44.0)
        thumb_curls = (22.0, 26.0, 14.0)
        thumb_splay = 18.0 if handedness == "left" else -18.0
    else:
        raise ValueError(f"Unsupported grip: {grip!r}")

    for control, degrees in zip(_CONTROLS, curls):
        _rotate_pose_bone(armature, control, "X", -degrees * variation)
    for bone_name, degrees in zip(("Bone.017", "Bone.018", "Bone.019"), thumb_curls):
        _rotate_pose_bone(armature, bone_name, "X", -degrees * variation)
    _rotate_pose_bone(armature, "Bone.017", "Z", thumb_splay * variation)
    bpy.context.view_layer.update()


def _evaluated_bounds(mesh):
    evaluated = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
    low = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    return low, high


def _normalize(mesh, armature, name: str, target_size: float):
    root = bpy.data.objects.new(f"{name}_Root", None)
    bpy.context.scene.collection.objects.link(root)
    world = armature.matrix_world.copy()
    armature.parent = root
    armature.matrix_world = world
    if mesh.parent is not armature:
        world = mesh.matrix_world.copy()
        mesh.parent = root
        mesh.matrix_world = world

    bpy.context.view_layer.update()
    low, high = _evaluated_bounds(mesh)
    center = (low + high) * 0.5
    longest = max(high[i] - low[i] for i in range(3))
    if longest <= 1e-9:
        raise RuntimeError("Hand has degenerate evaluated bounds")
    root.matrix_world = (Matrix.Scale(target_size / longest, 4)
                         @ Matrix.Translation(-center))
    bpy.context.view_layer.update()
    return root


def _bone_tip_world(armature, finger):
    return armature.matrix_world @ armature.pose.bones[_FINGER_TIPS[finger]].tail


def _contact_points(armature, grip):
    thumb = _bone_tip_world(armature, "thumb")
    index = _bone_tip_world(armature, "index")
    if grip == "pinch":
        opposition = index
    else:
        middle = _bone_tip_world(armature, "middle")
        opposition = (index + middle) * 0.5
    return thumb, opposition


def _grip_target(footprint, approach_deg, depth, grip, z):
    half_w, half_h = float(footprint[0]) * 0.5, float(footprint[1]) * 0.5
    angle = math.radians(float(approach_deg))
    direction = Vector((math.cos(angle), math.sin(angle), 0.0))
    tx = float("inf") if abs(direction.x) < 1e-9 else half_w / abs(direction.x)
    ty = float("inf") if abs(direction.y) < 1e-9 else half_h / abs(direction.y)
    boundary = min(tx, ty)
    # Pinches may reach inward. Side grips stay at the outer edge; deeper side
    # placement made the palm/arm intersect rigid toploaders during t14 review.
    depth_scale = 0.10 if grip == "side" else 0.75
    inset = boundary * depth_scale * float(depth)
    return direction * (boundary - inset) + Vector((0.0, 0.0, float(z)))


def _contact_separation_rotation(thumb, opposition, desired_separation):
    """Shortest 3D rotation placing thumb ahead by the requested distance."""
    difference = thumb - opposition
    length = difference.length
    if length <= desired_separation + 1e-9:
        raise RuntimeError(
            f"Digit-tip separation {length:.4f}m cannot span {desired_separation:.4f}m")
    horizontal = Vector((difference.x, difference.y, 0.0))
    if horizontal.length <= 1e-9:
        horizontal = Vector((1.0, 0.0, 0.0))
    else:
        horizontal.normalize()
    horizontal *= math.sqrt(length * length - desired_separation * desired_separation)
    target_difference = horizontal + Vector((0.0, 0.0, desired_separation))
    rotation = difference.rotation_difference(target_difference).to_matrix().to_4x4()
    return rotation, (rotation @ difference).z


def build_hand(name: str, handedness: str, grip: str, footprint, approach_deg: float,
               depth: float, protection_half_thickness: float, card_z: float,
               rng) -> HandInstance:
    """Load and place one posed hand around a card/protection footprint.

    Canonical source orientation has the wrist toward +Y and fingertips toward -Y.
    The root rotates that wrist direction to ``approach_deg`` and aligns the selected
    fingertip contact midpoint to a shallow/normal point inside the holder boundary.
    """
    mesh, armature = _load_pair(name, handedness)
    mesh.name = f"{name}_Mesh"
    armature.name = f"{name}_Armature"
    _pose_grip(armature, handedness, grip, rng)
    root = _normalize(mesh, armature, name, float(rng.uniform(0.195, 0.225)))

    if not mesh.data.materials:
        mesh.data.materials.append(None)
    mesh.material_slots[0].link = "OBJECT"
    mesh.material_slots[0].material = make_skin_material(f"{name}_Skin", rng)
    mesh["tcg_opaque_occluder"] = True

    thumb, opposition = _contact_points(armature, grip)
    contact = (thumb + opposition) * 0.5
    wrist = armature.matrix_world @ armature.pose.bones["Bone.020"].head
    target = _grip_target(footprint, approach_deg, depth, grip, card_z)
    # Bone tails represent digit centers, so keep them several millimeters outside
    # the two physical protection faces rather than placing both at the same plane.
    desired_separation = 2.0 * float(protection_half_thickness) + 0.004
    thumb_forward, achieved_separation = _contact_separation_rotation(
        thumb, opposition, desired_separation)
    if abs(achieved_separation - desired_separation) > 0.002:
        raise RuntimeError(
            f"Could not place thumb/finger contacts across the protection: requested "
            f"{desired_separation:.4f}m, closest {achieved_separation:.4f}m")
    mesh["tcg_contact_separation_m"] = float(achieved_separation)
    rotated_wrist = thumb_forward @ (wrist - contact)
    current_approach = math.atan2(rotated_wrist.y, rotated_wrist.x)
    around_z = Matrix.Rotation(
        math.radians(float(approach_deg)) - current_approach, 4, "Z")
    placement = Matrix.Translation(target) @ around_z @ thumb_forward @ Matrix.Translation(-contact)
    root.matrix_world = placement @ root.matrix_world
    bpy.context.view_layer.update()
    return HandInstance(root=root, mesh=mesh, armature=armature,
                        objects=[root, mesh, armature])
