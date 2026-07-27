r"""
Phase 4 (layout 5/5, checkpoint 1) - HAND ASSET VALIDATION.

Validates the supplied CC0 Blender 2.79 left/right hand meshes and armatures after
Blender 5.0 conversion. This does not yet construct card grips. It inventories the
asset, verifies mesh/armature relationships, normalizes both hands beside a real-scale
63 x 88mm card, replaces the legacy materials with two procedural skin tones, and
builds a 24-cell +/- X/Y/Z calibration grid for the four named finger controls.

HOW TO RUN (headless):
    set TCG_HAND_ASSET=C:\path\to\Hands + armature.blend
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tests\t13_hand_asset.py

OUTPUT (out/):
    t13_hand_asset_report.txt
    t13_hand_asset_front.png
    t13_hand_asset_side.png
    t13_hand_asset_oblique.png
    t13_hand_asset_controls.png

REPORT BACK: attach the report and four PNGs. PASS if both hands have intact geometry
at plausible scale beside the card; front/side/oblique views show no severe conversion
damage; light/dark skin materials look plausible; and at least one axis/sign in each
control-grid row visibly curls the corresponding finger. The report must identify two
mesh/armature pairs and four named controls per armature.

The source .blend is opened read-only through bpy.data.libraries.load and is never
saved or modified.
"""
from __future__ import annotations

import math
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import bpy
from mathutils import Matrix, Vector

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from blender import card_factory as cf  # noqa: E402
from blender import scene_common as sc  # noqa: E402
from blender.render_setup import setup_render  # noqa: E402

SEED = 20240724
EXPECTED_OBJECTS = ("Hand.L", "Hand.R", "Hand_Left", "Hand_Right")
CONTROL_NAMES = ("index control", "Major control", "Ring control", "Pinky control")
SKIN_LIGHT = (0.72, 0.43, 0.29)
SKIN_DARK = (0.16, 0.065, 0.035)
_LINES = []


def log(msg=""):
    print(msg)
    _LINES.append(str(msg))


def section(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


def _fmt_vec(value):
    return tuple(round(float(v), 6) for v in value)


def _write_report():
    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "t13_hand_asset_report.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LINES) + "\n")
    print(f"[t13] report written: {path}")


def _append_asset(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Hand asset not found: {path!r}. Set TCG_HAND_ASSET to the CC0 .blend file."
        )

    inventory = {}
    loaded = []
    with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
        for kind in ("objects", "meshes", "armatures", "materials", "images",
                     "texts", "actions", "collections", "scenes"):
            inventory[kind] = list(getattr(data_from, kind, ()))
        missing = [name for name in EXPECTED_OBJECTS if name not in data_from.objects]
        if missing:
            raise RuntimeError(f"Hand asset is missing expected objects: {missing}")
        data_to.objects = [name for name in EXPECTED_OBJECTS]
        loaded = data_to.objects

    collection = bpy.data.collections.new("T13_ImportedHands")
    bpy.context.scene.collection.children.link(collection)
    objects = [obj for obj in loaded if obj is not None]
    for obj in objects:
        if not obj.users_collection:
            collection.objects.link(obj)
    return inventory, objects, collection


def _find_pairs(objects):
    pairs = []
    for mesh in (obj for obj in objects if obj.type == "MESH"):
        modifiers = [mod for mod in mesh.modifiers if mod.type == "ARMATURE"]
        if len(modifiers) != 1 or modifiers[0].object is None:
            raise RuntimeError(
                f"Mesh {mesh.name!r} needs exactly one resolved Armature modifier"
            )
        armature = modifiers[0].object
        if armature.type != "ARMATURE":
            raise RuntimeError(f"Modifier target for {mesh.name!r} is not an armature")
        weighted = {group.name for group in mesh.vertex_groups}
        deform = {bone.name for bone in armature.data.bones if bone.use_deform}
        overlap = weighted & deform
        if len(overlap) < 15:
            raise RuntimeError(
                f"Mesh {mesh.name!r} has only {len(overlap)} matching deform groups"
            )
        pairs.append({"mesh": mesh, "armature": armature, "weighted": overlap})
    if len(pairs) != 2:
        raise RuntimeError(f"Expected two hand mesh/armature pairs, found {len(pairs)}")
    return sorted(pairs, key=lambda p: p["mesh"].name)


def _evaluated_bounds(mesh):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(depsgraph)
    points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
    low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    return low, high


def _evaluated_counts(mesh):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        return len(evaluated_mesh.vertices), len(evaluated_mesh.polygons)
    finally:
        evaluated.to_mesh_clear()


def _insert_normalized_root(pair, name, target, target_size):
    mesh, armature = pair["mesh"], pair["armature"]
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)

    top_level = [armature]
    if mesh.parent is not armature:
        top_level.append(mesh)
    for obj in top_level:
        world = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = world

    bpy.context.view_layer.update()
    low, high = _evaluated_bounds(mesh)
    center = (low + high) * 0.5
    longest = max(high[i] - low[i] for i in range(3))
    if longest <= 1e-9:
        raise RuntimeError(f"Degenerate evaluated bounds for {mesh.name!r}")
    scale = target_size / longest
    root.matrix_world = (Matrix.Translation(Vector(target)) @ Matrix.Scale(scale, 4)
                         @ Matrix.Translation(-center))
    bpy.context.view_layer.update()
    return root, scale


def _skin_material(name, base, seed_offset=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        raise RuntimeError("Blender 5 Principled BSDF node was not created")

    bsdf.inputs["Roughness"].default_value = 0.48
    if bsdf.inputs.get("IOR"):
        bsdf.inputs["IOR"].default_value = 1.4
    if bsdf.inputs.get("Subsurface Weight"):
        bsdf.inputs["Subsurface Weight"].default_value = 0.055

    texcoord = nt.nodes.new("ShaderNodeTexCoord")
    mottling = nt.nodes.new("ShaderNodeTexNoise")
    mottling.inputs["Scale"].default_value = 4.0
    mottling.inputs["Detail"].default_value = 3.0
    mottling.inputs["Roughness"].default_value = 0.55
    mottling.inputs["Distortion"].default_value = 0.12 + seed_offset
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    darker = tuple(max(0.0, c * 0.78) for c in base)
    lighter = tuple(min(1.0, c * 1.08 + 0.01) for c in base)
    ramp.color_ramp.elements[0].color = (*darker, 1.0)
    ramp.color_ramp.elements[1].color = (*lighter, 1.0)
    ramp.color_ramp.elements[0].position = 0.28
    ramp.color_ramp.elements[1].position = 0.72

    pores = nt.nodes.new("ShaderNodeTexNoise")
    pores.inputs["Scale"].default_value = 95.0
    pores.inputs["Detail"].default_value = 2.0
    pores.inputs["Roughness"].default_value = 0.7
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.075
    bump.inputs["Distance"].default_value = 0.00035

    nt.links.new(texcoord.outputs["Generated"], mottling.inputs["Vector"])
    nt.links.new(mottling.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(texcoord.outputs["Generated"], pores.inputs["Vector"])
    nt.links.new(pores.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _assign_object_material(obj, material):
    if not obj.data.materials:
        obj.data.materials.append(None)
    slot = obj.material_slots[0]
    slot.link = "OBJECT"
    slot.material = material


def _limit_multires(mesh, level):
    for modifier in mesh.modifiers:
        if modifier.type == "MULTIRES":
            modifier.levels = min(level, modifier.total_levels)
            modifier.render_levels = min(level, modifier.total_levels)


def _log_inventory(path, inventory, pairs):
    section("Environment and source inventory")
    log(f"Blender version: {bpy.app.version_string}")
    log(f"Source path: {path}")
    log(f"Source size: {os.path.getsize(path)} bytes")
    for kind, names in inventory.items():
        log(f"{kind}: {names}")

    section("Resolved hand pairs")
    for pair in pairs:
        mesh, armature = pair["mesh"], pair["armature"]
        bpy.context.view_layer.update()
        low, high = _evaluated_bounds(mesh)
        eval_vertices, eval_polygons = _evaluated_counts(mesh)
        log(f"PAIR mesh={mesh.name!r} armature={armature.name!r}")
        log(f"  mesh parent={getattr(mesh.parent, 'name', None)!r}")
        log(f"  mesh transform loc={_fmt_vec(mesh.location)} rot={_fmt_vec(mesh.rotation_euler)} "
            f"scale={_fmt_vec(mesh.scale)}")
        log(f"  arm transform loc={_fmt_vec(armature.location)} "
            f"rot={_fmt_vec(armature.rotation_euler)} scale={_fmt_vec(armature.scale)}")
        log(f"  base topology vertices={len(mesh.data.vertices)} polygons={len(mesh.data.polygons)}")
        log(f"  evaluated topology vertices={eval_vertices} polygons={eval_polygons}")
        log(f"  evaluated world bounds min={_fmt_vec(low)} max={_fmt_vec(high)}")
        log(f"  modifiers={[(m.name, m.type, getattr(m, 'levels', None), getattr(m, 'render_levels', None)) for m in mesh.modifiers]}")
        log(f"  vertex groups={len(mesh.vertex_groups)} matching deform groups={len(pair['weighted'])}")
        log(f"  materials={[slot.name for slot in mesh.material_slots]}")

        controls = [pb.name for pb in armature.pose.bones if "control" in pb.name.lower()]
        log(f"  bones={len(armature.data.bones)} controls={controls}")
        for bone in armature.data.bones:
            log(f"    BONE {bone.name!r} parent={getattr(bone.parent, 'name', None)!r} "
                f"deform={bone.use_deform} head={_fmt_vec(bone.head_local)} "
                f"tail={_fmt_vec(bone.tail_local)}")
        for pose_bone in armature.pose.bones:
            for constraint in pose_bone.constraints:
                log(f"    CONSTRAINT bone={pose_bone.name!r} type={constraint.type!r} "
                    f"target={getattr(getattr(constraint, 'target', None), 'name', None)!r} "
                    f"subtarget={getattr(constraint, 'subtarget', '')!r} "
                    f"owner_space={getattr(constraint, 'owner_space', None)!r} "
                    f"target_space={getattr(constraint, 'target_space', None)!r} "
                    f"influence={constraint.influence}")

        missing = [name for name in CONTROL_NAMES if name not in armature.pose.bones]
        if missing:
            raise RuntimeError(f"Armature {armature.name!r} is missing controls: {missing}")


def _probe_control_response(pair):
    armature = pair["armature"]
    section(f"Control response: {armature.name}")
    for control_name in CONTROL_NAMES:
        control = armature.pose.bones[control_name]
        original = control.matrix_basis.copy()
        for axis_index, axis_name in enumerate("XYZ"):
            for sign in (-1, 1):
                before = {pb.name: pb.tail.copy() for pb in armature.pose.bones
                          if pb.bone.use_deform}
                control.matrix_basis = original @ Matrix.Rotation(
                    math.radians(20.0 * sign), 4, axis_name)
                bpy.context.view_layer.update()
                moved = []
                for pose_bone in armature.pose.bones:
                    if not pose_bone.bone.use_deform:
                        continue
                    distance = (pose_bone.tail - before[pose_bone.name]).length
                    moved.append((distance, pose_bone.name))
                distance, bone_name = max(moved, default=(0.0, "<none>"))
                log(f"  {control_name:14} {axis_name}{sign:+d}: max tail delta="
                    f"{distance:.8f} on {bone_name}")
                control.matrix_basis = original
                bpy.context.view_layer.update()


def _copy_pair(pair, collection, suffix):
    source_mesh, source_armature = pair["mesh"], pair["armature"]
    mesh_world = source_mesh.matrix_world.copy()
    arm_world = source_armature.matrix_world.copy()

    armature = source_armature.copy()
    armature.data = source_armature.data.copy()
    armature.name = f"ProbeArm_{suffix}"
    collection.objects.link(armature)
    armature.parent = None
    armature.matrix_world = arm_world

    mesh = source_mesh.copy()
    mesh.name = f"ProbeHand_{suffix}"
    mesh.hide_render = False
    collection.objects.link(mesh)
    mesh.parent = armature
    mesh.matrix_world = mesh_world
    for modifier in mesh.modifiers:
        if modifier.type == "ARMATURE":
            modifier.object = armature
    for pose_bone in armature.pose.bones:
        for constraint in pose_bone.constraints:
            if getattr(constraint, "target", None) is source_armature:
                constraint.target = armature

    return {"mesh": mesh, "armature": armature,
            "weighted": set(pair["weighted"])}


def _solid_material(name, color, roughness=0.65):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return material


def _add_backdrop():
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, -0.14))
    plane = bpy.context.active_object
    plane.name = "T13_Backdrop"
    plane.data.materials.append(_solid_material("T13_BackdropMat", (0.075, 0.09, 0.11)))
    return plane


def _clear_cameras_and_lights():
    for obj in list(bpy.data.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def _render_view(name, azimuth, elevation, subject_h, target=(0.0, 0.0, 0.0)):
    _clear_cameras_and_lights()
    distance = sc.frame_distance(50, subject_h=subject_h, target_frac=0.82)
    camera = sc.setup_camera(50, azimuth, elevation, distance, target=target)
    import numpy as np
    sc.add_lights(camera.location, target=target,
                  rng=np.random.default_rng(SEED + int(azimuth * 10 + elevation)))
    bpy.context.view_layer.update()
    path = os.path.join(_ROOT, config.OUTPUT.root, f"t13_hand_asset_{name}.png")
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    log(f"Rendered {path}")


def _add_text(label, location, size=0.009):
    curve = bpy.data.curves.new(f"Text_{label}", type="FONT")
    curve.body = label
    curve.align_x = "CENTER"
    curve.size = size
    obj = bpy.data.objects.new(f"Text_{label}", curve)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    curve.materials.append(_solid_material(f"TextMat_{label}", (0.9, 0.92, 0.95), 0.5))
    return obj


def _build_control_grid(source_pair, collection, material):
    source_pair["mesh"].hide_render = True
    controls = [name for name in CONTROL_NAMES if name in source_pair["armature"].pose.bones]
    x_positions = [(-2.5 + i) * 0.115 for i in range(6)]
    y_positions = [(1.5 - i) * 0.12 for i in range(len(controls))]
    probes = []
    for row, control_name in enumerate(controls):
        for axis_index, axis_name in enumerate("XYZ"):
            for sign_index, sign in enumerate((-1, 1)):
                column = axis_index * 2 + sign_index
                suffix = f"{row}_{axis_name}_{'p' if sign > 0 else 'n'}"
                pair = _copy_pair(source_pair, collection, suffix)
                _limit_multires(pair["mesh"], 0)
                control = pair["armature"].pose.bones[control_name]
                control.matrix_basis = control.matrix_basis @ Matrix.Rotation(
                    math.radians(20.0 * sign), 4, axis_name)
                _assign_object_material(pair["mesh"], material)
                _insert_normalized_root(
                    pair, f"ProbeRoot_{suffix}",
                    (x_positions[column], y_positions[row], 0.0), 0.082,
                )
                _add_text(f"{control_name.split()[0]} {axis_name}{sign:+d}",
                          (x_positions[column], y_positions[row] - 0.050, 0.055))
                probes.append(pair)
    return probes


def run():
    import numpy as np
    np.random.default_rng(SEED)  # establish the deterministic diagnostic seed

    sc.reset_scene()
    sc.setup_world(gray=0.055)
    scene = bpy.context.scene
    setup_render(scene, verbose=True)
    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    os.makedirs(out_dir, exist_ok=True)

    path = config.hand_asset_path()
    inventory, objects, collection = _append_asset(path)
    bpy.context.view_layer.update()
    pairs = _find_pairs(objects)
    _log_inventory(path, inventory, pairs)

    # Record control effects while all legacy relationships are still untouched.
    for pair in pairs:
        _probe_control_response(pair)

    light = _skin_material("T13_SkinLight", SKIN_LIGHT, 0.01)
    dark = _skin_material("T13_SkinDark", SKIN_DARK, 0.03)
    for pair, target, material in zip(
            pairs, ((-0.15, 0.0, 0.0), (0.15, 0.0, 0.0)), (light, dark)):
        _limit_multires(pair["mesh"], 1)
        root, scale = _insert_normalized_root(
            pair, f"T13_Root_{pair['mesh'].name}", target, 0.22)
        _assign_object_material(pair["mesh"], material)
        low, high = _evaluated_bounds(pair["mesh"])
        log(f"Normalized {pair['mesh'].name}: root={root.name} scale={scale:.8f} "
            f"bounds={_fmt_vec(low)}..{_fmt_vec(high)}")

    card = cf.build_card_unit("T13_ScaleCard", "scale-reference")
    card.location = (0.0, 0.0, 0.0)
    _add_backdrop()

    _render_view("front", 0.0, 0.0, 0.55)
    _render_view("side", 90.0, 0.0, 0.55)
    _render_view("oblique", 35.0, 24.0, 0.58)

    # Calibration grid uses independent armature copies and the left/light hand.
    card.hide_render = True
    pairs[1]["mesh"].hide_render = True
    grid_material = _skin_material("T13_SkinGrid", (0.46, 0.235, 0.13), 0.02)
    _build_control_grid(pairs[0], collection, grid_material)
    _render_view("controls", 0.0, 0.0, 0.72)

    section("Acceptance summary")
    log("PASS requires two intact real-scale hands and four visibly responsive control rows.")
    log("The source asset was appended locally and was not saved or modified.")


def main():
    error = None
    try:
        run()
    except Exception as exc:  # noqa: BLE001 - preserve a report for Blender-only failures
        error = exc
        section("FATAL ERROR")
        log(f"{type(exc).__name__}: {exc}")
        log(traceback.format_exc())
    finally:
        _write_report()
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
