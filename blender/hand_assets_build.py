"""Build the permanent compact hand-rig library (Blender 5 / bpy REQUIRED).

The supplied source contains cameras, lighting, embedded images, text, and legacy
materials that generation never uses. This script appends only the two hand meshes and
their armatures, strips the replaced materials, and writes their dependency closure to
``assets/hand_rig.blend``. Multires data is deliberately preserved because applying or
collapsing it could change the already accepted hand geometry.

Run from the project root:

    blender -b -P blender/hand_assets_build.py -- --source "C:\\path\\to\\Hands + armature.blend"
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import bpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from blender.hand import HAND_OBJECTS, HAND_REQUIRED_BONES  # noqa: E402

DEFAULT_OUTPUT = os.path.join(_ROOT, "assets", "hand_rig.blend")
EXPECTED_OBJECTS = tuple(name for pair in HAND_OBJECTS.values() for name in pair)
EXPECTED_SOURCE_SHA256 = "6cca25beb3f48460f977f1e47f76612eb87900b64b71cc8b4627d52e54561f40"
FORBIDDEN_DATABLOCKS = (
    "materials", "images", "texts", "actions", "collections", "scenes", "cameras", "lights",
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Supplied source .blend path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Destination library (default: assets/hand_rig.blend)")
    return parser.parse_args(args)


def _same_file_or_path(first: str, second: str) -> bool:
    first_real = os.path.normcase(os.path.realpath(first))
    second_real = os.path.normcase(os.path.realpath(second))
    if first_real == second_real:
        return True
    if os.path.exists(first) and os.path.exists(second):
        return os.path.samefile(first, second)
    return False


def _append_required_objects(source: str):
    with bpy.data.libraries.load(source, link=False) as (data_from, data_to):
        missing = [name for name in EXPECTED_OBJECTS if name not in data_from.objects]
        if missing:
            raise RuntimeError(f"Source hand asset is missing objects: {missing}")
        data_to.objects = list(EXPECTED_OBJECTS)
    objects = [obj for obj in data_to.objects if obj is not None]
    if len(objects) != len(EXPECTED_OBJECTS):
        raise RuntimeError("Not all required hand objects were appended")
    return objects


def validate_hand_objects(objects) -> None:
    """Validate relationships and runtime-required rig data for both hand pairs."""
    by_name = {obj.name: obj for obj in objects}
    missing = [name for name in EXPECTED_OBJECTS if name not in by_name]
    if missing:
        raise RuntimeError(f"Hand library is missing objects: {missing}")

    for handedness, (mesh_name, armature_name) in HAND_OBJECTS.items():
        mesh = by_name[mesh_name]
        armature = by_name[armature_name]
        if mesh.type != "MESH" or armature.type != "ARMATURE":
            raise RuntimeError(f"{handedness} hand does not resolve to mesh/armature")
        modifiers = [modifier for modifier in mesh.modifiers
                     if modifier.type == "ARMATURE"]
        if len(modifiers) != 1 or modifiers[0].object is not armature:
            raise RuntimeError(f"{mesh_name!r} has an invalid Armature modifier")
        deform = {bone.name for bone in armature.data.bones if bone.use_deform}
        weighted = {group.name for group in mesh.vertex_groups}
        missing_groups = sorted(deform - weighted)
        if missing_groups:
            raise RuntimeError(f"{mesh_name!r} is missing deform groups: {missing_groups}")
        missing_bones = sorted(HAND_REQUIRED_BONES - set(armature.pose.bones.keys()))
        if missing_bones:
            raise RuntimeError(f"{armature_name!r} is missing bones: {missing_bones}")
        multires = [modifier for modifier in mesh.modifiers if modifier.type == "MULTIRES"]
        if len(multires) != 1 or int(getattr(multires[0], "total_levels", 0)) < 1:
            raise RuntimeError(f"{mesh_name!r} did not retain its Multires data")
        control_constraints = {control: 0 for control in HAND_REQUIRED_BONES
                               if "control" in control.lower()}
        for pose_bone in armature.pose.bones:
            for constraint in pose_bone.constraints:
                subtarget = getattr(constraint, "subtarget", "")
                if (constraint.type == "COPY_ROTATION"
                        and getattr(constraint, "target", None) is armature
                        and subtarget in control_constraints):
                    control_constraints[subtarget] += 1
        invalid_controls = {name: count for name, count in control_constraints.items()
                            if count < 3}
        if invalid_controls:
            raise RuntimeError(
                f"{armature_name!r} lost finger-control constraints: {invalid_controls}")


def build_library(source: str, output: str) -> None:
    if tuple(bpy.app.version) != (5, 0, 0):
        raise RuntimeError(
            f"Hand library must be built with Blender 5.0.0, got {bpy.app.version_string}")
    source = os.path.abspath(source)
    output = os.path.abspath(output)
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Source hand asset not found: {source!r}")
    if _same_file_or_path(source, output):
        raise ValueError("Source and output hand asset paths must differ")

    source_size = os.path.getsize(source)
    source_hash = _sha256(source)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"Unexpected source hand asset checksum: {source_hash}; "
            f"expected {EXPECTED_SOURCE_SHA256}")
    objects = _append_required_objects(source)
    validate_hand_objects(objects)

    for obj in objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            for modifier in obj.modifiers:
                if modifier.type == "MULTIRES":
                    total = int(getattr(modifier, "total_levels", 1))
                    modifier.levels = min(1, total)
                    modifier.render_levels = min(1, total)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp.blend"
    if _same_file_or_path(source, temporary):
        raise ValueError("Temporary output path aliases the source hand asset")
    if os.path.isfile(temporary):
        os.remove(temporary)
    bpy.data.libraries.write(temporary, set(objects), fake_user=True, compress=True)
    with bpy.data.libraries.load(temporary, link=False) as (data_from, _data_to):
        written = set(data_from.objects)
        extras = sorted(written - set(EXPECTED_OBJECTS))
        missing = sorted(set(EXPECTED_OBJECTS) - written)
        if missing or extras:
            raise RuntimeError(
                f"Compact library inventory mismatch: missing={missing}, extras={extras}")
        retained = {kind: list(getattr(data_from, kind, ()))
                    for kind in FORBIDDEN_DATABLOCKS
                    if getattr(data_from, kind, ())}
        if retained:
            raise RuntimeError(f"Compact library retained unrelated datablocks: {retained}")

    if os.path.getsize(source) != source_size or _sha256(source) != source_hash:
        raise RuntimeError("Source hand asset changed while building the compact library")
    os.replace(temporary, output)
    if os.path.getsize(source) != source_size or _sha256(source) != source_hash:
        raise RuntimeError("Source hand asset changed while publishing the compact library")
    print(f"[hand-assets] source sha256: {source_hash}")
    print(f"[hand-assets] source bytes: {source_size}")
    print(f"[hand-assets] bundled bytes: {os.path.getsize(output)}")
    print(f"[hand-assets] wrote: {output}")


def main() -> None:
    args = _parse_args()
    build_library(args.source, args.output)


if __name__ == "__main__":
    main()
