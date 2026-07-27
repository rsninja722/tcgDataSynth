r"""Phase 4 final checkpoint - bundled compact hand-rig validation.

Run after ``blender/hand_assets_build.py``:

    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tests\t15_hand_asset_bundle.py

The test verifies that ``assets/hand_rig.blend`` is self-contained, contains only the
four required hand objects and their dependencies, and retains all runtime rig data.
It writes ``out/t15_hand_asset_bundle_report.txt``. Then rerun ``tests/t14_hand.py``
without ``TCG_HAND_ASSET`` to confirm render equivalence.
"""
from __future__ import annotations

import os
import sys
import traceback

import bpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from blender.hand_assets_build import (EXPECTED_OBJECTS, FORBIDDEN_DATABLOCKS,  # noqa: E402
                                       validate_hand_objects)

_LINES = []


def _log(message=""):
    print(message)
    _LINES.append(str(message))


def _write_report():
    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "t15_hand_asset_bundle_report.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(_LINES) + "\n")
    print(f"[t15] report written: {path}")


def run():
    if tuple(bpy.app.version) != (5, 0, 0):
        raise RuntimeError(
            f"t15 requires Blender 5.0.0, got {bpy.app.version_string}")
    path = config.DEFAULT_HAND_ASSET_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Bundled hand asset not found: {path!r}; run blender/hand_assets_build.py")

    with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
        inventory = {
            kind: list(getattr(data_from, kind, ()))
            for kind in ("objects", "meshes", "armatures", *FORBIDDEN_DATABLOCKS)
        }
        if set(data_from.objects) != set(EXPECTED_OBJECTS):
            raise RuntimeError(f"Unexpected object inventory: {data_from.objects}")
        for kind in FORBIDDEN_DATABLOCKS:
            if inventory[kind]:
                raise RuntimeError(f"Bundled hand asset unexpectedly contains {kind}")
        data_to.objects = list(EXPECTED_OBJECTS)

    objects = [obj for obj in data_to.objects if obj is not None]
    validate_hand_objects(objects)
    _log(f"Blender version: {bpy.app.version_string}")
    _log(f"Bundled path: {path}")
    _log(f"Bundled size: {os.path.getsize(path)} bytes")
    for kind, names in inventory.items():
        _log(f"{kind}: {names}")
    for obj in sorted(objects, key=lambda item: item.name):
        if obj.type == "MESH":
            modifiers = [(modifier.name, modifier.type,
                          getattr(modifier, "levels", None),
                          getattr(modifier, "render_levels", None))
                         for modifier in obj.modifiers]
            _log(f"mesh={obj.name!r} vertices={len(obj.data.vertices)} "
                 f"polygons={len(obj.data.polygons)} modifiers={modifiers} "
                 f"materials={len(obj.data.materials)}")
        elif obj.type == "ARMATURE":
            _log(f"armature={obj.name!r} bones={len(obj.data.bones)}")
    _log("PASS: compact hand library is self-contained and runtime-compatible.")


def main():
    error = None
    try:
        run()
    except Exception as exc:  # noqa: BLE001 - preserve Blender-only failure report
        error = exc
        _log(f"FATAL: {type(exc).__name__}: {exc}")
        _log(traceback.format_exc())
    finally:
        _write_report()
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
