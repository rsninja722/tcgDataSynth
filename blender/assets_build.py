"""
assets_build.py - build the card-independent PROTECTION meshes once into a library
.blend so generation links them instead of rebuilding heavy geometry per card
(esp. the fine sleeve mesh). Materials are NOT baked in here: they are per-instance
(applied at generation time via blender/protection.py's make_* functions), because
tint/wear/warp-crop vary per instance.

Builds, as objects in the file, the shared meshes:
  asset_sleeve_1mm, asset_sleeve_2p5mm, asset_toploader, asset_semirigid, asset_slab

HOW TO RUN (headless), from inside the tcgDataSynth folder:
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P blender/assets_build.py

OUTPUT
    assets/protection_lib.blend   (+ console summary of what was written)

Generation-time usage (Phase 4):
    from blender.asset_lib import load_protection_mesh
    mesh = load_protection_mesh("asset_toploader")   # links the shared datablock
    obj = bpy.data.objects.new("Toploader", mesh)
    # then apply per-instance materials with protection.make_* + assign to slots
"""
import os
import sys

import bpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from blender import protection as prot  # noqa: E402

LIB_PATH = os.path.join(_ROOT, "assets", "protection_lib.blend")


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.objects):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def build_protection_meshes():
    """Return {asset_name: mesh} for every card-independent protection mesh."""
    return {
        "asset_sleeve_1mm": prot.build_sleeve_mesh("1mm", name="asset_sleeve_1mm"),
        "asset_sleeve_2p5mm": prot.build_sleeve_mesh("2.5mm", name="asset_sleeve_2p5mm"),
        "asset_toploader": prot._flat_pocket_mesh(
            "asset_toploader", prot.TOPLOADER_W, prot.TOPLOADER_H,
            prot.TOPLOADER_GAP, 0.0, spine_width=prot._SPINE_W),
        "asset_semirigid": prot._flat_pocket_mesh(
            "asset_semirigid", prot.SEMIRIGID_W, prot.SEMIRIGID_H,
            prot.SEMIRIGID_GAP, prot.SEMIRIGID_LIP, spine_width=0.0),
        "asset_slab": prot._slab_mesh("asset_slab"),
    }


def main():
    _clear_scene()
    meshes = build_protection_meshes()
    # Lay the assets out in a row as visible objects (so the .blend is inspectable
    # and appendable object-by-object).
    for i, (name, mesh) in enumerate(sorted(meshes.items())):
        obj = bpy.data.objects.new(name, mesh)
        obj.location = (i * 0.12, 0.0, 0.0)
        bpy.context.collection.objects.link(obj)
        print(f"  built {name}: {len(mesh.vertices)} verts, {len(mesh.polygons)} faces, "
              f"{len(mesh.materials)} slots")

    os.makedirs(os.path.dirname(LIB_PATH), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=LIB_PATH)
    print(f"\n[assets_build] wrote {LIB_PATH} with {len(meshes)} protection assets.")
    print("[assets_build] Phase 2 asset library complete.")


if __name__ == "__main__":
    main()
