"""
Phase 2 diagnostic 2 - isolate GLASS vs NORMAL-MAP as the black-spot cause, and
test a candidate fix (alpha transparency instead of refractive glass).

Four quick renders (clear 2.5mm sleeve):
  1) sleeveonly_glass_noWarp : refractive glass, NO normal map, card hidden.
        black persists -> it's the GLASS geometry (TIR at curved seams), not the map.
        black gone      -> the normal map was involved.
  2) rotz90_glass_noWarp     : same glass+noWarp, card+sleeve spun 90, card visible.
  3) sleeveonly_matte_warp   : NON-refractive matte material WITH the normal map.
        black here      -> the normal-map data itself is bad.
        clean here      -> the map is fine; the glass was the problem.
  4) sleeveonly_alpha_warp   : CANDIDATE FIX - thin plastic as straight-through alpha
        transparency (no refraction => no TIR) + glossy reflections + normal map.
        clean here      -> adopt this material model for all sleeve/plastic.

HOW TO RUN:
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t03e_material_isolate.py

REPORT BACK: attach the 4 out/t03e_*.png and say which are clean vs black-spotted.
"""
import os
import sys
import math
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import bpy
from mathutils import Vector

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from blender import card_factory as cf  # noqa: E402
from blender import protection as prot  # noqa: E402
from blender.render_setup import setup_render  # noqa: E402

try:
    from texturegen.cardsource import CardLibrary
except Exception:  # noqa: BLE001
    CardLibrary = None

H = config.CARD_H_M
WARP = os.path.join(_ROOT, "assets", "plastic_warp_0.png")


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights,
                 bpy.data.cameras, bpy.data.images):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def setup_world(gray=0.25):
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (gray, gray, gray, 1.0)
    bg.inputs["Strength"].default_value = 1.0


def frame_distance(focal_mm, target_frac=0.72):
    fov = 2.0 * math.atan((36.0 / 2.0) / focal_mm)
    return (H / target_frac) / (2.0 * math.tan(fov / 2.0))


def setup_camera(focal_mm, az_deg, el_deg, distance):
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = focal_mm
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.001
    cam_data.clip_end = 100.0
    az, el = math.radians(az_deg), math.radians(el_deg)
    loc = Vector((math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el))) * distance
    cam = bpy.data.objects.new("Cam", cam_data)
    cam.location = loc
    cam.rotation_euler = (Vector((0, 0, 0)) - loc).normalized().to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def add_lights(cam_loc):
    sd = bpy.data.lights.new("Sun", type="SUN")
    sd.energy = 2.0
    s = bpy.data.objects.new("Sun", sd)
    s.rotation_euler = (math.radians(50), 0.0, math.radians(25))
    bpy.context.collection.objects.link(s)
    pd = bpy.data.lights.new("Flash", type="POINT")
    pd.energy = 7.0
    p = bpy.data.objects.new("Flash", pd)
    p.location = cam_loc
    bpy.context.collection.objects.link(p)


def make_matte_warp(name, warp):
    """Non-refractive matte material carrying the warp normal map (isolates the map)."""
    return prot.make_opaque_plastic(name, (0.6, 0.6, 0.6), warp, roughness=0.4)


def make_alpha_plastic(name, warp, alpha=0.12, roughness=0.06):
    """Candidate fix: straight-through alpha transparency (no refraction => no TIR),
    glossy reflections from low roughness, uneven via the warp normal map."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Transmission Weight"].default_value = 0.0
    bsdf.inputs["Alpha"].default_value = alpha
    prot._add_warp_normal(nt, bsdf, warp)
    return mat


def override_materials(obj, mat):
    for slot in obj.material_slots:
        slot.link = "OBJECT"
        slot.material = mat


def run_case(name, rotz, show_card, warp_for_sleeve, override_mat_fn, front, back):
    reset_scene()
    setup_world()
    setup_render(bpy.context.scene, verbose=False)

    card = cf.build_card_unit("Card", "diag", front_image_path=front, back_image_path=back)
    card.rotation_euler = (0, 0, math.radians(rotz))
    sleeve = prot.build_sleeve("Sleeve", card, "clear", "2.5mm", warp_for_sleeve)
    if override_mat_fn is not None:
        override_materials(sleeve, override_mat_fn(name + "_mat", WARP))
    if not show_card:
        card.hide_render = True

    cam = setup_camera(50, 25, 18, frame_distance(50))
    add_lights(cam.location)
    bpy.context.view_layer.update()
    bpy.context.scene.render.filepath = os.path.join(_ROOT, config.OUTPUT.root, f"t03e_{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"  rendered t03e_{name}")


def main():
    front = None
    back = config.back_image_path()
    if CardLibrary is not None:
        try:
            lib = CardLibrary()
            if not lib.is_empty():
                import numpy as np
                front = lib.select(np.random.default_rng(3)).path
        except Exception as exc:  # noqa: BLE001
            print(f"[t03e] card lib: {exc}")

    # name, rotz, show_card, warp_for_sleeve(base build), override_material_fn
    run_case("sleeveonly_glass_noWarp", 0, False, None, None, front, back)
    run_case("rotz90_glass_noWarp", 90, True, None, None, front, back)
    run_case("sleeveonly_matte_warp", 0, False, None, make_matte_warp, front, back)
    run_case("sleeveonly_alpha_warp", 0, False, None, make_alpha_plastic, front, back)
    print("\n[t03e] done. Report which of the 4 are clean vs black-spotted.")


if __name__ == "__main__":
    main()
