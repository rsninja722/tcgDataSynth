"""
Phase 2 diagnostic - locate the "black corner" on the sleeve.

WHY THIS EXISTS
    A dark spot appears at the bottom-right of every sleeve. This test renders ONE
    clear sleeve under controlled variations so we can tell what the spot tracks:

      - headon            : camera exactly head-on (bottom corners viewed
                            symmetrically). If only ONE bottom corner is dark ->
                            geometry is asymmetric. If BOTH or NEITHER -> not geometry.
      - tilt_default      : the original problem view (az=25, el=18).
      - rotz_90 / rotz_180: card+sleeve spun in-plane, camera fixed. If the dark spot
                            MOVES to a different image corner -> it's attached to the
                            GEOMETRY. If it stays bottom-right -> it's VIEW/screen-based.
      - cam_left/cam_right: camera moved, card fixed. Does the spot follow the camera?
      - no_warp           : warp normal map DISABLED. If the spot vanishes -> it's the
                            normal map at grazing angles. If it persists -> it's not.
      - sleeve_only       : card hidden. If the spot persists on the bare sleeve ->
                            it's the sleeve glass itself, not the card/edge showing through.

HOW TO RUN (headless):
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t03d_sleeve_diagnose.py

WHAT TO REPORT BACK
    Attach the 8 out/t03d_*.png. Tell me, for each, WHERE the dark spot is (which
    image corner) or if it's absent. That pins the cause and I'll fix it directly.
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
ASSETS = os.path.join(_ROOT, "assets")


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


def setup_camera(focal_mm, azimuth_deg, elevation_deg, distance):
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = "PERSP"
    cam_data.lens = focal_mm
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.001
    cam_data.clip_end = 100.0
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    loc = Vector((math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el))) * distance
    cam = bpy.data.objects.new("Cam", cam_data)
    cam.location = loc
    cam.rotation_euler = (Vector((0, 0, 0)) - loc).normalized().to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def add_lights(cam_loc):
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 2.0
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(25))
    bpy.context.collection.objects.link(sun)
    pt_data = bpy.data.lights.new("Flash", type="POINT")
    pt_data.energy = 7.0
    pt = bpy.data.objects.new("Flash", pt_data)
    pt.location = cam_loc
    bpy.context.collection.objects.link(pt)


# name, cam_az, cam_el, card_rot_z_deg, use_warp, show_card
CASES = [
    ("headon",       0,   0,   0,   True,  True),
    ("tilt_default", 25,  18,  0,   True,  True),
    ("rotz_90",      25,  18,  90,  True,  True),
    ("rotz_180",     25,  18,  180, True,  True),
    ("cam_left",     -35, 12,  0,   True,  True),
    ("cam_right",    35,  12,  0,   True,  True),
    ("no_warp",      25,  18,  0,   False, True),
    ("sleeve_only",  25,  18,  0,   True,  False),
]


def run_case(name, az, el, rotz, use_warp, show_card, front_path, back_path):
    reset_scene()
    setup_world()
    setup_render(bpy.context.scene, verbose=False)

    card = cf.build_card_unit("Card", "diag", front_image_path=front_path,
                              back_image_path=back_path)
    card.rotation_euler = (0, 0, math.radians(rotz))
    warp = os.path.join(ASSETS, "plastic_warp_0.png") if use_warp else None
    prot.build_sleeve("Sleeve", card, "clear", "2.5mm", warp)
    if not show_card:
        card.hide_render = True

    dist = frame_distance(50)
    cam = setup_camera(50, az, el, dist)
    add_lights(cam.location)

    scene = bpy.context.scene
    bpy.context.view_layer.update()
    scene.render.filepath = os.path.join(_ROOT, config.OUTPUT.root, f"t03d_{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"  rendered t03d_{name}  (cam az={az} el={el}, card rotZ={rotz}, "
          f"warp={use_warp}, card_visible={show_card})")


def main():
    front_path = None
    back_path = config.back_image_path()
    if CardLibrary is not None:
        try:
            lib = CardLibrary()
            if not lib.is_empty():
                import numpy as np
                front_path = lib.select(np.random.default_rng(3)).path
        except Exception as exc:  # noqa: BLE001
            print(f"[t03d] card library: {exc}")
    print(f"[t03d] front={front_path}")
    for (name, az, el, rotz, warp, show) in CASES:
        run_case(name, az, el, rotz, warp, show, front_path, back_path)
    print("\n[t03d] done. Report where the dark spot is (or absent) in each image.")


if __name__ == "__main__":
    main()
