"""
Phase 2 (step 2) - SLEEVES: clear + opaque-back, both sizes, plastic-warp normals.

WHY THIS EXISTS
    Validates sleeve geometry (two layers 0.05mm off each card face, curving to
    meet on left/right/bottom, open at top, extended +1mm / +2.5mm on all sides),
    plastic transparency, and the uneven warp reflections. Also probes the EEVEE
    transparency/raytracing API so the plastic shader can be tuned from ground
    truth rather than assumptions.

HOW TO RUN (headless):
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t03_sleeve.py

WHAT IT PRODUCES (out/)
    t03_clear_1mm.png/.txt,  t03_clear_2p5mm.png/.txt,
    t03_opaque_1mm.png/.txt, t03_opaque_2p5mm.png/.txt
    Console: an EEVEE/material API probe + per-case label decision.

WHAT TO REPORT BACK
    1) Paste console output (ESPECIALLY the API PROBE block).
    2) Attach the 4 PNGs (and optionally run the visualizer to confirm the card is
       still labeled through the sleeve).
    PASS if: the card sits inside a slightly-larger plastic sleeve; clear sleeves
    show the card through them with uneven (not mirror-flat) reflections; the
    opaque-back sleeve shows a colored back; the sleeve is open at the top; and the
    card remains labeled in every case.
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
from blender.labeling import label_card  # noqa: E402
from blender.render_setup import setup_render  # noqa: E402
from labeltools.yolo_pose import write_label_file  # noqa: E402

try:
    from texturegen.cardsource import CardLibrary
except Exception:  # noqa: BLE001
    CardLibrary = None

H = config.CARD_H_M
ASSETS = os.path.join(_ROOT, "assets")


# --------------------------------------------------------------------------- #
# Scene setup
# --------------------------------------------------------------------------- #
def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights,
                 bpy.data.cameras, bpy.data.images):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def setup_world(gray=0.2):
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


def probe_transparency_api():
    """Print the real EEVEE/material transparency & raytracing API surface."""
    print("\n----- API PROBE: EEVEE raytracing / material transparency -----")
    scene = bpy.context.scene
    ev = scene.eevee
    for p in ev.bl_rna.properties:
        pid = p.identifier
        if any(k in pid for k in ("ray", "horizon", "refr", "transm", "clamp", "fast_gi")):
            print(f"  scene.eevee.{pid:32} = {getattr(ev, pid, '?')!r}")
    rto = getattr(ev, "ray_tracing_options", None)
    if rto is not None:
        print("  --- scene.eevee.ray_tracing_options ---")
        for p in rto.bl_rna.properties:
            if p.identifier == "rna_type":
                continue
            print(f"  ray_tracing_options.{p.identifier:26} = {getattr(rto, p.identifier, '?')!r}")
    mat = bpy.data.materials.new("probe_mat")
    print("  --- Material properties (transparency-related) ---")
    for p in mat.bl_rna.properties:
        pid = p.identifier
        if any(k in pid for k in ("blend", "method", "transp", "shadow", "ray",
                                  "refr", "render", "displace")):
            val = getattr(mat, pid, "?")
            opts = ""
            if p.type == "ENUM":
                opts = " opts=" + str([e.identifier for e in p.enum_items])
            print(f"  material.{pid:28} = {val!r}{opts}")
    bpy.data.materials.remove(mat)
    print("----- END API PROBE -----\n")


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
CASES = [
    # name, sleeve_type, size, front_finish
    ("clear_1mm", "clear", "1mm", "clear"),
    ("clear_2p5mm", "clear", "2.5mm", "clear"),
    ("opaque_clearfront_1mm", "opaque_back", "1mm", "clear"),
    ("opaque_mattefront_2p5mm", "opaque_back", "2.5mm", "matte"),
]


def run_case(i, name, sleeve_type, size, front_finish, front_path, back_path):
    reset_scene()
    setup_world()
    scene = bpy.context.scene
    setup_render(scene, verbose=(i == 0))

    card_id = os.path.splitext(os.path.basename(front_path))[0] if front_path else "testcard"
    card = cf.build_card_unit("Card", card_id, front_image_path=front_path,
                              back_image_path=back_path)
    warp = os.path.join(ASSETS, f"plastic_warp_{i % 3}.png")
    prot.build_sleeve("Sleeve", card, sleeve_type, size, warp,
                      back_color=(0.06, 0.12, 0.45), front_finish=front_finish)

    dist = frame_distance(50)
    cam = setup_camera(50, 25, 18, dist)
    add_lights(cam.location)

    bpy.context.view_layer.update()
    label, reason, dbg = label_card(scene, cam, card, card_id)

    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    os.makedirs(out_dir, exist_ok=True)
    scene.render.filepath = os.path.join(out_dir, f"t03_{name}.png")
    bpy.ops.render.render(write_still=True)
    write_label_file(os.path.join(out_dir, f"t03_{name}.txt"), [label] if label else [])
    print(f"=== case {name} ({sleeve_type}, {size}) warp=plastic_warp_{i%3} -> {reason} ===")
    print(f"   {'LABEL: ' + label.to_line() if label else 'NO LABEL (' + reason + ')'}")


def main():
    front_path = None
    back_path = config.back_image_path()
    if CardLibrary is not None:
        try:
            lib = CardLibrary()
            if not lib.is_empty():
                import numpy as np
                front_path = lib.select(np.random.default_rng(3)).path
                print(f"[t03] front card: {front_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[t03] card library: {exc}")
    if not front_path:
        print("[t03] no card image; front uses grey fallback")

    # One-time API probe before the renders (informational; Cycles is now primary).
    reset_scene()
    setup_render(bpy.context.scene, verbose=True)
    probe_transparency_api()

    for i, (name, stype, size, front_finish) in enumerate(CASES):
        run_case(i, name, stype, size, front_finish, front_path, back_path)

    print("\n[t03] done. Expect all 4 cases labeled; eyeball plastic geometry/reflections.")


if __name__ == "__main__":
    main()
