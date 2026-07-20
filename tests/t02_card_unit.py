"""
Phase 2 (step 1) - Base card UNIT: thickness + front/back/edge materials.

WHY THIS EXISTS
    Validates the card_factory base unit before protection layers are added:
      - real 0.45mm thickness with a mid-grey edge,
      - front (+Z) shows the card image, back (-Z) shows back.png (fixes the
        Phase-1 flipped-front-on-back issue),
      - labeling still lands on the sharp ideal corners now that corners sit on
        the front face (z=+T/2),
      - a back-facing card shows back.png and is NOT labeled.

HOW TO RUN (headless), from inside the tcgDataSynth folder:
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t02_card_unit.py

WHAT IT PRODUCES (out/)
    t02_front_headon.png/.txt   - front, head-on            (EXPECT 1 label)
    t02_front_tilt.png/.txt     - tilted, edge visible      (EXPECT 1 label)
    t02_back_view.png/.txt      - back.png facing camera    (EXPECT 0 labels)
    Console: per-case corner NDC + label decision, and a datablock-sharing check.

WHAT TO REPORT BACK
    1) Paste console output.
    2) Run the visualizer and attach the results:
         python3 labeltools/visualize.py out/t02_front_headon.png out/t02_front_headon.txt
         python3 labeltools/visualize.py out/t02_front_tilt.png   out/t02_front_tilt.txt
         python3 labeltools/visualize.py out/t02_back_view.png    out/t02_back_view.txt
    PASS if: front cases show the card image with corners on the sharp corners and
    a thin grey edge where visible; back_view shows back.png and "labels: 0".
"""
import os
import sys
import math
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)  # quiet use_nodes notice

import bpy
from mathutils import Vector

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from blender import card_factory as cf  # noqa: E402
from blender.labeling import label_card  # noqa: E402
from labeltools.yolo_pose import write_label_file, write_dataset_yaml  # noqa: E402

try:
    from texturegen.cardsource import CardLibrary
except Exception:  # noqa: BLE001
    CardLibrary = None

H = config.CARD_H_M


# --------------------------------------------------------------------------- #
# Scene setup (compact; mirrors t01)
# --------------------------------------------------------------------------- #
def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def setup_world(gray=0.18):
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (gray, gray, gray, 1.0)
    bg.inputs["Strength"].default_value = 1.0


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = config.RENDER_ENGINE
    scene.render.resolution_x = config.RENDER_W
    scene.render.resolution_y = config.RENDER_H
    scene.render.resolution_percentage = 100
    try:
        scene.eevee.taa_render_samples = config.EEVEE_RENDER_SAMPLES
    except Exception as exc:  # noqa: BLE001
        print(f"[t02] eevee samples: {exc}")
    try:
        scene.view_settings.view_transform = config.VIEW_TRANSFORM
    except Exception as exc:  # noqa: BLE001
        print(f"[t02] view_transform: {exc}")
    scene.render.image_settings.file_format = "PNG"


def frame_distance(focal_mm, target_frac=0.80):
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
    sun.rotation_euler = (math.radians(45), 0.0, math.radians(30))
    bpy.context.collection.objects.link(sun)
    pt_data = bpy.data.lights.new("Flash", type="POINT")
    pt_data.energy = 6.0
    pt = bpy.data.objects.new("Flash", pt_data)
    pt.location = cam_loc
    bpy.context.collection.objects.link(pt)


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
# name, focal, azimuth, elevation, card_rotation_euler
CASES = [
    ("front_headon", 50, 0,  0,  (0, 0, 0)),
    ("front_tilt",   50, 32, 22, (0, 0, 0)),          # reveals thickness/grey edge
    ("back_view",    50, 0,  0,  (0, math.pi, 0)),    # back.png toward cam -> 0 labels
]


def run_case(name, focal, azimuth, elevation, rot, front_path, back_path):
    reset_scene()
    setup_world()
    setup_render()

    card_id = os.path.splitext(os.path.basename(front_path))[0] if front_path else "testcard"
    card = cf.build_card_unit("Card", card_id,
                              front_image_path=front_path, back_image_path=back_path)
    card.rotation_euler = rot

    dist = frame_distance(focal)
    cam = setup_camera(focal, azimuth, elevation, dist)
    add_lights(cam.location)

    scene = bpy.context.scene
    bpy.context.view_layer.update()
    label, reason, dbg = label_card(scene, cam, card, card_id)

    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    os.makedirs(out_dir, exist_ok=True)
    scene.render.filepath = os.path.join(out_dir, f"t02_{name}.png")
    bpy.ops.render.render(write_still=True)
    write_label_file(os.path.join(out_dir, f"t02_{name}.txt"), [label] if label else [])

    print(f"\n=== case {name}  focal={focal} az={azimuth} el={elevation} dist={dist:.3f}m -> {reason} ===")
    for (idx, x, y, z, inf) in dbg:
        print(f"   corner{idx}  ndc=({x:+.4f}, {y:+.4f}, z={z:+.4f})  in_frustum={inf}")
    print(f"   {'LABEL: ' + label.to_line() if label else 'NO LABEL (' + reason + ')'}")


def sharing_check(front_a, front_b, back_path):
    """Prove the datablock-sharing path: two card objects, ONE shared mesh, but
    distinct per-instance front materials."""
    reset_scene()
    mesh = cf.build_card_mesh("Shared")
    a = cf.build_card_unit("CardA", "idA", front_image_path=front_a,
                           back_image_path=back_path, shared_mesh=mesh)
    b = cf.build_card_unit("CardB", "idB", front_image_path=front_b,
                           back_image_path=back_path, shared_mesh=mesh)
    same_mesh = a.data is b.data is mesh
    distinct_front = a.material_slots[0].material is not b.material_slots[0].material
    obj_linked = a.material_slots[0].link == "OBJECT"
    print(f"\n[sharing check] same_mesh={same_mesh}  distinct_front_mat={distinct_front}  "
          f"front_slot_link_OBJECT={obj_linked}  (all should be True)")


def main():
    front_path = None
    front_b = None
    back_path = config.back_image_path()
    if CardLibrary is not None:
        try:
            lib = CardLibrary()
            if not lib.is_empty():
                import numpy as np
                front_path = lib.select(np.random.default_rng(7)).path
                front_b = lib.select(np.random.default_rng(11)).path
                print(f"[t02] front card: {front_path}")
                print(f"[t02] back.png:   {back_path or '<not found>'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[t02] card library unavailable: {exc}")
    if not front_path:
        print("[t02] no card image found; front will use grey fallback")
    if not back_path:
        print("[t02] WARNING: back.png not found at image root; back uses grey fallback")

    for (name, focal, az, el, rot) in CASES:
        run_case(name, focal, az, el, rot, front_path, back_path)

    sharing_check(front_path, front_b or front_path, back_path)
    write_dataset_yaml(os.path.join(_ROOT, config.OUTPUT.root, "dataset.yaml"))
    print("\n[t02] done. Expect: front_headon & front_tilt labeled; back_view = 0 labels.")


if __name__ == "__main__":
    main()
