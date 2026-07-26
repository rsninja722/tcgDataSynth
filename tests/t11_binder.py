"""
Phase 4 (layout 3/5) - BINDER (spec §3.5.1). Renders 4 binder scenes covering every
grid size (1x1/2x2/3x3/4x3), both page types (clear/solid) and all content types
(sleeved/toploader/slab). Warped, glossy pages; to exercise their mirror-like
reflections, colorful prisms/cylinders AND the non-sun lights are placed BEHIND the
camera so they show up only as reflections. Every filled, front-facing, in-frame card
is labeled with its holo tag.

HOW TO RUN (headless; cv2 and shapely are required in Blender's Python):
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t11_binder.py

OUTPUT (out/): t11_binder_<grid>_<page>_<content>.png + .txt (x4). Verify e.g.:
    python3 labeltools/visualize.py out/t11_binder_3x3_clear_sleeved.png out/t11_binder_3x3_clear_sleeved.txt

PASS if: each binder shows a grid of cards (some empty slots) on a colored board+spine,
warped glossy pages reflecting the behind-camera objects, and every card is labeled.
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
from rules import combinations as C  # noqa: E402
from rules.combinations import (SceneConfig, LayoutConfig, CardConfig, ProtectionConfig,  # noqa: E402
                                FinishConfig, DamageConfig, SleeveConfig, validate_scene_config)
from blender import layouts  # noqa: E402
from blender.labeling import label_scene  # noqa: E402
from blender.render_setup import setup_render  # noqa: E402
from blender import scene_common as sc  # noqa: E402
from labeltools.yolo_pose import write_poly_label_file  # noqa: E402

try:
    from texturegen.cardsource import CardLibrary
except Exception:  # noqa: BLE001
    CardLibrary = None

SEED = 20240722

# (grid, page_color, content_type) - collectively cover every option value.
SCENES = [
    ("1x1", "clear", "slab"),
    ("2x2", "solid", "toploader"),
    ("3x3", "clear", "sleeved"),
    ("4x3", "solid", "sleeved"),
]


def _rand_finish(rng):
    if rng.random() < 0.5:
        return FinishConfig(kind="normal")
    region = ["entire", "picture", "reverse"][int(rng.integers(0, 3))]
    pattern = ["none", "cosmos", "horizontal_lines", "water_web"][int(rng.integers(0, 4))]
    return FinishConfig(kind="holo", holo_region=region, holo_pattern=pattern,
                        physical_texture=False)


def _content_prot(content_type, rng):
    size = ["1mm", "2.5mm"][int(rng.integers(0, 2))]
    if content_type == "sleeved":
        return ProtectionConfig("sleeve", sleeve=SleeveConfig("clear", size))
    if content_type == "toploader":
        return ProtectionConfig("toploader", sleeve=SleeveConfig("clear", size),
                                inner_offset_mm=[float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2))],
                                inner_rot_deg=float(rng.uniform(-1, 1)))
    return ProtectionConfig("slab")


def build_binder_config(grid, page_color, content_type, rng):
    rows, cols = (int(x) for x in grid.split("x"))
    cap = rows * cols
    n = min(cap, max(1, int(round(cap * float(rng.uniform(0.6, 1.0))))))
    filled = sorted(int(s) for s in rng.choice(cap, size=n, replace=False))
    cards = [CardConfig(slot, _content_prot(content_type, rng), _rand_finish(rng),
                        DamageConfig(surface=bool(rng.random() < 0.3)))
             for slot in filled]
    params = {"grid": grid, "content_type": content_type, "page_color": page_color,
              "slot_gap_mm": float(rng.uniform(7.0, 18.0)),
              "two_pages": bool(rng.random() < 0.4),
              "capacity": cap, "filled_slots": filled}
    base = C.sample_scene_config({"layouts": ["binder"]}, SEED)
    cfg = SceneConfig(seed=SEED, layout=LayoutConfig("binder", params), cards=cards,
                      lighting=base.lighting, camera=base.camera, postfx=base.postfx)
    validate_scene_config(cfg)
    return cfg


def setup_reflection_lighting(rng, cam):
    """Low sun + non-sun point lights + colorful reflector objects, all BEHIND the
    camera so they appear only as reflections in the glossy pages."""
    cam_loc = Vector(cam.location)
    view = (Vector((0.0, 0.0, 0.0)) - cam_loc).normalized()
    behind = cam_loc - view * 0.25
    layouts.scatter_reflectors(rng, tuple(behind), spread=0.5, n=10)

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 0.2
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(25))
    bpy.context.collection.objects.link(sun)
    for i in range(3):
        pos = behind + Vector((float(rng.uniform(-0.3, 0.3)), float(rng.uniform(-0.3, 0.3)),
                               float(rng.uniform(-0.15, 0.15))))
        t = float(rng.random())
        pd = bpy.data.lights.new(f"P{i}", type="POINT")
        pd.energy = float(rng.uniform(6.0, 14.0))
        pd.color = (1.0, 0.82 + 0.18 * t, 0.66 + 0.34 * t)
        po = bpy.data.objects.new(f"P{i}", pd)
        po.location = pos
        bpy.context.collection.objects.link(po)


def run_scene(i, grid, page_color, content, lib, cache):
    import numpy as np
    rng = np.random.default_rng(SEED + i)
    sc.reset_scene()
    sc.setup_world(gray=0.12)
    scene = bpy.context.scene
    setup_render(scene, verbose=(i == 0))

    cfg = build_binder_config(grid, page_color, content, rng)
    instances, extent = layouts.build_binder(cfg, lib, cache, rng)

    dist = sc.frame_distance(35, subject_h=extent * 1.12, target_frac=0.9)
    cam = sc.setup_camera(35, 8, 6, dist, target=(0.0, 0.0, 0.0))
    setup_reflection_lighting(rng, cam)

    bpy.context.view_layer.update()
    labels = []
    for _inst, lbl, _reason in label_scene(scene, cam, instances):   # occlusion-aware
        if lbl:
            labels.append(lbl)

    name = f"t11_binder_{grid}_{page_color}_{content}"
    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    scene.render.filepath = os.path.join(out_dir, f"{name}.png")
    bpy.ops.render.render(write_still=True)
    write_poly_label_file(os.path.join(out_dir, f"{name}.txt"), labels)
    print(f"  {name}: {len(labels)}/{len(instances)} cards labeled")


def main():
    if CardLibrary is None:
        print("[t11] cardsource unavailable")
        return
    lib = CardLibrary()
    if lib.is_empty():
        print("[t11] no card images found; aborting")
        return
    cache = os.path.join(_ROOT, config.OUTPUT.root, "card_cache")
    os.makedirs(cache, exist_ok=True)
    for i, (grid, page_color, content) in enumerate(SCENES):
        run_scene(i, grid, page_color, content, lib, cache)
    print("\n[t11] done. 4 binder scenes cover all grids/page-types/content. Visualize each.")


if __name__ == "__main__":
    main()
