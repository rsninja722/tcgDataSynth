"""
Phase 4 (layout 1/5) - TABLE. First full-pipeline scene: a SceneConfig from
rules/combinations drives assembled card instances (finish + damage + physical +
protection) laid out on a cluttered table, then every eligible card is labeled.

Validates multi-instance labeling and frustum handling: the last card is deliberately
shoved fully out of frame and must not be labeled, while the others are.

HOW TO RUN (headless; cv2 and shapely are required in Blender's Python):
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t08_table.py

OUTPUT (out/): t08_table.png + t08_table.txt (custom polygon labels).
Then verify labels:
    python3 labeltools/visualize.py out/t08_table.png out/t08_table.txt

REPORT BACK: attach t08_table_viz.png + paste console. PASS if: multiple cards on a
table with varied finishes/protection; every in-frame front-facing card is boxed with
corners on its sharp corners; the deliberately out-of-frame card is not labeled.
"""
import os
import sys
import math
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import bpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from rules import combinations as C  # noqa: E402
from rules.combinations import (SceneConfig, LayoutConfig, CardConfig, ProtectionConfig,  # noqa: E402
                                FinishConfig, DamageConfig, SleeveConfig, validate_scene_config)
from blender import layouts  # noqa: E402
from blender import scene_builder as sb  # noqa: E402
from blender.labeling import label_scene  # noqa: E402
from blender.render_setup import setup_render  # noqa: E402
from blender import scene_common as sc  # noqa: E402
from labeltools.yolo_pose import write_poly_label_file  # noqa: E402

try:
    from texturegen.cardsource import CardLibrary
except Exception:  # noqa: BLE001
    CardLibrary = None

SEED = 20240720


def _finish(region):
    if region is None:
        return FinishConfig(kind="normal")
    return FinishConfig(kind="holo", holo_region=region, holo_pattern="none",
                        physical_texture=False)


def build_demo_config():
    """Deliberate table scene with one of each holo tag in-frustum (none/full/holo/
    reverse) plus a 5th card shoved out of frame for the frustum-rule test."""
    cards = [
        CardConfig(0, ProtectionConfig("none"), _finish(None), DamageConfig()),
        CardConfig(1, ProtectionConfig("none"), _finish("entire"), DamageConfig()),
        CardConfig(2, ProtectionConfig("sleeve", sleeve=SleeveConfig("clear", "1mm")),
                   _finish("picture"), DamageConfig()),
        CardConfig(3, ProtectionConfig("none"), _finish("reverse"), DamageConfig(surface=True)),
        CardConfig(4, ProtectionConfig("toploader", sleeve=SleeveConfig("clear", "1mm"),
                                       inner_offset_mm=[1.0, 0.0], inner_rot_deg=0.5),
                   _finish(None), DamageConfig()),  # deliberately out of frame
    ]
    layout = LayoutConfig("table", {"clutter_rects": 3, "max_overlap_frac": 0.15})
    base = C.sample_scene_config({"layouts": ["table"]}, SEED)  # reuse lighting/cam/postfx
    cfg = SceneConfig(seed=SEED, layout=layout, cards=cards,
                      lighting=base.lighting, camera=base.camera, postfx=base.postfx)
    validate_scene_config(cfg)
    return cfg


def main():
    import numpy as np
    rng = np.random.default_rng(SEED)
    cache = os.path.join(_ROOT, config.OUTPUT.root, "card_cache")
    os.makedirs(cache, exist_ok=True)

    if CardLibrary is None:
        print("[t08] cardsource unavailable")
        return
    lib = CardLibrary()
    if lib.is_empty():
        print("[t08] no card images found; aborting")
        return

    params = config.load_layout_params("table")
    print(f"[t08] table params: {params}")
    cfg = build_demo_config()
    print(f"[t08] demo table: {len(cfg.cards)} cards; "
          f"protections={[c.protection.kind for c in cfg.cards]}; "
          f"holo_tags={[sb.holo_tag_for_finish(c.finish) for c in cfg.cards]}")

    sc.reset_scene()
    sc.setup_world(gray=0.2)
    scene = bpy.context.scene
    setup_render(scene, verbose=True)

    instances = layouts.build_table(cfg, lib, cache, rng, allow_overlap=params["allow_overlap"])

    # Deliberately shove the last card out of frame (frustum-rule test); keeps the
    # first four (one of each holo tag) in-frustum.
    if instances:
        z = instances[-1].root.location[2]
        instances[-1].root.location = (0.30, 0.0, z)

    # Camera above the table, looking down at an angle (generous frame so the 4
    # in-frustum cards are comfortably inside).
    dist = sc.frame_distance(35, subject_h=0.42, target_frac=0.9)
    cam = sc.setup_camera(35, 18, 58, dist, target=(0.0, 0.0, 0.0))
    sc.add_lights(cam.location, rng=rng, target=(0.0, 0.0, 0.0))

    bpy.context.view_layer.update()
    labels = []
    removed = 0
    for inst, lbl, reason in label_scene(scene, cam, instances):   # occlusion-aware
        print(f"   {inst.card_id:20} tag={inst.holo_tag:8} {reason}")
        if lbl:
            labels.append(lbl)
        elif reason == "fully-out-of-frustum" and params["out_of_frustum"] == "remove":
            for obj in inst.objects:      # don't render frustum-excluded cards at all
                obj.hide_render = True
            removed += 1

    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    scene.render.filepath = os.path.join(out_dir, "t08_table.png")
    bpy.ops.render.render(write_still=True)
    write_poly_label_file(os.path.join(out_dir, "t08_table.txt"), labels)
    print(f"\n[t08] labeled {len(labels)}/{len(instances)} cards; "
          f"removed {removed} out-of-frustum (mode={params['out_of_frustum']}). "
          f"Visualize out/t08_table.png + .txt")


if __name__ == "__main__":
    main()
