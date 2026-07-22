"""
Phase 4 (layout 2/5) - FLOATING (spec §3.5.5). Cards float in space at varied depths
and orientations; random textured prisms + cylinders are scattered in the background
behind a background plane. Full pipeline driven by a SceneConfig; every eligible card
(front-facing + in frustum) is labeled, with the holo tag.

HOW TO RUN (headless; cv2 in Blender recommended):
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t09_floating.py

OUTPUT (out/): t09_floating.png + t09_floating.txt. Verify:
    python3 labeltools/visualize.py out/t09_floating.png out/t09_floating.txt

REPORT BACK: attach the viz. PASS if: cards float at different depths/angles over
scattered prisms/cylinders; front-facing in-frame cards are labeled (corners correct,
holo tags right); back-facing / out-of-frame cards are not.
"""
import os
import sys
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
from blender.labeling import label_scene  # noqa: E402
from blender.render_setup import setup_render  # noqa: E402
from blender import scene_common as sc  # noqa: E402
from labeltools.yolo_pose import write_poly_label_file, write_dataset_yaml  # noqa: E402

try:
    from texturegen.cardsource import CardLibrary
except Exception:  # noqa: BLE001
    CardLibrary = None

SEED = 20240721


def _holo(region, pattern):
    return FinishConfig(kind="holo", holo_region=region, holo_pattern=pattern,
                        physical_texture=False)


def _sleeve():
    return ProtectionConfig("sleeve", sleeve=SleeveConfig("clear", "1mm"))


def _toploader():
    return ProtectionConfig("toploader", sleeve=SleeveConfig("clear", "1mm"),
                            inner_offset_mm=[1.0, 0.0], inner_rot_deg=0.5)


def build_holo_protection_demo(max_cards):
    """ALL holo cards, each in a sleeve or toploader, varied region/pattern -- to see
    whether the plastic affects the holo across many combos + (floating) angles."""
    specs = [
        (_sleeve(),    _holo("entire", "cosmos")),
        (_toploader(), _holo("entire", "horizontal_lines")),
        (_sleeve(),    _holo("picture", "water_web")),
        (_toploader(), _holo("reverse", "cosmos")),
        (_sleeve(),    _holo("entire", "none")),
        (_toploader(), _holo("picture", "cosmos")),
    ][:max(1, max_cards)]
    cards = [CardConfig(i, prot, fin, DamageConfig()) for i, (prot, fin) in enumerate(specs)]
    layout = LayoutConfig("floating", {})
    base = C.sample_scene_config({"layouts": ["floating"]}, SEED)
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
        print("[t09] cardsource unavailable")
        return
    lib = CardLibrary()
    if lib.is_empty():
        print("[t09] no card images found; aborting")
        return

    params = config.load_layout_params("floating")
    cfg = build_holo_protection_demo(params["max_cards"])
    print(f"[t09] floating (all holo in sleeve/toploader): {len(cfg.cards)} cards; params={params}")

    sc.reset_scene()
    sc.setup_world(gray=0.12)
    scene = bpy.context.scene
    setup_render(scene, verbose=True)

    instances = layouts.build_floating(cfg, lib, cache, rng,
                                       allow_overlap=params["allow_overlap"],
                                       max_shapes=params["max_shapes"])

    dist = sc.frame_distance(40, subject_h=0.34, target_frac=0.85)
    cam = sc.setup_camera(40, 12, 8, dist, target=(0.0, 0.0, -0.03))
    sc.add_lights(cam.location, rng=rng, target=(0.0, 0.0, -0.03))

    bpy.context.view_layer.update()
    labels, removed = [], 0
    for inst, lbl, reason in label_scene(scene, cam, instances):   # occlusion-aware
        print(f"   {inst.card_id:20} tag={inst.holo_tag:8} {reason}")
        if lbl:
            labels.append(lbl)
        elif reason == "fully-out-of-frustum" and params["out_of_frustum"] == "remove":
            for obj in inst.objects:
                obj.hide_render = True
            removed += 1

    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    scene.render.filepath = os.path.join(out_dir, "t09_floating.png")
    bpy.ops.render.render(write_still=True)
    write_poly_label_file(os.path.join(out_dir, "t09_floating.txt"), labels)
    write_dataset_yaml(os.path.join(out_dir, "dataset.yaml"))
    print(f"\n[t09] labeled {len(labels)}/{len(instances)} cards; removed {removed}. "
          f"Visualize out/t09_floating.png + .txt")


if __name__ == "__main__":
    main()
