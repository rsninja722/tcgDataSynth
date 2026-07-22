"""
Phase 4 (layout 4/5) - DISPLAY CASE (spec §3.5.3). A tight grid of toploadered or
slabbed cards on a random-material base with side walls, all flat OR all tilted
forward 25deg, under a scratched/smudged 6mm acrylic cover 40mm above the tallest
item. Grid capped at 24 cards (user). A 20% chance adds a stray card ON TOP of the
lid. Every front-facing card that is fully OR partially in the frustum is labeled
(partial -> class 1 'partial_card'); fully-out cards are not labeled.

Scene 0 is rendered at THREE zoom levels: 'out' (whole case small in frame), 'full'
(default), 'in' (most zoomed in -> only a few cards fully in frame, the rest become
partial_card). Scene 0 also shoves the LAST card fully out to show it is unlabeled.

HOW TO RUN (headless; cv2 in Blender recommended):
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t12_display_case.py

OUTPUT (out/): t12_case_<content>_<flat|tilt25>_<CxR>_<out|full|in>.png + .txt. Verify e.g.:
    python3 labeltools/visualize.py out/t12_case_toploader_flat_5x5_in.png out/t12_case_toploader_flat_5x5_in.txt

REPORT BACK: attach each viz. PASS if: cards sit in a tight walled case under a clear
scratched lid ~40mm above them; tilted scenes lean cards forward ~25deg; full cards
are labeled (white boxes, 4 corners); the ZOOMED-IN view shows orange 'partial_card'
boxes whose 2 crossing keypoints sit exactly on the frame edge; the shoved card is NOT
labeled; the 5x5 scene has 24 cards; ~1 in 5 scenes shows a card resting on the lid.
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

SEED = 20240723

# (content_type, tilt_forward, cols, rows). Collectively cover both contents, both
# flat/tilted, and the 24-cap (5x5 -> 24). Scene 0 also does the frustum shove.
SCENES = [
    ("toploader", False, 5, 5),   # 25 -> capped to 24 (one empty corner), flat
    ("toploader", True,  4, 3),   # 12, tilted forward 25deg
    ("slab",      False, 3, 3),   # 9, flat
    ("slab",      True,  4, 4),   # 16, tilted forward 25deg
]


def _rand_finish(rng):
    if rng.random() < 0.5:
        return FinishConfig(kind="normal")
    region = ["entire", "picture", "reverse"][int(rng.integers(0, 3))]
    pattern = ["none", "cosmos", "horizontal_lines", "water_web"][int(rng.integers(0, 4))]
    return FinishConfig(kind="holo", holo_region=region, holo_pattern=pattern,
                        physical_texture=False)


def _content_prot(content_type, rng):
    if content_type == "toploader":
        size = ["1mm", "2.5mm"][int(rng.integers(0, 2))]
        return ProtectionConfig("toploader", sleeve=SleeveConfig("clear", size),
                                inner_offset_mm=[float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2))],
                                inner_rot_deg=float(rng.uniform(-1, 1)))
    return ProtectionConfig("slab")


def build_case_config(content_type, tilt, cols, rows, rng):
    n = min(cols * rows, C.DISPLAY_CASE_MAX_CARDS)     # honor the 24-card cap
    rows_eff = (n + cols - 1) // cols
    cards = [CardConfig(i, _content_prot(content_type, rng), _rand_finish(rng),
                        DamageConfig(surface=bool(rng.random() < 0.3)))
             for i in range(n)]
    params = {"cols": cols, "rows": rows_eff, "tilt_forward": tilt,
              "tilt_deg": 25.0 if tilt else 0.0, "cover_scratches": True}
    base = C.sample_scene_config({"layouts": ["display_case"]}, SEED)
    cfg = SceneConfig(seed=SEED, layout=LayoutConfig("display_case", params), cards=cards,
                      lighting=base.lighting, camera=base.camera, postfx=base.postfx)
    validate_scene_config(cfg)
    return cfg


def _clear_cams_lights():
    """Remove existing cameras + lights so we can re-frame the SAME built scene."""
    for o in list(bpy.data.objects):
        if o.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(o, do_unlink=True)


def case_camera(mode, extent):
    """Three case framings from one viewpoint (az10/el30). 'out' = whole case small
    in frame (most zoomed out), 'full' = whole case fills frame (default), 'in' =
    most zoomed in so only a few cards are fully in frame (the rest go partial)."""
    if mode == "out":
        dist = sc.frame_distance(35, subject_h=extent + 0.08, target_frac=0.30)
    elif mode == "in":
        dist = sc.frame_distance(35, subject_h=0.18, target_frac=0.90)
    else:  # full
        dist = sc.frame_distance(35, subject_h=extent + 0.08, target_frac=0.82)
    return sc.setup_camera(35, 10, 30, dist, target=(0.0, 0.0, 0.04))


def run_scene(i, content, tilt, cols, rows, lib, cache, modes=("full",), shove_last=False):
    import numpy as np
    rng = np.random.default_rng(SEED + i)
    sc.reset_scene()
    sc.setup_world(gray=0.12)
    scene = bpy.context.scene
    setup_render(scene, verbose=(i == 0))
    scene_params = config.load_layout_params("display_case")

    cfg = build_case_config(content, tilt, cols, rows, rng)
    instances, extent = layouts.build_display_case(cfg, lib, cache, rng)

    if shove_last and instances:
        instances[-1].root.location.x += extent      # fully out of frame -> unlabeled
        bpy.context.view_layer.update()

    tiltname = "tilt25" if tilt else "flat"
    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    for mode in modes:
        _clear_cams_lights()
        for inst in instances:                        # reset any prior 'remove' hiding
            for obj in inst.objects:
                obj.hide_render = False
        cam = case_camera(mode, extent)
        sc.add_lights(cam.location, rng=np.random.default_rng(SEED + i + 777),
                      target=(0.0, 0.0, 0.04))
        bpy.context.view_layer.update()
        # Occlusion-aware second pass: cards carve the bounds of cards behind them.
        results = label_scene(scene, cam, instances)
        labels, n_partial, removed = [], 0, 0
        for inst, lbl, reason in results:
            if lbl:
                labels.append(lbl)
                n_partial += int(reason == "labeled-partial")
            elif reason == "fully-out-of-frustum" and scene_params["out_of_frustum"] == "remove":
                for obj in inst.objects:
                    obj.hide_render = True
                removed += 1
        name = f"t12_case_{content}_{tiltname}_{cols}x{rows}_{mode}"
        scene.render.filepath = os.path.join(out_dir, f"{name}.png")
        bpy.ops.render.render(write_still=True)
        write_poly_label_file(os.path.join(out_dir, f"{name}.txt"), labels)
        print(f"  {name}: {len(instances)} cards, {len(labels)} labeled "
              f"({n_partial} partial, removed {removed})")


def main():
    if CardLibrary is None:
        print("[t12] cardsource unavailable")
        return
    lib = CardLibrary()
    if lib.is_empty():
        print("[t12] no card images found; aborting")
        return
    cache = os.path.join(_ROOT, config.OUTPUT.root, "card_cache")
    os.makedirs(cache, exist_ok=True)
    for i, (content, tilt, cols, rows) in enumerate(SCENES):
        # Scene 0 is rendered at all three zoom levels (out/full/in); the zoomed-in
        # view crops edge cards into 'partial_card' labels. Others use the full frame.
        modes = ("out", "full", "in") if i == 0 else ("full",)
        run_scene(i, content, tilt, cols, rows, lib, cache, modes=modes, shove_last=(i == 0))
    write_dataset_yaml(os.path.join(_ROOT, config.OUTPUT.root, "dataset.yaml"))
    print("\n[t12] done. Scene 0 at 3 zooms (out/full/in), scenes 1-3 full. "
          "Visualize each; scene 0's shoved card must be unlabeled; the zoomed-in "
          "view should show orange 'partial_card' boxes with on-edge keypoints.")


if __name__ == "__main__":
    main()
