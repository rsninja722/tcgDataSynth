r"""
Phase 4 (layout 5/5, checkpoint 2) - INTEGRATED HAND GRIPS.

Builds five deterministic one-card scenes from the validated CC0 left/right hand rig:
bare/sleeved/toploader pinch grips and sleeved/toploader side grips. Hands approach
from varied angles and grip at shallow/normal positions above the reused noisy table.
Hands are deliberately excluded from label occlusion, so each in-frame card keeps its
original four-corner polygon even where fingers cover the render.

HOW TO RUN (headless; the bundled hand rig, cv2, and shapely are required):
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tests\t14_hand.py

OUTPUT (out/): t14_hand_<case>.png + .txt. Then run:
    python labeltools\visualize_all.py

REPORT BACK: attach all five ``t14_hand_*_viz.png`` files and paste the console.
PASS if each hand reaches the card from the named direction; pinch has thumb in front
and index behind; side has thumb on one face and fingers on the other; shallow/normal
placement visibly differs; the table fills the background; skin tones vary plausibly;
and every card keeps exactly its four original corner flags (1,2,4,3). Neither hands
nor transparent sleeves/toploaders may carve hand-scene labels.
"""
from __future__ import annotations

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
from blender import layouts  # noqa: E402
from blender import scene_common as sc  # noqa: E402
from blender.labeling import label_scene  # noqa: E402
from blender.render_setup import setup_render  # noqa: E402
from labeltools.yolo_pose import write_poly_label_file  # noqa: E402
from rules import combinations as C  # noqa: E402
from rules.combinations import (CardConfig, DamageConfig, FinishConfig, LayoutConfig,  # noqa: E402
                                ProtectionConfig, SceneConfig, SleeveConfig,
                                validate_scene_config)

try:
    from texturegen.cardsource import CardLibrary
except Exception as exc:  # noqa: BLE001
    CardLibrary = None
    _CARD_IMPORT_ERROR = exc
else:
    _CARD_IMPORT_ERROR = None

SEED = 20240725
# name, grip, protection, hand, approach degrees, normalized depth
CASES = (
    ("pinch_bare_left_east_shallow", "pinch", "none", "left", 0.0, 0.12),
    ("pinch_sleeve_right_north_normal", "pinch", "sleeve", "right", 90.0, 0.34),
    ("pinch_toploader_left_southwest", "pinch", "toploader", "left", 225.0, 0.34),
    ("side_sleeve_left_northeast", "side", "sleeve", "left", 45.0, 0.20),
    ("side_toploader_right_west_normal", "side", "toploader", "right", 180.0, 0.34),
)
LEGACY_DEEP_STEMS = (
    "t14_hand_pinch_sleeve_right_north_deep",
    "t14_hand_side_toploader_right_west_deep",
)


def _protection(kind):
    if kind == "none":
        return ProtectionConfig("none")
    sleeve = SleeveConfig("clear", "1mm")
    if kind == "sleeve":
        return ProtectionConfig("sleeve", sleeve=sleeve)
    return ProtectionConfig(
        "toploader", sleeve=sleeve, inner_offset_mm=[1.2, -0.8], inner_rot_deg=0.6)


def _config(index, grip, protection, handedness, approach, depth):
    finish = (FinishConfig("normal") if index % 2 == 0 else
              FinishConfig("holo", holo_region="picture", holo_pattern="cosmos",
                           physical_texture=False))
    card = CardConfig(index, _protection(protection), finish, DamageConfig())
    params = {"grip": grip, "handedness": handedness,
              "approach_deg": approach, "depth": depth}
    base = C.sample_scene_config({"layouts": ["hand"]}, SEED + index)
    cfg = SceneConfig(SEED + index, LayoutConfig("hand", params), [card],
                      base.lighting, base.camera, base.postfx)
    validate_scene_config(cfg)
    return cfg


def _run_case(index, spec, library, cache):
    import numpy as np
    name, grip, protection, handedness, approach, depth = spec
    rng = np.random.default_rng(SEED + index)
    sc.reset_scene()
    sc.setup_world(gray=0.10)
    scene = bpy.context.scene
    setup_render(scene, verbose=(index == 0))

    cfg = _config(index, grip, protection, handedness, approach, depth)
    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"t14_hand_{name}"
    for extension in (".png", ".txt", "_viz.png"):
        stale = os.path.join(out_dir, stem + extension)
        if os.path.isfile(stale):
            os.remove(stale)

    instances, extent = layouts.build_hand(cfg, library, cache, rng)
    distance = sc.frame_distance(45, subject_h=extent, target_frac=0.88)
    camera = sc.setup_camera(45, 10.0, 16.0, distance, target=(0.0, 0.0, 0.0))
    sc.add_lights(camera.location, target=(0.0, 0.0, 0.0), rng=rng)
    bpy.context.view_layer.update()

    results = label_scene(scene, camera, instances)
    labels = [label for _instance, label, _reason in results if label is not None]
    reasons = [reason for _instance, _label, reason in results]
    acceptance_error = None
    if len(labels) != 1:
        acceptance_error = f"expected one hand-held card label, got {reasons}"
    elif [flag for _x, _y, flag in labels[0].points] != [1, 2, 4, 3]:
        acceptance_error = "hand-held card did not retain its original four-corner label"
    points = len(labels[0].points) if labels else 0

    scene.render.filepath = os.path.join(out_dir, stem + ".png")
    bpy.ops.render.render(write_still=True)
    write_poly_label_file(os.path.join(out_dir, stem + ".txt"), labels)
    print(f"[t14] {name}: grip={grip} protection={protection} hand={handedness} "
          f"approach={approach:.0f} depth={depth:.2f} reasons={reasons} "
          f"polygon_points={points}")
    if acceptance_error:
        return f"{name}: {acceptance_error}"
    return None


def main():
    if CardLibrary is None:
        raise RuntimeError("[t14] cardsource unavailable") from _CARD_IMPORT_ERROR
    library = CardLibrary()
    if library.is_empty():
        raise RuntimeError("[t14] no card images found")
    cache = os.path.join(_ROOT, config.OUTPUT.root, "card_cache")
    os.makedirs(cache, exist_ok=True)
    for stem in LEGACY_DEEP_STEMS:
        for extension in (".png", ".txt", "_viz.png"):
            stale = os.path.join(_ROOT, config.OUTPUT.root, stem + extension)
            if os.path.isfile(stale):
                os.remove(stale)
    failures = []
    for index, spec in enumerate(CASES):
        failure = _run_case(index, spec, library, cache)
        if failure:
            failures.append(failure)
    if failures:
        raise AssertionError("[t14] acceptance failures (renders retained): "
                             + "; ".join(failures))
    print("\n[t14] done. Visualize and report all five t14_hand_* pairs.")


if __name__ == "__main__":
    main()
