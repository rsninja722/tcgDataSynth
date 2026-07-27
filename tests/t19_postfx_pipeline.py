r"""Phase 6 integration acceptance - rendered post-effect image/label transaction.

Builds one sampled table scene with at least one active post effect, renders it through
the production output helper, and checks that only the completed PNG/custom-label pair
is published. The raw and intermediate staged images must not remain in ``out/``.

HOW TO RUN (headless; cv2 and shapely are required in Blender's Python):
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tests\t19_postfx_pipeline.py

OUTPUT (out/):
    t19_postfx_pipeline.png + .txt + _viz.png
    t19_postfx_pipeline_report.json

REPORT BACK: attach the visualization and report, and paste the console. PASS if the
processed render looks plausible, label corners remain aligned, and the report says
there were no staged leftovers and at least one sampled post effect was active.
"""
from __future__ import annotations

from dataclasses import asdict
import json
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
from blender import camera as phase5_camera  # noqa: E402
from blender import layouts  # noqa: E402
from blender import lighting as phase5_lighting  # noqa: E402
from blender import scene_common as sc  # noqa: E402
from blender.labeling import label_scene  # noqa: E402
from blender.render_output import render_poly_label_pair  # noqa: E402
from blender.render_setup import setup_render  # noqa: E402
from labeltools.visualize import visualize_poly_label_file  # noqa: E402
from rules import combinations as C  # noqa: E402

try:
    from texturegen.cardsource import CardLibrary
except Exception as exc:  # noqa: BLE001
    CardLibrary = None
    _CARD_IMPORT_ERROR = exc
else:
    _CARD_IMPORT_ERROR = None


def _sample_with_postfx():
    for seed in range(20260730, 20270730):
        cfg = C.sample_scene_config({"layouts": ["table"], "back_to_camera_prob": 0.0}, seed)
        if any(getattr(cfg.postfx, name) is not None for name in C.POST_EFFECTS):
            return cfg
    raise RuntimeError("No post effect enabled; raise a postfx probability above zero in config.json")


def main() -> None:
    if tuple(bpy.app.version) != (5, 0, 0):
        raise RuntimeError(f"t19 requires Blender 5.0.0, got {bpy.app.version_string}")
    if CardLibrary is None:
        raise RuntimeError("[t19] cardsource unavailable") from _CARD_IMPORT_ERROR
    library = CardLibrary()
    if library.is_empty():
        raise RuntimeError("[t19] no card images found")

    import numpy as np

    cfg = _sample_with_postfx()
    active = [name for name in C.POST_EFFECTS if getattr(cfg.postfx, name) is not None]
    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    cache = os.path.join(out_dir, "card_cache")
    image_path = os.path.join(out_dir, "t19_postfx_pipeline.png")
    label_path = os.path.join(out_dir, "t19_postfx_pipeline.txt")
    visualization_path = os.path.join(out_dir, "t19_postfx_pipeline_viz.png")
    report_path = os.path.join(out_dir, "t19_postfx_pipeline_report.json")
    for path in (image_path, label_path, visualization_path, report_path):
        if os.path.isfile(path):
            os.remove(path)
    os.makedirs(cache, exist_ok=True)

    sc.reset_scene()
    sc.setup_world(gray=0.025)
    scene = bpy.context.scene
    setup_render(scene, verbose=True)
    instances = layouts.build_table(
        cfg, library, cache, np.random.default_rng(cfg.seed),
        allow_overlap=config.load_layout_params("table")["allow_overlap"])
    bpy.context.view_layer.update()
    target, extent = phase5_camera.subject_target_and_extent(instances)
    camera = phase5_camera.build_camera(cfg.camera, target, extent)
    rig = phase5_lighting.build_lighting(cfg.lighting, camera, target, extent)
    results = label_scene(scene, camera, instances)
    phase5_camera.focus_on_labeled_card(camera, cfg.camera, results)
    bpy.context.view_layer.update()
    labels = [label for _instance, label, _reason in results if label is not None]
    assert labels, "t19 requires at least one visible labeled card"

    render_poly_label_pair(scene, image_path, label_path, labels, cfg.postfx)
    assert os.path.isfile(image_path) and os.path.isfile(label_path)
    leftovers = [name for name in os.listdir(out_dir)
                 if name.startswith("t19_postfx_pipeline.postfx-")]
    assert not leftovers, f"staged output leaked: {leftovers}"
    visualization_path = visualize_poly_label_file(image_path, label_path, visualization_path)
    report = {
        "blender_version": bpy.app.version_string,
        "scene_seed": cfg.seed,
        "active_postfx": active,
        "postfx": asdict(cfg.postfx),
        "label_count": len(labels),
        "image_exists": os.path.isfile(image_path),
        "label_exists": os.path.isfile(label_path),
        "staged_leftovers": leftovers,
        "lighting_occluders": len(rig.occluders),
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[t19] PASS: effects={active}; wrote {visualization_path} and {report_path}")
    print("[t19] Review image realism, label alignment, and report staged_leftovers=[]")


if __name__ == "__main__":
    main()
