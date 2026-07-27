r"""Phase 5 checkpoint 1 - sampled lighting, camera, framing, and DoF.

Builds one four-card table scene once, including one card showing its back, then renders
it under nine distinct sampled lighting/camera configurations. The nine views cover
spot-only,
points-only, combined lights, four points, a patterned shadow mask, wide/telephoto,
and low/high off-axis cameras.

HOW TO RUN (headless; cv2 and shapely are required in Blender's Python):
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tests\t16_lighting_camera.py

OUTPUT (out/):
    t16_phase5_00..08.png + .txt + _viz.png
    t16_phase5_contact.png
    t16_phase5_report.json

REPORT BACK: attach the contact sheet and report, and paste the console. PASS if the
same cards/layout appear in all nine tiles; camera angle/scale and lighting visibly
vary; spot-only resembles a cold phone flash; warm/cold points vary; the occluder tile
has a patterned cast shadow rather than a fully black scene; at least one card is visibly
sharp in every tile; and labels remain aligned.
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from dataclasses import asdict

warnings.filterwarnings("ignore", category=DeprecationWarning)

import bpy
import cv2
from mathutils import Vector

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
from blender.render_setup import setup_render  # noqa: E402
from labeltools.visualize import visualize_poly_label_file  # noqa: E402
from labeltools.yolo_pose import write_poly_label_file  # noqa: E402
from rules import combinations as C  # noqa: E402

try:
    from texturegen.cardsource import CardLibrary
except Exception as exc:  # noqa: BLE001
    CardLibrary = None
    _CARD_IMPORT_ERROR = exc
else:
    _CARD_IMPORT_ERROR = None

SCENE_SEED = 20240727
ENVIRONMENT_SEED_START = 20250000
_BLENDER_FLOAT_ABS_TOL = 1e-5
_BLENDER_ANGLE_ABS_TOL_DEG = 1e-3
_ENV_OPTIONS = {"layouts": ["table"], "back_to_camera_prob": 0.5}


def _shadow_mask_count(lighting_cfg):
    return int(lighting_cfg.spotlight_shadow_mask_seed is not None) + \
        sum(point.shadow_mask_seed is not None for point in lighting_cfg.point_lights)


_CRITERIA = (
    ("spot_only", lambda cfg: cfg.lighting.spotlight_beside_camera
     and not cfg.lighting.point_lights),
    ("warm_points", lambda cfg: not cfg.lighting.spotlight_beside_camera
     and any(point.color_temp <= 4000.0 for point in cfg.lighting.point_lights)),
    ("cold_combined", lambda cfg: cfg.lighting.spotlight_beside_camera
     and any(point.color_temp >= 8000.0 for point in cfg.lighting.point_lights)),
    ("four_points", lambda cfg: len(cfg.lighting.point_lights) == 4),
    ("shadow_mask", lambda cfg: _shadow_mask_count(cfg.lighting) > 0),
    ("wide", lambda cfg: cfg.camera.focal_mm <= 20.0),
    ("tele", lambda cfg: cfg.camera.focal_mm >= 50.0),
    ("high_offaxis", lambda cfg: cfg.camera.offaxis_deg >= 45.0),
    ("low_offaxis", lambda cfg: cfg.camera.offaxis_deg <= 5.0),
)


def _select_environments():
    missing = dict(_CRITERIA)
    selected = []
    for seed in range(ENVIRONMENT_SEED_START, ENVIRONMENT_SEED_START + 10000):
        cfg = C.sample_scene_config(_ENV_OPTIONS, seed)
        for name, predicate in tuple(missing.items()):
            if predicate(cfg):
                selected.append((name, cfg))
                del missing[name]
                break
        if not missing:
            return selected
    raise RuntimeError(f"Could not sample Phase 5 acceptance coverage: {sorted(missing)}")


def _layout_signature(instances):
    signature = []
    for instance in instances:
        values = tuple(round(float(value), 9)
                       for row in instance.root.matrix_world for value in row)
        signature.append((instance.card_id, values))
    return tuple(signature)


def _actual_offaxis(camera, target):
    direction = (camera.matrix_world.translation - Vector(target)).normalized()
    return math.degrees(math.acos(max(-1.0, min(1.0, direction.z))))


def _assert_view(scene, camera, camera_cfg, lighting_cfg, rig, target,
                 results, signature, instances):
    assert scene.render.resolution_x == 1280 and scene.render.resolution_y == 1280
    assert camera.data.type == "PERSP"
    stored_lens = float(camera.data.lens)
    configured_lens = float(camera_cfg.focal_mm)
    assert math.isclose(stored_lens, configured_lens, rel_tol=0.0,
                        abs_tol=_BLENDER_FLOAT_ABS_TOL), (
        f"Blender lens differs from config: stored={stored_lens:.9f}mm "
        f"configured={configured_lens:.9f}mm")
    actual_offaxis = _actual_offaxis(camera, target)
    configured_offaxis = float(camera_cfg.offaxis_deg)
    assert math.isclose(actual_offaxis, configured_offaxis, rel_tol=0.0,
                        abs_tol=_BLENDER_ANGLE_ABS_TOL_DEG), (
        f"Camera off-axis angle differs from config: actual={actual_offaxis:.9f}deg "
        f"configured={configured_offaxis:.9f}deg")
    assert camera.data.dof.use_dof is True
    stored_fstop = float(camera.data.dof.aperture_fstop)
    configured_fstop = float(camera_cfg.aperture_fstop)
    assert math.isclose(stored_fstop, configured_fstop, rel_tol=0.0,
                        abs_tol=_BLENDER_FLOAT_ABS_TOL), (
        f"Blender f-stop differs from config: stored={stored_fstop:.9f} "
        f"configured={configured_fstop:.9f}")
    labeled_instances = [instance for instance, label, _reason in results if label is not None]
    assert labeled_instances, "each Phase 5 view needs at least one label"
    assert camera.data.dof.focus_object in [instance.card for instance in labeled_instances]
    assert rig.sun.type == "LIGHT" and rig.sun.data.type == "SUN"
    assert rig.sun["tcg_front_dot"] > 0.0
    assert (rig.spotlight is not None) == lighting_cfg.spotlight_beside_camera
    assert len(rig.points) == len(lighting_cfg.point_lights)
    assert rig.spotlight is not None or rig.points
    assert all(point["tcg_front_dot"] > 0.0 for point in rig.points)
    assert len(rig.occluders) == _shadow_mask_count(lighting_cfg)
    assert all(mask["tcg_shadow_grid_faces"] == 50 for mask in rig.occluders)
    assert _layout_signature(instances) == signature, "environment draw changed the layout"


def _write_contact_sheet(out_dir, views):
    tiles = []
    for view in views:
        image = cv2.imread(view["visualization"])
        if image is None:
            raise RuntimeError(
                f"Could not read label visualization {view['visualization']!r}")
        tile = cv2.resize(image, (416, 416), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (416, 48), (12, 12, 12), -1)
        line1 = f"{view['index']:02d} {view['coverage']} seed={view['seed']}"
        line2 = (f"{view['focal_mm']:.1f}mm off={view['offaxis_deg']:.1f} "
                 f"f/{view['fstop']:.1f} spot={int(view['spot'])} "
                 f"pts={view['points']} masks={view['shadow_masks']}")
        cv2.putText(tile, line1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(tile, line2, (8, 39), cv2.FONT_HERSHEY_SIMPLEX,
                    0.39, (215, 225, 235), 1, cv2.LINE_AA)
        tiles.append(tile)
    rows = [cv2.hconcat(tiles[index:index + 3]) for index in range(0, 9, 3)]
    contact = cv2.vconcat(rows)
    path = os.path.join(out_dir, "t16_phase5_contact.png")
    if not cv2.imwrite(path, contact):
        raise RuntimeError(f"Could not write contact sheet {path!r}")
    return path


def main():
    if tuple(bpy.app.version) != (5, 0, 0):
        raise RuntimeError(f"t16 requires Blender 5.0.0, got {bpy.app.version_string}")
    if CardLibrary is None:
        raise RuntimeError("[t16] cardsource unavailable") from _CARD_IMPORT_ERROR
    library = CardLibrary()
    if library.is_empty():
        raise RuntimeError("[t16] no card images found")

    import numpy as np

    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    cache = os.path.join(out_dir, "card_cache")
    os.makedirs(cache, exist_ok=True)
    for filename in os.listdir(out_dir):
        if filename.startswith("t16_phase5_"):
            os.remove(os.path.join(out_dir, filename))

    scene_cfg = C.sample_scene_config(_ENV_OPTIONS, SCENE_SEED)
    if len(scene_cfg.cards) != 4:
        raise RuntimeError(f"Expected four cards for fixed seed, got {len(scene_cfg.cards)}")
    if sum(card.back_to_camera for card in scene_cfg.cards) != 1:
        raise RuntimeError("Fixed Phase 5 scene must contain exactly one back-facing card")
    layout_rng = np.random.default_rng(SCENE_SEED)
    sc.reset_scene()
    sc.setup_world(gray=0.025)
    scene = bpy.context.scene
    setup_render(scene, verbose=True)
    params = config.load_layout_params("table")
    instances = layouts.build_table(
        scene_cfg, library, cache, layout_rng, allow_overlap=params["allow_overlap"])
    bpy.context.view_layer.update()
    target, extent = phase5_camera.subject_target_and_extent(instances)
    signature = _layout_signature(instances)

    views = []
    camera = None
    rig = None
    for index, (coverage, environment_cfg) in enumerate(_select_environments()):
        phase5_lighting.remove_light_rig(rig)
        phase5_camera.remove_camera(camera)
        camera = phase5_camera.build_camera(environment_cfg.camera, target, extent)
        rig = phase5_lighting.build_lighting(
            environment_cfg.lighting, camera, target, extent)
        bpy.context.view_layer.update()
        results = label_scene(scene, camera, instances)
        focused = phase5_camera.focus_on_labeled_card(
            camera, environment_cfg.camera, results)
        bpy.context.view_layer.update()
        _assert_view(scene, camera, environment_cfg.camera, environment_cfg.lighting,
                     rig, target, results, signature, instances)

        stem = f"t16_phase5_{index:02d}"
        image_path = os.path.join(out_dir, stem + ".png")
        label_path = os.path.join(out_dir, stem + ".txt")
        labels = [label for _instance, label, _reason in results if label is not None]
        scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)
        write_poly_label_file(label_path, labels)
        visualization_path = os.path.join(out_dir, stem + "_viz.png")
        visualize_poly_label_file(image_path, label_path, visualization_path)

        view = {
            "index": index,
            "coverage": coverage,
            "seed": environment_cfg.seed,
            "focal_mm": environment_cfg.camera.focal_mm,
            "stored_focal_mm": float(camera.data.lens),
            "offaxis_deg": environment_cfg.camera.offaxis_deg,
            "actual_offaxis_deg": _actual_offaxis(camera, target),
            "orbit_deg": environment_cfg.camera.orbit_deg,
            "fstop": environment_cfg.camera.aperture_fstop,
            "spot": environment_cfg.lighting.spotlight_beside_camera,
            "points": len(environment_cfg.lighting.point_lights),
            "shadow_masks": _shadow_mask_count(environment_cfg.lighting),
            "focused_card_id": focused.card_id,
            "label_count": len(labels),
            "camera": asdict(environment_cfg.camera),
            "lighting": asdict(environment_cfg.lighting),
            "image": image_path,
            "label": label_path,
            "visualization": visualization_path,
        }
        views.append(view)
        print(f"[t16] {index:02d} {coverage:12} seed={environment_cfg.seed} "
              f"lens={view['focal_mm']:.6f}/{view['stored_focal_mm']:.6f}mm "
              f"off={view['offaxis_deg']:.6f}/{view['actual_offaxis_deg']:.6f}deg "
              f"orbit={view['orbit_deg']:.1f} f/{view['fstop']:.1f} "
               f"spot={view['spot']} points={view['points']} masks={view['shadow_masks']} "
              f"focus={focused.card_id} labels={len(labels)}")

    contact_path = _write_contact_sheet(out_dir, views)
    report = {
        "blender_version": bpy.app.version_string,
        "scene_seed": SCENE_SEED,
        "card_ids": [instance.card_id for instance in instances],
        "back_facing_card_ids": [instance.card_id for instance, card_cfg
                                 in zip(instances, scene_cfg.cards)
                                 if card_cfg.back_to_camera],
        "target": [float(value) for value in target],
        "frame_extent_m": float(extent),
        "same_layout_all_views": True,
        "views": [{key: value for key, value in view.items()
                   if key not in {"image", "label", "visualization"}} for view in views],
    }
    report_path = os.path.join(out_dir, "t16_phase5_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\n[t16] PASS: wrote {contact_path} and {report_path}")
    print("[t16] Review all nine views for lighting, framing, partial shadow, and DoF.")


if __name__ == "__main__":
    main()
