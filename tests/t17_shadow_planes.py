r"""Lighting revamp checkpoint - deterministic simplex shadow planes.

Renders one fixed card/background scene with isolated phone-flash and point sources.
Each source gets no mask, a 50x50 patterned mask, and a no-hole control.

HOW TO RUN (headless; cv2 and shapely are required in Blender's Python):
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tests\t17_shadow_planes.py

OUTPUT (out/):
    t17_shadow_00..05.png + .txt + _viz.png
    t17_shadow_contact.png
    t17_shadow_report.json

REPORT BACK: attach the contact sheet and report, and paste the console. PASS if each
patterned tile shows broad, softly edged shadow regions with fine breakup; each no-hole
control attenuates light across the visible scene without becoming fully black; no mask
mesh appears in camera; and labels are identical in all six views.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import bpy
import cv2

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


SCENE_SEED = 20260727
MASK_SEED = 170117
CAMERA_CONFIG = C.CameraConfig(
    focal_mm=35.0,
    offaxis_deg=22.0,
    orbit_deg=35.0,
    dof_enabled=True,
    aperture_fstop=5.6,
)
CASES = tuple((source, mode) for source in ("spotlight", "point")
              for mode in ("none", "pattern", "solid"))


def _lighting_config(source: str, mode: str) -> C.LightingConfig:
    seed = MASK_SEED if mode != "none" else None
    spotlight = source == "spotlight"
    points = []
    if source == "point":
        points.append(C.PointLightConfig(
            color_temp=5200.0,
            intensity=14.0,
            position=[0.04, -0.03, 0.40],
            shadow_mask_seed=seed,
        ))
    return C.LightingConfig(
        sun_angle_deg=[55.0, 20.0],
        sun_energy=C.SUN_ENERGY_RANGE[1],
        spotlight_beside_camera=spotlight,
        point_lights=points,
        spotlight_shadow_mask_seed=seed if source == "spotlight" else None,
    )


def _isolate_source(rig, source: str) -> None:
    rig.sun.data.energy = 0.0
    if rig.spotlight is not None and source != "spotlight":
        rig.spotlight.data.energy = 0.0
    for point in rig.points:
        if source != "point":
            point.data.energy = 0.0


def _assert_mask(mask, source: str, mode: str):
    assert mask["tcg_shadow_source"] == source
    assert mask["tcg_shadow_grid_faces"] == 50
    assert mask["tcg_shadow_candidate_faces"] == 2500
    assert len(mask.data.vertices) == 2601
    retained = int(mask["tcg_shadow_retained_faces"])
    assert len(mask.data.polygons) == retained
    if mode == "solid":
        assert retained == 2500 and mask["tcg_shadow_solid_control"]
    else:
        assert 0 < retained < 2500 and not mask["tcg_shadow_solid_control"]
    assert mask.visible_camera is False
    assert mask.visible_shadow is True
    assert abs(float(mask["tcg_shadow_opacity"])
               - phase5_lighting.SHADOW_OPACITY) < 1e-6
    assert abs(float(mask.active_material["tcg_shadow_opacity"])
               - phase5_lighting.SHADOW_OPACITY) < 1e-6
    assert 0.0 < float(mask["tcg_shadow_source_fraction"]) < 1.0
    if source == "spotlight":
        assert mask["tcg_shadow_placement"] == "camera_invisible_fallback"
    return retained


def _write_contact_sheet(out_dir, views):
    tiles = []
    for view in views:
        image = cv2.imread(view["image"])
        if image is None:
            raise RuntimeError(f"Could not read {view['image']!r}")
        tile = cv2.resize(image, (416, 416), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (416, 43), (12, 12, 12), -1)
        line1 = f"{view['index']:02d} {view['source']} / {view['mode']}"
        line2 = (f"faces={view['retained_faces']} soft={view['light_shadow_soft_size_m']:.3f} "
                 f"placement={view['placement'] or 'none'}")
        cv2.putText(tile, line1, (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(tile, line2, (8, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.37, (215, 225, 235), 1, cv2.LINE_AA)
        tiles.append(tile)
    rows = [cv2.hconcat(tiles[index:index + 3]) for index in range(0, len(tiles), 3)]
    contact = cv2.vconcat(rows)
    path = os.path.join(out_dir, "t17_shadow_contact.png")
    if not cv2.imwrite(path, contact):
        raise RuntimeError(f"Could not write contact sheet {path!r}")
    return path


def main():
    if tuple(bpy.app.version) != (5, 0, 0):
        raise RuntimeError(f"t17 requires Blender 5.0.0, got {bpy.app.version_string}")
    if CardLibrary is None:
        raise RuntimeError("[t17] cardsource unavailable") from _CARD_IMPORT_ERROR
    library = CardLibrary()
    if library.is_empty():
        raise RuntimeError("[t17] no card images found")

    import numpy as np

    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    cache = os.path.join(out_dir, "card_cache")
    os.makedirs(cache, exist_ok=True)
    for filename in os.listdir(out_dir):
        if filename.startswith("t17_shadow_"):
            os.remove(os.path.join(out_dir, filename))

    scene_cfg = C.sample_scene_config(
        {"layouts": ["table"], "back_to_camera_prob": 0.0,
         "lighting": {"occluders": False}},
        SCENE_SEED,
        max_cards=1,
    )
    layout_rng = np.random.default_rng(SCENE_SEED)
    sc.reset_scene()
    sc.setup_world(gray=0.0)
    scene = bpy.context.scene
    setup_render(scene, verbose=True)
    instances = layouts.build_table(
        scene_cfg, library, cache, layout_rng,
        allow_overlap=config.load_layout_params("table")["allow_overlap"])
    bpy.context.view_layer.update()
    target, extent = phase5_camera.subject_target_and_extent(instances)
    camera = phase5_camera.build_camera(CAMERA_CONFIG, target, extent)
    results = label_scene(scene, camera, instances)
    phase5_camera.focus_on_labeled_card(camera, CAMERA_CONFIG, results)
    labels = [label for _instance, label, _reason in results if label is not None]
    assert labels, "t17 requires a visible labeled card"
    label_signature = tuple(label.to_line() for label in labels)

    views = []
    rig = None
    for index, (source, mode) in enumerate(CASES):
        phase5_lighting.remove_light_rig(rig)
        lighting_cfg = _lighting_config(source, mode)
        scene_cfg.camera = CAMERA_CONFIG
        scene_cfg.lighting = lighting_cfg
        C.validate_scene_config(scene_cfg)
        rig = phase5_lighting.build_lighting(
            lighting_cfg, camera, target, extent,
            solid_shadow_masks=(mode == "solid"))
        _isolate_source(rig, source)
        bpy.context.view_layer.update()

        expected_masks = int(mode != "none")
        assert len(rig.occluders) == expected_masks
        retained = 0
        placement = None
        if rig.occluders:
            retained = _assert_mask(rig.occluders[0], source, mode)
            placement = rig.occluders[0]["tcg_shadow_placement"]
        source_light = rig.spotlight if source == "spotlight" else rig.points[0]
        soft_size = float(source_light.data.shadow_soft_size)
        if source == "spotlight":
            expected_soft_size = 0.012 if mode == "none" else 0.0185
        else:
            expected_soft_size = (max(0.005, extent * 0.025) if mode == "none"
                                  else (max(0.005, extent * 0.025)
                                        + max(0.012, extent * 0.05)) / 2.0)
        assert abs(soft_size - expected_soft_size) < 1e-6

        current_results = label_scene(scene, camera, instances)
        current_labels = [label for _instance, label, _reason in current_results
                          if label is not None]
        assert tuple(label.to_line() for label in current_labels) == label_signature

        stem = f"t17_shadow_{index:02d}"
        image_path = os.path.join(out_dir, stem + ".png")
        label_path = os.path.join(out_dir, stem + ".txt")
        scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)
        write_poly_label_file(label_path, current_labels)
        visualization_path = os.path.join(out_dir, stem + "_viz.png")
        visualize_poly_label_file(image_path, label_path, visualization_path)

        view = {
            "index": index,
            "source": source,
            "mode": mode,
            "retained_faces": retained,
            "retained_ratio": retained / 2500.0,
            "placement": placement,
            "source_fraction": (float(rig.occluders[0]["tcg_shadow_source_fraction"])
                                if rig.occluders else None),
            "light_shadow_soft_size_m": soft_size,
            "label_count": len(current_labels),
            "image": image_path,
            "label": label_path,
            "visualization": visualization_path,
        }
        views.append(view)
        print(f"[t17] {index:02d} source={source:9} mode={mode:7} "
              f"faces={retained:4d} placement={placement or 'none'}")

    phase5_lighting.remove_light_rig(rig)
    rig = None
    leaked = [obj.name for obj in bpy.data.objects
              if obj.get("tcg_phase5_environment", False)]
    assert not leaked, f"generated lighting objects leaked after cleanup: {leaked}"

    contact_path = _write_contact_sheet(out_dir, views)
    report = {
        "blender_version": bpy.app.version_string,
        "scene_seed": SCENE_SEED,
        "mask_seed": MASK_SEED,
        "grid_faces": 50,
        "candidate_faces": 2500,
        "coarse_frequency": 2.0,
        "fine_frequency": 12.0,
        "coarse_weight": 0.65,
        "fine_weight": 0.35,
        "remove_above_brightness": 0.50,
        "shadow_opacity": phase5_lighting.SHADOW_OPACITY,
        "target": [float(value) for value in target],
        "frame_extent_m": float(extent),
        "labels_identical": True,
        "views": [{key: value for key, value in view.items()
                   if key not in {"image", "label", "visualization"}}
                  for view in views],
    }
    report_path = os.path.join(out_dir, "t17_shadow_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\n[t17] PASS: wrote {contact_path} and {report_path}")
    print("[t17] Review patterned breakup, no-hole attenuation, and absent mask geometry.")


if __name__ == "__main__":
    main()
