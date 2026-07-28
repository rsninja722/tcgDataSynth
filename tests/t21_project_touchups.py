r"""Project-wide touch-up acceptance: grid jitter, camera framing, and shadows.

HOW TO RUN (headless; cv2 and shapely are required in Blender's Python):
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tests\t21_project_touchups.py

OUTPUT (out/):
    t21_touchups_binder.png + .txt + _viz.png
    t21_touchups_display_case.png + .txt + _viz.png
    t21_touchups_contact.png
    t21_touchups_report.json

PASS if the report assertions succeed, both grids have subtle nonuniform placement,
the selected focus card is centered and sharp, framing is near a card boundary, and
the patterned shadow has visibly softer edges without becoming fully opaque.
"""
from __future__ import annotations

import json
import math
import os
import sys

import bpy
import cv2
import numpy as np
from bpy_extras.object_utils import world_to_camera_view

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from blender import camera, layouts, lighting, scene_common  # noqa: E402
from blender.labeling import label_scene  # noqa: E402
from blender.render_setup import setup_render  # noqa: E402
from labeltools.visualize import visualize_poly_label_file  # noqa: E402
from labeltools.yolo_pose import write_poly_label_file  # noqa: E402
from rules import combinations as C  # noqa: E402
from texturegen.cardsource import CardLibrary  # noqa: E402

MASK_OPACITY = 0.42


def _sample(kind: str, minimum_cards: int):
    options = {"layouts": [kind], "back_to_camera_prob": 0.0}
    for seed in range(20260800, 20261800):
        probe = C.sample_scene_config(options, seed)
        if len(probe.cards) >= minimum_cards:
            rng = np.random.default_rng(seed)
            return C.sample_scene_config(options, seed, rng=rng), rng
    raise RuntimeError(f"Could not sample a populated {kind} scene")


def _assert_jitter(instances):
    values = []
    for instance in instances:
        dx = float(instance.root["tcg_grid_jitter_x_mm"])
        dy = float(instance.root["tcg_grid_jitter_y_mm"])
        rotation = float(instance.root["tcg_grid_jitter_rotation_deg"])
        assert -2.0 <= dx <= 2.0 and -2.0 <= dy <= 2.0
        assert -1.0 <= rotation <= 1.0
        values.append([dx, dy, rotation])
    assert any(any(abs(value) > 1e-4 for value in row) for row in values)
    return values


def _build_view(kind: str, library, cache_dir: str, out_dir: str):
    cfg, rng = _sample(kind, minimum_cards=4)
    if kind == "display_case":
        assert cfg.camera.offaxis_deg <= 30.0
    cfg.camera.aperture_fstop = 2.4
    cfg.lighting = C.LightingConfig(
        sun_angle_deg=[55.0, 20.0],
        sun_energy=C.SUN_ENERGY_RANGE[1],
        spotlight_beside_camera=True,
        point_lights=[],
        spotlight_shadow_mask_seed=210021,
        shadow_plane_opacity=MASK_OPACITY,
    )
    if kind == "display_case":
        cfg.layout.params["top_card_probability"] = 0.0
    C.validate_scene_config(cfg)

    scene_common.reset_scene()
    scene_common.setup_world(gray=0.025)
    scene = bpy.context.scene
    setup_render(scene, verbose=(kind == "binder"))
    if kind == "binder":
        instances, known_extent = layouts.build_binder(cfg, library, cache_dir, rng)
    else:
        instances, known_extent = layouts.build_display_case(
            cfg, library, cache_dir, rng, enabled_options=C.default_enabled_options())
    bpy.context.view_layer.update()
    jitter = _assert_jitter(instances)

    target, focused = camera.random_focus_target(instances, rng)
    measured_extent = camera.subject_extent_from_target(instances, target)
    extent = max(float(known_extent), measured_extent)
    cam = camera.build_camera(cfg.camera, target, extent)
    camera.focus_on_card(cam, cfg.camera, focused)
    assert camera.all_cards_fully_contained(scene, cam, instances)
    initial_lens = float(cam.data.lens)
    final_lens = camera.zoom_to_card_boundary(scene, cam, instances, rng)
    rollback = int(cam["tcg_zoom_rollback"])
    crossing_lens = float(cam["tcg_zoom_crossing_mm"])
    assert rollback in (0, 1, 2)
    assert math.isclose(final_lens, crossing_lens - rollback, abs_tol=1e-6)
    assert cam.data.dof.focus_object is focused.card
    assert math.isclose(float(cam.data.dof.aperture_fstop), 2.4, abs_tol=1e-6)
    center = world_to_camera_view(scene, cam, focused.card.matrix_world.translation)
    assert math.isclose(float(center.x), 0.5, abs_tol=1e-5)
    assert math.isclose(float(center.y), 0.5, abs_tol=1e-5)

    rig = lighting.build_lighting(cfg.lighting, cam, target, extent)
    assert len(rig.occluders) == 1
    assert math.isclose(float(rig.occluders[0]["tcg_shadow_opacity"]),
                        MASK_OPACITY, abs_tol=1e-6)
    assert float(rig.spotlight.data.shadow_soft_size) > 0.0185
    assert float(rig.sun.data.angle) > math.radians(0.5)

    results = label_scene(scene, cam, instances)
    labels = [label for _instance, label, _reason in results if label is not None]
    assert labels
    stem = f"t21_touchups_{kind}"
    image_path = os.path.join(out_dir, stem + ".png")
    label_path = os.path.join(out_dir, stem + ".txt")
    visualization_path = os.path.join(out_dir, stem + "_viz.png")
    scene.render.filepath = image_path
    bpy.ops.render.render(write_still=True)
    write_poly_label_file(label_path, labels)
    visualize_poly_label_file(image_path, label_path, visualization_path)

    # Explicitly exercise the no-op branch for a scene that starts with an outlier.
    cam.data.lens = initial_lens
    instances[-1].root.location.x += extent * 10.0
    bpy.context.view_layer.update()
    assert not camera.all_cards_fully_contained(scene, cam, instances)
    assert camera.zoom_to_card_boundary(scene, cam, instances, rng) == initial_lens
    assert not bool(cam["tcg_zoom_adjusted"])
    return {
        "layout": kind,
        "seed": cfg.seed,
        "card_count": len(instances),
        "jitter": jitter,
        "focus_card_id": focused.card_id,
        "focus_ndc": [float(center.x), float(center.y)],
        "initial_focal_mm": initial_lens,
        "crossing_focal_mm": crossing_lens,
        "zoom_rollback": rollback,
        "final_focal_mm": final_lens,
        "shadow_opacity": MASK_OPACITY,
        "spot_shadow_soft_size_m": float(rig.spotlight.data.shadow_soft_size),
        "sun_angle_deg": math.degrees(float(rig.sun.data.angle)),
        "label_count": len(labels),
        "visualization": visualization_path,
    }


def main():
    if tuple(bpy.app.version) != (5, 0, 0):
        raise RuntimeError(f"t21 requires Blender 5.0.0, got {bpy.app.version_string}")
    library = CardLibrary()
    if library.is_empty():
        raise RuntimeError("[t21] no card images found")
    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    cache_dir = os.path.join(out_dir, "card_cache")
    os.makedirs(cache_dir, exist_ok=True)
    views = [_build_view(kind, library, cache_dir, out_dir)
             for kind in ("binder", "display_case")]

    tiles = []
    for view in views:
        image = cv2.imread(view["visualization"], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {view['visualization']!r}")
        tiles.append(cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA))
    contact_path = os.path.join(out_dir, "t21_touchups_contact.png")
    if not cv2.imwrite(contact_path, cv2.hconcat(tiles)):
        raise RuntimeError(f"Could not write {contact_path!r}")
    report_path = os.path.join(out_dir, "t21_touchups_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump({"blender_version": bpy.app.version_string, "views": views}, handle, indent=2)
    print(f"[t21] PASS: wrote {contact_path} and {report_path}")


if __name__ == "__main__":
    main()
