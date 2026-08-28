"""Headless Blender worker for one deterministic standalone-GUI output pair.

The standalone GUI launches this script in a separate Blender process for each index.
That keeps the GUI responsive and makes Pause safe: it simply waits for this worker to
finish publishing its image/label pair before starting no further worker.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import bpy
import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from blender import camera, layouts, lighting, scene_common  # noqa: E402
from blender.labeling import label_scene  # noqa: E402
from blender.render_output import render_poly_label_pair  # noqa: E402
from blender.render_setup import setup_render  # noqa: E402
from rules import combinations  # noqa: E402
from rules.generation import (append_completed_pair, append_refraction_failures,  # noqa: E402
                              pair_paths, resume_next_index)
from texturegen.cardsource import CardLibrary  # noqa: E402
from texturegen.table_texture import discover_table_textures  # noqa: E402


def _output_layout(root: str | None) -> config.OutputLayout:
    return config.OutputLayout(
        root=os.path.abspath(root or os.path.join(_ROOT, config.OUTPUT.root)),
        images_subdir=config.OUTPUT.images_subdir,
        labels_subdir=config.OUTPUT.labels_subdir,
        yolo_labels_subdir=config.OUTPUT.yolo_labels_subdir,
        extra_labels_subdir=config.OUTPUT.extra_labels_subdir,
        manifest_name=config.OUTPUT.manifest_name,
        refraction_failures_name=config.OUTPUT.refraction_failures_name,
    )


def _build_layout(scene_cfg, library, cache_dir: str, rng, enabled_options,
                  config_path: str | None, table_texture_paths):
    params = config.load_layout_params(scene_cfg.layout.kind, config_path)
    if scene_cfg.layout.kind == "table":
        instances = layouts.build_table(
            scene_cfg, library, cache_dir, rng, allow_overlap=params["allow_overlap"],
            table_texture_paths=table_texture_paths)
        return instances, 0.35, (0.0, 0.0, 0.0)
    if scene_cfg.layout.kind == "floating":
        instances = layouts.build_floating(
            scene_cfg, library, cache_dir, rng, allow_overlap=params["allow_overlap"],
            max_shapes=params["max_shapes"])
        return instances, 0.55, (0.0, 0.0, -0.03)
    if scene_cfg.layout.kind == "binder":
        instances, extent = layouts.build_binder(
            scene_cfg, library, cache_dir, rng, table_texture_paths=table_texture_paths)
        return instances, extent, (0.0, 0.0, 0.0)
    if scene_cfg.layout.kind == "display_case":
        instances, extent = layouts.build_display_case(
            scene_cfg, library, cache_dir, rng, enabled_options=enabled_options,
            table_texture_paths=table_texture_paths)
        return instances, extent, (0.0, 0.0, 0.0)
    if scene_cfg.layout.kind == "hand":
        instances, extent = layouts.build_hand(
            scene_cfg, library, cache_dir, rng, table_texture_paths=table_texture_paths)
        return instances, extent, (0.0, 0.0, 0.0)
    if scene_cfg.layout.kind == "stack":
        instances, extent = layouts.build_stack(
            scene_cfg, library, cache_dir, rng, table_texture_paths=table_texture_paths)
        protection = scene_cfg.cards[-1].protection
        thickness = layouts.sb.stack_thickness(protection)
        half = thickness / 2.0
        target_z = (float(scene_cfg.layout.params["table_clearance_m"]) + half
                    + (len(scene_cfg.cards) - 1) * (thickness + 0.0001))
        return instances, extent, (0.0, 0.0, target_z)
    raise ValueError(f"Unsupported layout {scene_cfg.layout.kind!r}")


def run_one(index: int, config_path: str | None = None, output_root: str | None = None) -> dict:
    """Build, render, post-process, label, and manifest exactly one output index."""
    if tuple(bpy.app.version) != (5, 0, 0):
        raise RuntimeError(f"Generation worker requires Blender 5.0.0, got {bpy.app.version_string}")
    settings = config.load_generation_settings(config_path)
    output = _output_layout(output_root)
    export_yolo = bool(settings["export_yolo_segmentation"])
    next_index = resume_next_index(
        output, settings["base_seed"], settings["count"],
        require_yolo_segmentation=export_yolo)
    if index != next_index:
        raise RuntimeError(f"Worker received index {index}; resume state requires {next_index}")
    if index >= settings["count"]:
        raise RuntimeError("Requested pair count is already complete")
    paths = pair_paths(output, settings["base_seed"], index)
    scene_rng = np.random.default_rng(paths.seed)
    scene_cfg = combinations.sample_scene_config(
        settings["enabled_options"], paths.seed, config_path=config_path, rng=scene_rng)
    library = CardLibrary()
    if library.is_empty():
        raise RuntimeError("No card images found; set TCG_CARD_IMAGE_ROOT before generation")
    table_texture_paths = []
    if scene_cfg.layout.kind != "floating":
        texture_dir = config.load_table_texture_dir(config_path)
        if not texture_dir:
            raise RuntimeError(
                "Set table_texture_dir in config.json before generating a table-bearing layout")
        table_texture_paths = discover_table_textures(texture_dir)
    cache_dir = os.path.join(output.root, "card_cache")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        scene_common.reset_scene()
        scene_common.setup_world(gray=0.025)
        scene = bpy.context.scene
        setup_render(scene, verbose=True)
        instances, known_extent, empty_target = _build_layout(
            scene_cfg, library, cache_dir, scene_rng, settings["enabled_options"],
            config_path, table_texture_paths)
        bpy.context.view_layer.update()
        if instances:
            target, focus_instance = camera.random_focus_target(instances, scene_rng)
            measured_extent = camera.subject_extent_from_target(instances, target)
            extent = max(float(known_extent or 0.0), measured_extent)
        else:
            target, focus_instance = empty_target, None
            extent = float(known_extent)
        cam = camera.build_camera(scene_cfg.camera, target, extent)
        if focus_instance is not None:
            camera.focus_on_card(cam, scene_cfg.camera, focus_instance)
            camera.zoom_to_card_boundary(scene, cam, instances, scene_rng)
        lighting.build_lighting(scene_cfg.lighting, cam, target, extent)
        refraction_failures = []
        results = label_scene(
            scene, cam, instances, refraction_failures=refraction_failures) if instances else []
        for failure in refraction_failures:
            failure["layout"] = scene_cfg.layout.kind
        remove_outside = config.load_layout_params(
            scene_cfg.layout.kind, config_path)["out_of_frustum"] == "remove"
        labels = []
        for instance, label, reason in results:
            if label is not None:
                labels.append(label)
            elif remove_outside and reason == "fully-out-of-frustum":
                for obj in instance.objects:
                    obj.hide_render = True
        bpy.context.view_layer.update()
        render_poly_label_pair(
            scene, paths.image_path, paths.label_path, labels, scene_cfg.postfx,
            yolo_label_path=paths.yolo_label_path if export_yolo else None,
            extra_label_path=paths.extra_label_path if export_yolo else None)
        failure_path = append_refraction_failures(output, paths, refraction_failures)
        if refraction_failures:
            print(f"[generation_worker] WARNING used direct polygons for "
                  f"{len(refraction_failures)} card(s) with unsolved refraction; "
                  f"details: {failure_path}")
        append_completed_pair(
            output, paths, scene_cfg.to_dict(), require_yolo_segmentation=export_yolo)
        return {
            "index": index,
            "seed": paths.seed,
            "stem": paths.stem,
            "image": paths.image_path,
            "label": paths.label_path,
            "yolo_label": paths.yolo_label_path if export_yolo else None,
            "extra_label": paths.extra_label_path if export_yolo else None,
            "label_count": len(labels),
            "layout": scene_cfg.layout.kind,
            "cardless": scene_cfg.cardless,
            "refraction_failure_count": len(refraction_failures),
            "refraction_failures": failure_path,
        }
    finally:
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[sys.argv.index("--") + 1:] if argv is None and "--" in sys.argv
                       else (argv or []))
    result = run_one(args.index, args.config, args.output_root)
    print("[generation_worker] PASS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
