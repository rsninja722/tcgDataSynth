r"""Acceptance for YOLO export, table textures, stacks, negatives, and cache cleanup.

HOW TO RUN (headless; card images, cv2, and shapely are required):
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tests\t22_requested_features.py

The script renders one positive stack through the production worker, verifies all label
streams and cache cleanup, then constructs every layout in cardless mode without extra
renders. It writes ``out/t22_requested_features_report.json``.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys

import bpy
import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from blender import layouts, scene_common  # noqa: E402
from blender.generation_worker import _build_layout, run_one  # noqa: E402
from labeltools.yolo_pose import parse_poly_line  # noqa: E402
from rules import combinations as C  # noqa: E402
from rules.generation import resume_next_index  # noqa: E402
from texturegen.cardsource import CardLibrary  # noqa: E402


def _write_texture_sources(directory: str) -> list[str]:
    os.makedirs(directory, exist_ok=True)
    colors = ((32, 64, 192), (64, 192, 32), (192, 32, 64), (180, 160, 40))
    paths = []
    for index, color in enumerate(colors):
        path = os.path.join(directory, f"texture_{index}.png")
        assert cv2.imwrite(path, np.full((64, 64, 3), color, dtype=np.uint8))
        paths.append(path)
    return paths


def _base_test_config(texture_dir: str) -> dict:
    with open(os.path.join(_ROOT, config.CONFIG_FILENAME), encoding="utf-8") as handle:
        values = json.load(handle)
    values["table_texture_dir"] = texture_dir
    generation = values["generation"]
    generation["count"] = 1
    generation["export_yolo_segmentation"] = True
    options = generation["enabled_options"]
    options["layouts"] = ["stack"]
    options["protections"] = ["sleeve"]
    options["cardless_scene_prob"] = 0.0
    options["back_to_camera_prob"] = 0.0
    options["post_effects"] = []
    return values


def _stack_seed(values: dict, config_path: str) -> int:
    options = values["generation"]["enabled_options"]
    for seed in range(20260828, 20261828):
        cfg = C.sample_scene_config(options, seed, config_path=config_path)
        if len(cfg.cards) >= 4 and cfg.layout.params["with_hand"]:
            return seed
    raise RuntimeError("Could not find a multi-card stack seed with a hand")


def _assert_label_exports(result: dict) -> None:
    assert result["label_count"] == 1
    with open(result["label"], encoding="utf-8") as handle:
        custom_lines = handle.read().splitlines()
    with open(result["yolo_label"], encoding="utf-8") as handle:
        yolo_lines = handle.read().splitlines()
    with open(result["extra_label"], encoding="utf-8") as handle:
        extra_lines = handle.read().splitlines()
    assert len(custom_lines) == len(yolo_lines) == len(extra_lines) == 1
    custom = parse_poly_line(custom_lines[0])
    yolo = yolo_lines[0].split()
    assert yolo[0] == "0"
    expected = [f"{value:.6f}" for point in custom.points for value in point[:2]]
    assert yolo[1:] == expected
    assert extra_lines[0] == f"{custom.card_id}|{custom.holo_tag}"


def _assert_stack_scene(scene_cfg) -> dict:
    cards = sorted(
        (obj for obj in bpy.data.objects if "card_id" in obj),
        key=lambda obj: int(obj.name.split("Card", 1)[1].split("_", 1)[0]))
    assert len(cards) == len(scene_cfg.cards)
    assert [bool(card["tcg_label_enabled"]) for card in cards] \
        == [False] * (len(cards) - 1) + [True]
    thickness = layouts.sb.stack_thickness(scene_cfg.cards[0].protection)
    z_values = [float(card.location.z) for card in cards]
    for lower, upper in zip(z_values, z_values[1:]):
        assert math.isclose(upper - lower, thickness + 0.0001, abs_tol=1e-7)
    material = bpy.data.objects["StackTable"].data.materials[0]
    mode = material["tcg_table_texture_mode"]
    assert mode in {"single", "quad"}
    assert any(obj.name.startswith("StackHand") for obj in bpy.data.objects)
    return {"card_count": len(cards), "z_values": z_values, "table_mode": mode}


def _assert_both_table_modes(texture_paths: list[str]) -> dict:
    seen = {}
    for seed in range(100):
        material = layouts._table_material(
            f"T22TableProbe{seed}", np.random.default_rng(seed), texture_paths)
        seen.setdefault(material["tcg_table_texture_mode"], material)
        if set(seen) == {"single", "quad"}:
            break
    assert set(seen) == {"single", "quad"}
    assert sum(node.bl_idname == "ShaderNodeTexImage"
               for node in seen["single"].node_tree.nodes) == 1
    assert sum(node.bl_idname == "ShaderNodeTexImage"
               for node in seen["quad"].node_tree.nodes) == 4
    assert math.isclose(float(seen["quad"]["tcg_table_texture_seam_overlap"]), 0.05)
    return {mode: material["tcg_table_texture_paths"].split("|")
            for mode, material in seen.items()}


def _assert_cardless_layouts(config_path: str, texture_paths: list[str],
                             cache_dir: str, library) -> dict:
    reports = {}
    overrides = {
        "table": {"protections": ["sleeve"]},
        "floating": {"protections": ["sleeve"]},
        "binder": {"protections": ["toploader"], "binder_contents": ["toploader"]},
        "display_case": {"protections": ["toploader"]},
        "hand": {"protections": ["sleeve"]},
        "stack": {"protections": ["sleeve"]},
    }
    for index, layout in enumerate(C.LAYOUTS):
        options = C.default_enabled_options()
        options.update(overrides[layout])
        options["layouts"] = [layout]
        options["cardless_scene_prob"] = 1.0
        seed = 20262000 + index
        rng = np.random.default_rng(seed)
        cfg = C.sample_scene_config(options, seed, config_path=config_path, rng=rng)
        assert cfg.cardless
        scene_common.reset_scene()
        scene_common.setup_world(gray=0.025)
        instances, extent, _target = _build_layout(
            cfg, library, cache_dir, rng, options, config_path,
            [] if layout == "floating" else texture_paths)
        assert instances == []
        assert not any("card_id" in obj for obj in bpy.data.objects)
        reports[layout] = {
            "planned_card_count": len(cfg.cards),
            "render_object_count": len(bpy.data.objects),
            "frame_extent_m": extent,
        }
    return reports


def main() -> None:
    if tuple(bpy.app.version) != (5, 0, 0):
        raise RuntimeError(f"t22 requires Blender 5.0.0, got {bpy.app.version_string}")
    library = CardLibrary()
    if library.is_empty():
        raise RuntimeError("t22 requires TCG_CARD_IMAGE_ROOT with card images")

    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    test_root = os.path.join(out_dir, "t22_requested_features")
    if os.path.isdir(test_root):
        shutil.rmtree(test_root)
    os.makedirs(test_root)
    texture_dir = os.path.join(test_root, "texture_sources")
    texture_paths = _write_texture_sources(texture_dir)
    config_path = os.path.join(test_root, "config.json")
    values = _base_test_config(texture_dir)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2)
    seed = _stack_seed(values, config_path)
    values["generation"]["base_seed"] = seed
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2)

    expected_cfg = C.sample_scene_config(
        values["generation"]["enabled_options"], seed, config_path=config_path)
    result = run_one(0, config_path=config_path, output_root=test_root)
    _assert_label_exports(result)
    stack_report = _assert_stack_scene(expected_cfg)
    output = config.OutputLayout(root=test_root)
    assert resume_next_index(
        output, seed, 1, require_yolo_segmentation=True) == 1
    assert not os.path.exists(os.path.join(test_root, "card_cache"))

    modes = _assert_both_table_modes(texture_paths)
    cache_dir = os.path.join(test_root, "cardless_cache")
    os.makedirs(cache_dir)
    negatives = _assert_cardless_layouts(
        config_path, texture_paths, cache_dir, library)
    shutil.rmtree(cache_dir)

    report = {
        "blender_version": bpy.app.version_string,
        "worker": result,
        "stack": stack_report,
        "texture_modes": modes,
        "cardless_layouts": negatives,
        "card_cache_removed": True,
    }
    report_path = os.path.join(out_dir, "t22_requested_features_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[t22] PASS: wrote {report_path}")


if __name__ == "__main__":
    main()
