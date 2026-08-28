r"""Regression acceptance for the missing stack label at production seed 867779.

HOW TO RUN (headless; card images, cv2, and shapely are required):
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tests\t23_stack_yolo_seed.py

The second-highest card at this seed has a mean camera depth nearer than the physical
top card because of its lateral offset. It must remain render-only and must not suppress
the top card from either the custom or YOLO segmentation labels.
"""
from __future__ import annotations

import json
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
from blender.generation_worker import run_one  # noqa: E402
from labeltools.yolo_pose import parse_poly_line  # noqa: E402
from rules import combinations  # noqa: E402


SEED = 867779


def main() -> None:
    if tuple(bpy.app.version) != (5, 0, 0):
        raise RuntimeError(f"t23 requires Blender 5.0.0, got {bpy.app.version_string}")

    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    test_root = os.path.join(out_dir, "t23_stack_yolo_seed")
    if os.path.isdir(test_root):
        shutil.rmtree(test_root)
    os.makedirs(test_root)

    texture_dir = os.path.join(test_root, "texture_sources")
    os.makedirs(texture_dir)
    texture_path = os.path.join(texture_dir, "table.png")
    assert cv2.imwrite(texture_path, np.full((64, 64, 3), (48, 96, 144), dtype=np.uint8))

    values = config.load_config()
    values["table_texture_dir"] = texture_dir
    values["generation"]["count"] = 1
    values["generation"]["base_seed"] = SEED
    values["generation"]["export_yolo_segmentation"] = True
    values["generation"]["enabled_options"] = combinations.default_enabled_options()
    config_path = os.path.join(test_root, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2)

    scene_cfg = combinations.sample_scene_config(
        values["generation"]["enabled_options"], SEED, config_path=config_path)
    assert scene_cfg.layout.kind == "stack"
    assert len(scene_cfg.cards) == 10
    result = run_one(0, config_path=config_path, output_root=test_root)
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
    assert yolo[1:] == [f"{value:.6f}" for point in custom.points for value in point[:2]]
    assert extra_lines[0] == f"{custom.card_id}|{custom.holo_tag}"
    print(f"[t23] PASS: seed {SEED} emitted one synchronized stack label")


if __name__ == "__main__":
    main()
