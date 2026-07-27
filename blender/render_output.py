"""Atomic render/image-label output pipeline (bpy REQUIRED).

The Blender render is staged, post-processed, and paired with its custom polygon
label before either final output is published. Labels are published first so a failed
publish can never leave a final image without its matching label.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import bpy

from labeltools.yolo_pose import PolyLabel, write_poly_label_file
from postfx.effects import apply_postfx_file
from rules.combinations import PostFxConfig


def _stage_path(path: str, role: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}.postfx-{role}{ext}"


def _remove_if_exists(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)


@dataclass
class RenderPairStage:
    """Staged render paths retained while Blender's asynchronous render runs."""
    image_path: str
    label_path: str
    raw_path: str
    staged_image: str
    staged_label: str
    labels: Sequence[PolyLabel]
    postfx: PostFxConfig
    previous_filepath: str


def stage_poly_label_pair(scene, image_path: str, label_path: str,
                          labels: Sequence[PolyLabel], postfx: PostFxConfig) -> RenderPairStage:
    """Prepare a new staged pair and direct Blender's next render to its raw image."""
    image_path = os.path.abspath(image_path)
    label_path = os.path.abspath(label_path)
    if os.path.exists(image_path) or os.path.exists(label_path):
        raise FileExistsError(
            "Refusing to overwrite an existing output pair; choose a new image stem: "
            f"{image_path!r}, {label_path!r}")
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    raw_path = _stage_path(image_path, "raw")
    staged_image = _stage_path(image_path, "image")
    staged_label = _stage_path(label_path, "label")
    for path in (raw_path, staged_image, staged_label):
        _remove_if_exists(path)
    stage = RenderPairStage(
        image_path, label_path, raw_path, staged_image, staged_label, tuple(labels), postfx,
        scene.render.filepath)
    scene.render.filepath = raw_path
    return stage


def discard_staged_pair(scene, stage: RenderPairStage) -> None:
    """Restore output settings and remove incomplete render/post-processing files."""
    scene.render.filepath = stage.previous_filepath
    for path in (stage.raw_path, stage.staged_image, stage.staged_label):
        _remove_if_exists(path)


def publish_staged_pair(scene, stage: RenderPairStage) -> None:
    """Post-process the completed raw render, then publish its label and image pair."""
    try:
        if not os.path.isfile(stage.raw_path):
            raise RuntimeError(f"Blender did not write staged render {stage.raw_path!r}")
        apply_postfx_file(stage.raw_path, stage.staged_image, stage.postfx)
        write_poly_label_file(stage.staged_label, stage.labels)
        if not os.path.isfile(stage.staged_image) or not os.path.isfile(stage.staged_label):
            raise RuntimeError("Could not stage the post-processed image/label pair")
        # Publishing the label first preserves the no-image-without-label invariant.
        os.replace(stage.staged_label, stage.label_path)
        os.replace(stage.staged_image, stage.image_path)
    finally:
        discard_staged_pair(scene, stage)


def render_poly_label_pair(scene, image_path: str, label_path: str,
                           labels: Sequence[PolyLabel], postfx: PostFxConfig) -> None:
    """Render and publish one processed PNG/custom-label pair.

    ``image_path`` and ``label_path`` must both be new output names. The raw Blender
    render and processed image are staged beside the final image; no final image is
    written until post-processing and label serialization have both succeeded.
    """
    stage = stage_poly_label_pair(scene, image_path, label_path, labels, postfx)
    try:
        bpy.ops.render.render(write_still=True)
        publish_staged_pair(scene, stage)
    except Exception:
        discard_staged_pair(scene, stage)
        raise
