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
from labeltools.yolo_segmentation import write_yolo_segmentation_files
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
    yolo_label_path: str | None
    extra_label_path: str | None
    staged_yolo_label: str | None
    staged_extra_label: str | None
    labels: Sequence[PolyLabel]
    postfx: PostFxConfig
    previous_filepath: str


def stage_poly_label_pair(scene, image_path: str, label_path: str,
                          labels: Sequence[PolyLabel], postfx: PostFxConfig,
                          yolo_label_path: str | None = None,
                          extra_label_path: str | None = None) -> RenderPairStage:
    """Prepare a new staged pair and direct Blender's next render to its raw image."""
    image_path = os.path.abspath(image_path)
    label_path = os.path.abspath(label_path)
    if (yolo_label_path is None) != (extra_label_path is None):
        raise ValueError("YOLO segmentation and extra-label paths must be supplied together")
    yolo_label_path = os.path.abspath(yolo_label_path) if yolo_label_path else None
    extra_label_path = os.path.abspath(extra_label_path) if extra_label_path else None
    final_paths = [image_path, label_path]
    final_paths.extend(path for path in (yolo_label_path, extra_label_path) if path)
    if any(os.path.exists(path) for path in final_paths):
        raise FileExistsError(
            "Refusing to overwrite an existing output pair; choose a new image stem: "
            f"{final_paths!r}")
    for path in final_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_path = _stage_path(image_path, "raw")
    staged_image = _stage_path(image_path, "image")
    staged_label = _stage_path(label_path, "label")
    staged_yolo = _stage_path(yolo_label_path, "yolo") if yolo_label_path else None
    staged_extra = _stage_path(extra_label_path, "extra") if extra_label_path else None
    staged_paths = [raw_path, staged_image, staged_label]
    staged_paths.extend(path for path in (staged_yolo, staged_extra) if path)
    for path in staged_paths:
        _remove_if_exists(path)
    stage = RenderPairStage(
        image_path, label_path, raw_path, staged_image, staged_label,
        yolo_label_path, extra_label_path, staged_yolo, staged_extra,
        tuple(labels), postfx, scene.render.filepath)
    scene.render.filepath = raw_path
    return stage


def discard_staged_pair(scene, stage: RenderPairStage) -> None:
    """Restore output settings and remove incomplete render/post-processing files."""
    scene.render.filepath = stage.previous_filepath
    paths = [stage.raw_path, stage.staged_image, stage.staged_label]
    paths.extend(path for path in (stage.staged_yolo_label, stage.staged_extra_label) if path)
    for path in paths:
        _remove_if_exists(path)


def publish_staged_pair(scene, stage: RenderPairStage) -> None:
    """Post-process the completed raw render, then publish its label and image pair."""
    try:
        if not os.path.isfile(stage.raw_path):
            raise RuntimeError(f"Blender did not write staged render {stage.raw_path!r}")
        apply_postfx_file(stage.raw_path, stage.staged_image, stage.postfx)
        write_poly_label_file(stage.staged_label, stage.labels)
        if stage.staged_yolo_label and stage.staged_extra_label:
            write_yolo_segmentation_files(
                stage.staged_yolo_label, stage.staged_extra_label, stage.labels)
        required = [stage.staged_image, stage.staged_label]
        required.extend(path for path in (
            stage.staged_yolo_label, stage.staged_extra_label) if path)
        if not all(os.path.isfile(path) for path in required):
            raise RuntimeError("Could not stage the post-processed image/label pair")
        # Publishing the image last preserves the no-image-without-label invariant.
        # Roll back newly published labels if one of the subsequent replaces fails.
        published = []
        try:
            os.replace(stage.staged_label, stage.label_path)
            published.append(stage.label_path)
            if stage.staged_yolo_label and stage.yolo_label_path:
                os.replace(stage.staged_yolo_label, stage.yolo_label_path)
                published.append(stage.yolo_label_path)
            if stage.staged_extra_label and stage.extra_label_path:
                os.replace(stage.staged_extra_label, stage.extra_label_path)
                published.append(stage.extra_label_path)
            os.replace(stage.staged_image, stage.image_path)
        except Exception:
            for path in published:
                _remove_if_exists(path)
            raise
    finally:
        discard_staged_pair(scene, stage)


def render_poly_label_pair(scene, image_path: str, label_path: str,
                           labels: Sequence[PolyLabel], postfx: PostFxConfig,
                           yolo_label_path: str | None = None,
                           extra_label_path: str | None = None) -> None:
    """Render and publish one processed PNG/custom-label pair.

    ``image_path`` and ``label_path`` must both be new output names. The raw Blender
    render and processed image are staged beside the final image; no final image is
    written until post-processing and label serialization have both succeeded.
    """
    stage = stage_poly_label_pair(
        scene, image_path, label_path, labels, postfx,
        yolo_label_path=yolo_label_path, extra_label_path=extra_label_path)
    try:
        bpy.ops.render.render(write_still=True)
        publish_staged_pair(scene, stage)
    except Exception:
        discard_staged_pair(scene, stage)
        raise
