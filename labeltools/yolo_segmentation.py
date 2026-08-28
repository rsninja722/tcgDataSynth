"""Standard YOLO segmentation export paired with instance metadata."""
from __future__ import annotations

import math
import os
from typing import Sequence

from labeltools.yolo_pose import HOLO_TAGS, PolyLabel, _fmt


def _validate(label: PolyLabel) -> None:
    if len(label.points) < 3:
        raise ValueError("YOLO segmentation polygons require at least three points")
    if any(char in label.card_id for char in "|\r\n"):
        raise ValueError("card_id cannot contain a pipe or newline")
    if label.holo_tag not in HOLO_TAGS:
        raise ValueError(f"Invalid holo tag {label.holo_tag!r}")
    for x, y, _flag in label.points:
        if not math.isfinite(x) or not math.isfinite(y) or not (0.0 <= x <= 1.0) \
                or not (0.0 <= y <= 1.0):
            raise ValueError("YOLO segmentation coordinates must be finite and normalized")
    area_twice = abs(sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(label.points, label.points[1:] + label.points[:1])))
    if area_twice <= 1e-12:
        raise ValueError("YOLO segmentation polygons must have non-zero area")


def to_yolo_segmentation_line(label: PolyLabel) -> str:
    """Serialize one polygon as ``0 x1 y1 ... xn yn``.

    YOLO derives the instance bounding box from the polygon envelope, which is the
    same min/max envelope stored explicitly by the custom label.
    """
    _validate(label)
    parts = ["0"]
    for x, y, _flag in label.points:
        parts.extend((_fmt(x), _fmt(y)))
    return " ".join(parts)


def to_extra_label_line(label: PolyLabel) -> str:
    """Serialize matching identity metadata without changing instance order."""
    _validate(label)
    return f"{label.card_id}|{label.holo_tag}"


def write_yolo_segmentation_files(label_path: str, extra_label_path: str,
                                  labels: Sequence[PolyLabel]) -> None:
    """Write synchronized YOLO and metadata files in one ordered pass."""
    os.makedirs(os.path.dirname(os.path.abspath(label_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(extra_label_path)), exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as label_file, \
            open(extra_label_path, "w", encoding="utf-8") as extra_file:
        for label in labels:
            label_file.write(to_yolo_segmentation_line(label) + "\n")
            extra_file.write(to_extra_label_line(label) + "\n")
