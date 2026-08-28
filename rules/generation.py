"""Bpy-free deterministic output naming, manifest, and resume validation."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Optional

import config


class ResumeError(RuntimeError):
    """Raised when existing output cannot be resumed without a duplicate or gap."""


@dataclass(frozen=True)
class PairPaths:
    """Paths and deterministic identity for one image/custom-label output pair."""
    index: int
    seed: int
    stem: str
    image_path: str
    label_path: str
    yolo_label_path: str
    extra_label_path: str
    image_relpath: str
    label_relpath: str
    yolo_label_relpath: str
    extra_label_relpath: str


def seed_for_index(base_seed: int, index: int) -> int:
    """Scene seed for a zero-based output index; values are exact integers."""
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("index must be a non-negative integer")
    seed = base_seed + index
    if seed > config.GENERATION_SEED_MAX:
        raise ValueError("base_seed + index exceeds the accepted generation seed range")
    return seed


def stem_for_index(index: int, seed: int) -> str:
    """Filename stem which carries both stable index and reproducible scene seed."""
    if index < 0 or seed < 0:
        raise ValueError("index and seed must be non-negative")
    return f"{index:06d}_seed{seed}"


def pair_paths(output: config.OutputLayout, base_seed: int, index: int) -> PairPaths:
    """Return final image/label paths under the configured output root."""
    seed = seed_for_index(base_seed, index)
    stem = stem_for_index(index, seed)
    root = os.path.abspath(output.root)
    image_rel = os.path.join(output.images_subdir, stem + ".png")
    label_rel = os.path.join(output.labels_subdir, stem + ".txt")
    yolo_label_rel = os.path.join(output.yolo_labels_subdir, stem + ".txt")
    extra_label_rel = os.path.join(output.extra_labels_subdir, stem + ".txt")
    return PairPaths(
        index, seed, stem,
        os.path.join(root, image_rel), os.path.join(root, label_rel),
        os.path.join(root, yolo_label_rel), os.path.join(root, extra_label_rel),
        image_rel, label_rel, yolo_label_rel, extra_label_rel)


def _manifest_path(output: config.OutputLayout) -> str:
    return os.path.join(os.path.abspath(output.root), output.manifest_name)


def _read_manifest(output: config.OutputLayout) -> list[dict[str, Any]]:
    path = _manifest_path(output)
    if not os.path.isfile(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResumeError(f"Invalid manifest JSON at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ResumeError(f"Manifest line {line_number} is not an object")
            records.append(record)
    return records


def _validate_record(record: dict[str, Any], paths: PairPaths,
                     require_yolo_segmentation: bool = False) -> None:
    required = {
        "index": paths.index,
        "seed": paths.seed,
        "stem": paths.stem,
        "image": paths.image_relpath,
        "label": paths.label_relpath,
    }
    if require_yolo_segmentation:
        required.update({
            "yolo_label": paths.yolo_label_relpath,
            "extra_label": paths.extra_label_relpath,
        })
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ResumeError(
                f"Manifest record {paths.index} has {key}={record.get(key)!r}; "
                f"expected {expected!r}")


def _pair_file_stems(output: config.OutputLayout) -> dict[str, set[str]]:
    root = os.path.abspath(output.root)
    image_dir = os.path.join(root, output.images_subdir)

    def stems(directory: str, extension: str) -> set[str]:
        if not os.path.isdir(directory):
            return set()
        return {name[:-len(extension)] for name in os.listdir(directory)
                if name.endswith(extension) and os.path.isfile(os.path.join(directory, name))}

    return {
        "image": stems(image_dir, ".png"),
        "label": stems(os.path.join(root, output.labels_subdir), ".txt"),
        "yolo": stems(os.path.join(root, output.yolo_labels_subdir), ".txt"),
        "extra": stems(os.path.join(root, output.extra_labels_subdir), ".txt"),
    }


def append_completed_pair(output: config.OutputLayout, paths: PairPaths,
                          scene_config: Optional[dict[str, Any]] = None,
                          recovered: bool = False,
                          require_yolo_segmentation: bool = False) -> None:
    """Append a completed pair only after both final files are present."""
    if not os.path.isfile(paths.image_path) or not os.path.isfile(paths.label_path):
        raise ResumeError("Cannot record a manifest pair before both final files exist")
    if require_yolo_segmentation and (
            not os.path.isfile(paths.yolo_label_path)
            or not os.path.isfile(paths.extra_label_path)):
        raise ResumeError("Cannot record a YOLO pair before both export files exist")
    records = _read_manifest(output)
    if len(records) != paths.index:
        raise ResumeError(
            f"Cannot append index {paths.index}; manifest currently has {len(records)} records")
    record: dict[str, Any] = {
        "index": paths.index,
        "seed": paths.seed,
        "stem": paths.stem,
        "image": paths.image_relpath,
        "label": paths.label_relpath,
    }
    if scene_config is not None:
        record["scene_config"] = scene_config
    if require_yolo_segmentation:
        record["yolo_label"] = paths.yolo_label_relpath
        record["extra_label"] = paths.extra_label_relpath
    if recovered:
        record["recovered"] = True
    manifest_path = _manifest_path(output)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_refraction_failures(output: config.OutputLayout, paths: PairPaths,
                               failures: list[dict[str, Any]]) -> Optional[str]:
    """Append direct-fallback optical diagnostics after its output pair is published."""
    if not failures:
        return None
    if not os.path.isfile(paths.image_path) or not os.path.isfile(paths.label_path):
        raise ResumeError("Cannot record refraction failures before the output pair exists")
    output_path = os.path.abspath(output.refraction_failures_path())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as handle:
        for failure in failures:
            record = {
                "index": paths.index,
                "seed": paths.seed,
                "stem": paths.stem,
                "image": paths.image_relpath,
                "label": paths.label_relpath,
                **failure,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return output_path


def resume_next_index(output: config.OutputLayout, base_seed: int, count: int,
                      require_yolo_segmentation: bool = False) -> int:
    """Validate/recover a contiguous output prefix and return its next index.

    A completed pair that was published just before a process crash but was not yet
    appended to the manifest is recovered deterministically. Orphaned files, extra
    stems, and gaps fail explicitly instead of risking duplicate output.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    expected = [pair_paths(output, base_seed, index) for index in range(count)]
    expected_by_stem = {paths.stem: paths for paths in expected}
    file_stems = _pair_file_stems(output)
    image_stems, label_stems = file_stems["image"], file_stems["label"]
    if image_stems != label_stems:
        raise ResumeError("Image and label directories have different output stems")
    if require_yolo_segmentation and (
            file_stems["yolo"] != image_stems or file_stems["extra"] != image_stems):
        raise ResumeError(
            "YOLO segmentation, extra-label, image, and custom-label directories "
            "must have identical output stems")
    if not require_yolo_segmentation and (file_stems["yolo"] or file_stems["extra"]):
        raise ResumeError(
            "Output already contains YOLO segmentation files; keep the export enabled "
            "to avoid a mixed-format dataset")
    unknown = image_stems - set(expected_by_stem)
    if unknown:
        raise ResumeError(f"Output contains stems outside this generation request: {sorted(unknown)}")

    present = [index for index, paths in enumerate(expected) if paths.stem in image_stems]
    if present != list(range(len(present))):
        raise ResumeError(f"Output indices are not a contiguous prefix: {present}")
    records = _read_manifest(output)
    if len(records) > len(present):
        raise ResumeError("Manifest has more records than completed output pairs")
    for index, record in enumerate(records):
        _validate_record(record, expected[index], require_yolo_segmentation)
    for index in range(len(records), len(present)):
        append_completed_pair(
            output, expected[index], recovered=True,
            require_yolo_segmentation=require_yolo_segmentation)
    return len(present)
