"""Docker unit tests for deterministic Phase 7 output naming and resume recovery."""
from __future__ import annotations

import json
import os
import sys
import tempfile

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from rules.generation import (ResumeError, append_completed_pair,  # noqa: E402
                               append_refraction_failures, pair_paths,
                               record_failed_seed, resume_next_index,
                               seed_for_index, stem_for_index)


def _output(root: str) -> config.OutputLayout:
    return config.OutputLayout(root=root)


def _write_pair(paths) -> None:
    os.makedirs(os.path.dirname(paths.image_path), exist_ok=True)
    os.makedirs(os.path.dirname(paths.label_path), exist_ok=True)
    with open(paths.image_path, "wb") as handle:
        handle.write(b"png")
    with open(paths.label_path, "w", encoding="utf-8") as handle:
        handle.write("label\n")


def _write_yolo_pair(paths) -> None:
    for path, content in ((paths.yolo_label_path, "0 0 0 1 0 1 1\n"),
                          (paths.extra_label_path, "card|none\n")):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)


def test_seed_and_stem_are_deterministic_and_explicit():
    assert seed_for_index(100, 4) == 104
    assert stem_for_index(4, 104) == "000004_seed104"
    paths = pair_paths(_output("out"), 100, 4)
    assert paths.stem == "000004_seed104"
    assert paths.image_relpath == os.path.join("images", "000004_seed104.png")
    assert paths.label_relpath == os.path.join("labels", "000004_seed104.txt")
    assert paths.yolo_label_relpath == os.path.join(
        "labels_yolo", "000004_seed104.txt")
    assert paths.extra_label_relpath == os.path.join(
        "extra_label", "000004_seed104.txt")


def test_manifest_resume_and_recovery_of_published_pair():
    with tempfile.TemporaryDirectory() as d:
        output = _output(d)
        first = pair_paths(output, 50, 0)
        second = pair_paths(output, 50, 1)
        _write_pair(first)
        append_completed_pair(output, first, {"seed": 50})
        _write_pair(second)  # Simulates a crash after pair publish, before manifest append.
        assert resume_next_index(output, 50, 3) == 2
        with open(os.path.join(d, output.manifest_name), encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        assert [record["index"] for record in records] == [0, 1]
        assert records[1]["recovered"] is True


def test_resume_rejects_orphans_and_gaps():
    with tempfile.TemporaryDirectory() as d:
        output = _output(d)
        first = pair_paths(output, 10, 0)
        _write_pair(first)
        os.remove(first.label_path)
        try:
            resume_next_index(output, 10, 2)
        except ResumeError:
            pass
        else:
            raise AssertionError("image without label must fail resume")

    with tempfile.TemporaryDirectory() as d:
        output = _output(d)
        _write_pair(pair_paths(output, 10, 1))
        try:
            resume_next_index(output, 10, 3)
        except ResumeError:
            pass
        else:
            raise AssertionError("non-contiguous pair index must fail resume")


def test_manifest_mismatch_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        output = _output(d)
        first = pair_paths(output, 10, 0)
        _write_pair(first)
        manifest = os.path.join(d, output.manifest_name)
        with open(manifest, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"index": 0, "seed": 999, "stem": first.stem,
                                     "image": first.image_relpath,
                                     "label": first.label_relpath}) + "\n")
        try:
            resume_next_index(output, 10, 2)
        except ResumeError:
            pass
        else:
            raise AssertionError("manifest seed mismatch must fail resume")


def test_failed_seed_is_persistently_skipped_and_later_outputs_can_have_gaps():
    with tempfile.TemporaryDirectory() as d:
        output = _output(d)
        assert record_failed_seed(output, 100, 4, 0, "render failed") == "skipped"
        assert resume_next_index(output, 100, 4) == 1

        second = pair_paths(output, 100, 1)
        _write_pair(second)
        append_completed_pair(output, second)
        assert resume_next_index(output, 100, 4) == 2

        with open(os.path.join(d, output.manifest_name), encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        assert records[0] == {
            "index": 0, "seed": 100, "stem": "000000_seed100",
            "status": "skipped", "error": "render failed"}
        assert records[1]["index"] == 1


def test_failed_seed_removes_partial_files_but_recovers_complete_pair():
    with tempfile.TemporaryDirectory() as d:
        output = _output(d)
        first = pair_paths(output, 200, 0)
        os.makedirs(os.path.dirname(first.label_path), exist_ok=True)
        with open(first.label_path, "w", encoding="utf-8") as handle:
            handle.write("partial")
        assert record_failed_seed(output, 200, 2, 0, "partial publish") == "skipped"
        assert not os.path.exists(first.label_path)

    with tempfile.TemporaryDirectory() as d:
        output = _output(d)
        first = pair_paths(output, 300, 0)
        _write_pair(first)
        assert record_failed_seed(output, 300, 2, 0, "late failure") == "recovered"
        assert resume_next_index(output, 300, 2) == 1
        with open(os.path.join(d, output.manifest_name), encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        assert record["recovered"] is True
        assert record.get("status") != "skipped"


def test_failed_seed_removes_staging_files_and_resume_ignores_them():
    with tempfile.TemporaryDirectory() as d:
        output = _output(d)
        first = pair_paths(output, 400, 0)
        stages = []
        for path, role in ((first.image_path, "raw"), (first.image_path, "image"),
                           (first.label_path, "label"), (first.yolo_label_path, "yolo"),
                           (first.extra_label_path, "extra")):
            base, extension = os.path.splitext(path)
            staged = f"{base}.postfx-{role}{extension}"
            os.makedirs(os.path.dirname(staged), exist_ok=True)
            with open(staged, "wb") as handle:
                handle.write(b"partial")
            stages.append(staged)
        assert resume_next_index(output, 400, 2) == 0
        assert record_failed_seed(output, 400, 2, 0, "abrupt exit") == "skipped"
        assert not any(os.path.exists(path) for path in stages)
        assert resume_next_index(output, 400, 2) == 1


def test_yolo_resume_requires_both_synchronized_optional_files():
    with tempfile.TemporaryDirectory() as d:
        output = _output(d)
        paths = pair_paths(output, 70, 0)
        _write_pair(paths)
        try:
            resume_next_index(output, 70, 1, require_yolo_segmentation=True)
        except ResumeError:
            pass
        else:
            raise AssertionError("enabled YOLO export must reject a custom-only pair")

        _write_yolo_pair(paths)
        assert resume_next_index(
            output, 70, 1, require_yolo_segmentation=True) == 1
        with open(os.path.join(d, output.manifest_name), encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        assert record["yolo_label"] == paths.yolo_label_relpath
        assert record["extra_label"] == paths.extra_label_relpath

        try:
            resume_next_index(output, 70, 1, require_yolo_segmentation=False)
        except ResumeError:
            pass
        else:
            raise AssertionError("disabling export must not create a mixed-format dataset")


def test_refraction_failures_are_appended_beside_manifest():
    with tempfile.TemporaryDirectory() as d:
        output = _output(d)
        paths = pair_paths(output, 2026073, 1)
        assert append_refraction_failures(output, paths, []) is None
        try:
            append_refraction_failures(output, paths, [{"card_id": "before-pair"}])
        except ResumeError:
            pass
        else:
            raise AssertionError("diagnostics must not precede the published pair")

        _write_pair(paths)
        failures = [{
            "card_id": "bw8-40",
            "instance_name": "Card7_slab",
            "corner_index": 2,
            "corner_name": "TR",
            "error": "apparent-ray solver could not find a decreasing step",
        }]
        diagnostics = append_refraction_failures(output, paths, failures)
        assert diagnostics == os.path.abspath(output.refraction_failures_path())
        assert os.path.dirname(diagnostics) == os.path.dirname(
            os.path.join(os.path.abspath(d), output.manifest_name))
        with open(diagnostics, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        assert records == [{
            "index": 1,
            "seed": 2026074,
            "stem": "000001_seed2026074",
            "image": os.path.join("images", "000001_seed2026074.png"),
            "label": os.path.join("labels", "000001_seed2026074.txt"),
            **failures[0],
        }]


def _run_all():
    fns = [value for key, value in sorted(globals().items())
           if key.startswith("test_") and callable(value)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
