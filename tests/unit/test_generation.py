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
from rules.generation import (ResumeError, append_completed_pair, pair_paths,
                              resume_next_index, seed_for_index, stem_for_index)  # noqa: E402


def _output(root: str) -> config.OutputLayout:
    return config.OutputLayout(root=root)


def _write_pair(paths) -> None:
    os.makedirs(os.path.dirname(paths.image_path), exist_ok=True)
    os.makedirs(os.path.dirname(paths.label_path), exist_ok=True)
    with open(paths.image_path, "wb") as handle:
        handle.write(b"png")
    with open(paths.label_path, "w", encoding="utf-8") as handle:
        handle.write("label\n")


def test_seed_and_stem_are_deterministic_and_explicit():
    assert seed_for_index(100, 4) == 104
    assert stem_for_index(4, 104) == "000004_seed104"
    paths = pair_paths(_output("out"), 100, 4)
    assert paths.stem == "000004_seed104"
    assert paths.image_relpath == os.path.join("images", "000004_seed104.png")
    assert paths.label_relpath == os.path.join("labels", "000004_seed104.txt")


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


def _run_all():
    fns = [value for key, value in sorted(globals().items())
           if key.startswith("test_") and callable(value)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
