r"""Phase 7 acceptance - one standalone-GUI Blender worker pair.

Runs the same worker process entry point used by gui.py against an isolated output
folder. It validates staging, post effects, custom-label publication, and the manifest.

HOW TO RUN (headless; cv2 and shapely are required in Blender's Python):
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tests\t20_standalone_generation.py

OUTPUT (out/):
    t20_standalone/images/000000_seed20260731.png
    t20_standalone/labels/000000_seed20260731.txt
    t20_standalone/manifest.jsonl
    t20_standalone/refraction_failures.txt
    t20_standalone_report.json

INTERACTIVE FOLLOW-UP:
    Run `python gui.py` with a normal desktop Python. Confirm config.json's
    blender_executable points to Blender 5.0. Set Pairs to 3, Start, press Pause while
    Blender renders, then Start / Resume. PASS if only completed pairs appear, stems
    are contiguous, and manifest records are one-for-one with the image/label pairs.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import bpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from blender import labeling  # noqa: E402
from blender.generation_worker import run_one  # noqa: E402
from rules.generation import resume_next_index  # noqa: E402


def main() -> None:
    if tuple(bpy.app.version) != (5, 0, 0):
        raise RuntimeError(f"t20 requires Blender 5.0.0, got {bpy.app.version_string}")
    out_root = os.path.join(_ROOT, config.OUTPUT.root)
    test_root = os.path.join(out_root, "t20_standalone")
    if os.path.isdir(test_root):
        shutil.rmtree(test_root)
    os.makedirs(test_root, exist_ok=True)
    config_path = os.path.join(test_root, "config.json")
    with open(os.path.join(_ROOT, config.CONFIG_FILENAME), "r", encoding="utf-8") as handle:
        test_config = json.load(handle)
    test_config["generation"]["count"] = 1
    test_config["generation"]["base_seed"] = 20260731
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(test_config, handle, indent=2)

    original_projection = labeling._project_apparent_corner
    injected = False

    def inject_one_failure(*args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            raise labeling.CornerProjectionError(
                "forced t20 apparent-ray convergence failure")
        return original_projection(*args, **kwargs)

    labeling._project_apparent_corner = inject_one_failure
    try:
        result = run_one(0, config_path=config_path, output_root=test_root)
    finally:
        labeling._project_apparent_corner = original_projection
    output = config.OutputLayout(root=test_root)
    assert os.path.isfile(result["image"]) and os.path.isfile(result["label"])
    assert injected and result["refraction_failure_count"] == 1
    assert result["refraction_failures"] == os.path.abspath(
        output.refraction_failures_path())
    with open(result["refraction_failures"], encoding="utf-8") as handle:
        failures = [json.loads(line) for line in handle if line.strip()]
    assert len(failures) == 1
    assert failures[0]["index"] == 0 and failures[0]["seed"] == 20260731
    assert failures[0]["corner_name"] == "TL"
    assert failures[0]["fallback"] == "direct-card-polygon"
    assert "forced t20" in failures[0]["error"]
    assert resume_next_index(output, 20260731, 1) == 1
    report = {
        "blender_version": bpy.app.version_string,
        "worker_result": result,
        "resume_next_index": 1,
        "manifest": os.path.join(test_root, output.manifest_name),
        "refraction_failures": failures,
    }
    report_path = os.path.join(out_root, "t20_standalone_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[t20] PASS: wrote {report_path}")
    print("[t20] Complete the documented standalone GUI Start/Pause/Resume follow-up.")


if __name__ == "__main__":
    main()
