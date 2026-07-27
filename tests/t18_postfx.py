r"""Phase 6 acceptance - post-effect before/after strip from an accepted render.

Run after t16 or point TCG_POSTFX_INPUT at an accepted PNG. The matching custom-polygon
label defaults to the image stem plus '.txt'; override with TCG_POSTFX_LABEL.

HOW TO RUN:
    "C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe" tests\t18_postfx.py

OUTPUT (out/):
    t18_postfx_00..07_<effect>.png + .txt + _viz.png
    t18_postfx_contact.png
    t18_postfx_report.json

PASS if every effect is visible but plausible, labels overlay the processed images, and
the report says every output label is byte-identical to the source label.
"""
from __future__ import annotations

from dataclasses import asdict
import glob
import hashlib
import json
import os
import shutil
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from labeltools.visualize import visualize_poly_label_file  # noqa: E402
from postfx.effects import apply_postfx_file  # noqa: E402
from rules import combinations as C  # noqa: E402


def _source_paths(out_dir: str) -> tuple[str, str]:
    image = os.environ.get("TCG_POSTFX_INPUT")
    if not image:
        candidates = sorted(glob.glob(os.path.join(out_dir, "t16_phase5_*.png")))
        candidates = [path for path in candidates if "_contact" not in path and "_viz" not in path]
        if not candidates:
            raise RuntimeError("No t16 render found. Run t16 first or set TCG_POSTFX_INPUT.")
        image = candidates[0]
    label = os.environ.get("TCG_POSTFX_LABEL", os.path.splitext(image)[0] + ".txt")
    if not os.path.isfile(image) or not os.path.isfile(label):
        raise RuntimeError(f"Need an image/label pair, got image={image!r}, label={label!r}")
    return image, label


def _sample_enabled_effect(effect: str):
    for seed in range(100000, 110000):
        scene = C.sample_scene_config({"layouts": ["table"], "post_effects": [effect]}, seed)
        settings = getattr(scene.postfx, effect)
        if settings is not None:
            return seed, settings
    raise RuntimeError(f"{effect} never enabled; set postfx.{effect}.probability above zero.")


def _write_contact_sheet(views, path: str) -> None:
    rows = []
    for view in views:
        before = cv2.imread(view["before_viz"], cv2.IMREAD_COLOR)
        after = cv2.imread(view["after_viz"], cv2.IMREAD_COLOR)
        if before is None or after is None:
            raise RuntimeError("Could not read a postfx visualization")
        before = cv2.resize(before, (320, 320), interpolation=cv2.INTER_AREA)
        after = cv2.resize(after, (320, 320), interpolation=cv2.INTER_AREA)
        tile = cv2.hconcat((before, after))
        cv2.rectangle(tile, (0, 0), (640, 28), (12, 12, 12), -1)
        cv2.putText(tile, f"{view['index']:02d} {view['effect']} seed={view['seed']}  before | after",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
        rows.append(tile)
    if not cv2.imwrite(path, cv2.vconcat(rows)):
        raise RuntimeError(f"Could not write contact sheet {path!r}")


def main() -> None:
    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    os.makedirs(out_dir, exist_ok=True)
    image_path, label_path = _source_paths(out_dir)
    source_viz = os.path.join(out_dir, "t18_postfx_source_viz.png")
    visualize_poly_label_file(image_path, label_path, source_viz)
    with open(label_path, "rb") as handle:
        label_hash = hashlib.sha256(handle.read()).hexdigest()

    views = []
    for index, effect in enumerate(C.POST_EFFECTS):
        seed, settings = _sample_enabled_effect(effect)
        stem = f"t18_postfx_{index:02d}_{effect}"
        output = os.path.join(out_dir, stem + ".png")
        output_label = os.path.join(out_dir, stem + ".txt")
        shutil.copyfile(label_path, output_label)
        apply_postfx_file(image_path, output, C.PostFxConfig(**{effect: settings}))
        output_viz = os.path.join(out_dir, stem + "_viz.png")
        visualize_poly_label_file(output, output_label, output_viz)
        with open(output_label, "rb") as handle:
            output_hash = hashlib.sha256(handle.read()).hexdigest()
        views.append({
            "index": index,
            "effect": effect,
            "seed": seed,
            "settings": asdict(settings),
            "label_sha256": output_hash,
            "label_identical": output_hash == label_hash,
            "before_viz": source_viz,
            "after_viz": output_viz,
        })

    contact = os.path.join(out_dir, "t18_postfx_contact.png")
    _write_contact_sheet(views, contact)
    report = {
        "source_image": image_path,
        "source_label": label_path,
        "source_label_sha256": label_hash,
        "all_labels_identical": all(view["label_identical"] for view in views),
        "views": [{key: value for key, value in view.items()
                   if key not in {"before_viz", "after_viz"}} for view in views],
    }
    report_path = os.path.join(out_dir, "t18_postfx_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[t18] PASS: wrote {contact} and {report_path}")
    print("[t18] Review realism and label alignment in every before/after pair.")


if __name__ == "__main__":
    main()
