"""
Docker unit tests for labeltools/yolo_pose.py and labeltools/visualize.py.

Run:  python3 tests/unit/test_labeltools.py
Also writes out/unit_viz_check.png for eyeball confirmation.
"""
from __future__ import annotations

import os
import sys
import tempfile

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import cv2  # noqa: E402

import config  # noqa: E402
from labeltools import yolo_pose as yp  # noqa: E402
from labeltools.visualize import visualize_label_file, _CORNER_COLORS  # noqa: E402


def test_bbox_envelope():
    corners = [(0.2, 0.3), (0.7, 0.25), (0.75, 0.8), (0.15, 0.85)]
    cx, cy, w, h = yp.bbox_from_corners(corners)
    assert abs(w - (0.75 - 0.15)) < 1e-9
    assert abs(h - (0.85 - 0.25)) < 1e-9
    assert abs(cx - (0.75 + 0.15) / 2) < 1e-9
    assert abs(cy - (0.85 + 0.25) / 2) < 1e-9


def test_line_token_count_and_suffix():
    lab = yp.CardLabel("charizard", ((0.2, 0.3), (0.7, 0.25), (0.75, 0.8), (0.15, 0.85)),
                       holo_tag="full")
    line = lab.to_line(include_id=True)
    # ... coords ... |<card_id>|<holo_tag>
    segs = line.split("|")
    assert len(segs) == 3
    assert segs[1].strip() == "charizard" and segs[2].strip() == "full"
    core = segs[0].split()
    assert len(core) == 17          # 5 bbox + 12 kpt
    assert core[0] == str(config.YOLO_CLASS_ID)
    assert core[7] == "2" and core[10] == "2" and core[13] == "2" and core[16] == "2"
    # standard variant has no pipe (no id, no tag)
    assert "|" not in lab.to_line(include_id=False)


def test_parse_roundtrip_with_and_without_id():
    lab = yp.CardLabel("Base_Set_004", ((0.11, 0.22), (0.71, 0.20), (0.73, 0.79), (0.13, 0.81)),
                       holo_tag="reverse")
    for include_id in (True, False):
        parsed = yp.parse_pose_line(lab.to_line(include_id=include_id))
        assert parsed.card_id == ("Base_Set_004" if include_id else "")
        assert parsed.holo_tag == ("reverse" if include_id else "")
        for (ax, ay), (bx, by) in zip(parsed.corners, lab.corners):
            assert abs(ax - bx) < 1e-6 and abs(ay - by) < 1e-6
        assert parsed.visibilities == [2, 2, 2, 2]


def test_empty_label_file_written():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "labels", "empty.txt")
        yp.write_label_file(p, [])
        assert os.path.isfile(p)
        assert os.path.getsize(p) == 0


def test_dataset_yaml_kpt_shape():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dataset.yaml")
        yp.write_dataset_yaml(p)
        text = open(p).read()
        assert "kpt_shape: [4, 3]" in text
        assert "card" in text


def test_visualizer_places_corners_at_expected_pixels():
    """End-to-end: write a label with known corners, visualize, and confirm the
    drawn corner dots land at pixel = (x*W, y*H) in KEYPOINT_ORDER. This proves
    the writer's normalized coords and the visualizer's mapping agree (no flip)."""
    W = H = 400
    corners = [(0.20, 0.30), (0.70, 0.25), (0.75, 0.80), (0.15, 0.85)]  # TL,TR,BR,BL
    with tempfile.TemporaryDirectory() as d:
        img_path = os.path.join(d, "img.png")
        lab_path = os.path.join(d, "img.txt")
        out_path = os.path.join(config.OUTPUT.root, "unit_viz_check.png")
        os.makedirs(config.OUTPUT.root, exist_ok=True)
        cv2.imwrite(img_path, np.full((H, W, 3), 128, np.uint8))
        yp.write_label_file(lab_path, [yp.CardLabel("testcard", tuple(corners))])
        visualize_label_file(img_path, lab_path, out_path)

        viz = cv2.imread(out_path, cv2.IMREAD_COLOR)
        for i, (x, y) in enumerate(corners):
            px, py = int(round(x * W)), int(round(y * H))
            pixel = viz[py, px]  # BGR at the dot center
            expected = np.array(_CORNER_COLORS[i], dtype=np.int16)
            diff = np.abs(pixel.astype(np.int16) - expected).sum()
            assert diff <= 30, f"corner {i+1} color {tuple(pixel)} != {_CORNER_COLORS[i]} (diff {diff})"
    print(f"    (wrote {out_path} for eyeball check)")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
