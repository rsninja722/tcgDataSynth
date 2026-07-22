"""
Label visualizer (bpy-FREE, Docker-testable) — spec §3.9's primary verification tool.

Draws, for every card in a YOLO-pose label file, onto the corresponding image:
  - the bbox rectangle,
  - the four corner keypoints as numbered dots (1..4 in KEYPOINT_ORDER),
  - the quad outline connecting the corners in order,
  - the card_id (if present).

CLI:
    python3 labeltools/visualize.py <image.png> <label.txt> [<out.png>]
    # default out = <image>_viz.png next to the image

Programmatic:
    from labeltools.visualize import visualize_label_file
    visualize_label_file("out/t01_headon.png", "out/t01_headon.txt", "out/viz.png")

Uses OpenCV (headless ok). Coordinates in the label are normalized top-left origin,
so pixel = (x * W, y * H) directly — no flip here (the flip happened when writing).
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import cv2
import numpy as np

# Allow both `python3 labeltools/visualize.py` and package import.
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from labeltools.yolo_pose import ParsedLabel, parse_pose_line  # noqa: E402
from labeltools.yolo_pose import ParsedPoly, parse_poly_line  # noqa: E402

# BGR colors (OpenCV order). One distinct color per corner index so ordering is
# visually verifiable at a glance.
_CORNER_COLORS = [
    (60, 60, 255),    # 1 TL - red
    (60, 220, 60),    # 2 TR - green
    (255, 180, 40),   # 3 BR - blue
    (40, 220, 255),   # 4 BL - yellow
]
_BBOX_COLOR = (255, 255, 255)
_QUAD_COLOR = (200, 200, 200)
_PARTIAL_BBOX_COLOR = (0, 140, 255)   # orange: class 1 partial_card
_PARTIAL_QUAD_COLOR = (0, 140, 255)

try:
    import config as _config  # noqa: E402
    _PARTIAL_CLASS_ID = _config.PARTIAL_CLASS_ID
except Exception:  # noqa: BLE001
    _PARTIAL_CLASS_ID = 1


def _to_px(x: float, y: float, w: int, h: int):
    return (int(round(x * w)), int(round(y * h)))


def draw_label(img: np.ndarray, parsed: ParsedLabel) -> np.ndarray:
    """Draw one parsed label onto img (modified in place, also returned)."""
    h, w = img.shape[:2]
    is_partial = parsed.class_id == _PARTIAL_CLASS_ID
    bbox_color = _PARTIAL_BBOX_COLOR if is_partial else _BBOX_COLOR
    quad_color = _PARTIAL_QUAD_COLOR if is_partial else _QUAD_COLOR
    cx, cy, bw, bh = parsed.bbox
    x0, y0 = _to_px(cx - bw / 2, cy - bh / 2, w, h)
    x1, y1 = _to_px(cx + bw / 2, cy + bh / 2, w, h)
    cv2.rectangle(img, (x0, y0), (x1, y1), bbox_color, 1)

    pts = [_to_px(x, y, w, h) for (x, y) in parsed.corners]
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True, color=quad_color, thickness=1)

    for i, ((px, py), (nx, ny)) in enumerate(zip(pts, parsed.corners)):
        color = _CORNER_COLORS[i % len(_CORNER_COLORS)]
        # A crossing point sits on the frustum boundary (a component is 0 or 1);
        # ring it in white so the on-edge keypoints are obvious for partial cards.
        on_edge = min(nx, ny) <= 1e-4 or max(nx, ny) >= 1.0 - 1e-4
        cv2.circle(img, (px, py), 6, color, -1)
        cv2.circle(img, (px, py), 7, (255, 255, 255) if (is_partial and on_edge) else (0, 0, 0), 1)
        cv2.putText(img, str(i + 1), (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    label_txt = parsed.card_id
    if is_partial:
        label_txt = ("partial " + label_txt).strip() or "partial"
    if label_txt:
        tag = f" [{parsed.holo_tag}]" if parsed.holo_tag else ""
        cv2.putText(img, label_txt + tag, (x0, max(0, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1, cv2.LINE_AA)
    return img


# Flag -> BGR color for the custom (poly) format. 1=TL 2=TR 3=BL 4=BR, 5=created.
_FLAG_COLORS = {
    1: (60, 60, 255),     # TL red
    2: (60, 220, 60),     # TR green
    3: (40, 220, 255),    # BL yellow
    4: (255, 180, 40),    # BR blue
    5: (255, 60, 255),    # created magenta
}


def draw_poly_label(img: np.ndarray, parsed: ParsedPoly) -> np.ndarray:
    """Draw one custom-format bound: 2-corner bbox, concave polygon outline, and each
    point colored by flag (magenta = created/occlusion point)."""
    h, w = img.shape[:2]
    is_partial = parsed.class_id == _PARTIAL_CLASS_ID
    (bx0, by0), (bx1, by1) = parsed.bbox2
    box_color = _PARTIAL_BBOX_COLOR if is_partial else _BBOX_COLOR
    cv2.rectangle(img, _to_px(bx0, by0, w, h), _to_px(bx1, by1, w, h), box_color, 1)

    pts = [_to_px(x, y, w, h) for (x, y, _f) in parsed.points]
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True,
                      color=box_color, thickness=1)
    for (px, py), (_x, _y, flag) in zip(pts, parsed.points):
        color = _FLAG_COLORS.get(int(flag), (200, 200, 200))
        cv2.circle(img, (px, py), 6, color, -1)
        cv2.circle(img, (px, py), 6, (0, 0, 0), 1)
        cv2.putText(img, str(int(flag)), (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    label_txt = ("partial " if is_partial else "") + (parsed.card_id or "")
    if label_txt.strip():
        tag = f" [{parsed.holo_tag}]" if parsed.holo_tag else ""
        cv2.putText(img, label_txt.strip() + tag, (_to_px(bx0, by0, w, h)[0],
                    max(0, _to_px(bx0, by0, w, h)[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
    return img


def visualize_poly_label_file(image_path: str, label_path: str, out_path: Optional[str] = None) -> str:
    """Render all custom-format (poly) labels from label_path onto image_path."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    labels: List[ParsedPoly] = []
    if os.path.isfile(label_path):
        with open(label_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    labels.append(parse_poly_line(line))
    for parsed in labels:
        draw_poly_label(img, parsed)
    cv2.putText(img, f"labels: {len(labels)}", (8, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, f"labels: {len(labels)}", (8, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if out_path is None:
        base, ext = os.path.splitext(image_path)
        out_path = f"{base}_viz{ext or '.png'}"
    cv2.imwrite(out_path, img)
    return out_path


def visualize_label_file(image_path: str, label_path: str, out_path: Optional[str] = None) -> str:
    """Render all labels from label_path onto image_path; write annotated PNG.

    Returns the output path. An empty label file yields a copy with no overlays
    (useful for confirming a deliberately-unlabeled card renders but is skipped).
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    labels: List[ParsedLabel] = []
    if os.path.isfile(label_path):
        with open(label_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    labels.append(parse_pose_line(line))
    for parsed in labels:
        draw_label(img, parsed)

    # Footer: count of drawn instances, so an empty label is obvious.
    cv2.putText(img, f"labels: {len(labels)}", (8, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, f"labels: {len(labels)}", (8, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if out_path is None:
        base, ext = os.path.splitext(image_path)
        out_path = f"{base}_viz{ext or '.png'}"
    cv2.imwrite(out_path, img)
    return out_path


def _main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    image_path, label_path = argv[1], argv[2]
    out_path = argv[3] if len(argv) > 3 else None
    # The generator now writes the custom occlusion-aware (poly) format.
    result = visualize_poly_label_file(image_path, label_path, out_path)
    print(f"wrote {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
