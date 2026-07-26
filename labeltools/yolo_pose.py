"""
Ultralytics YOLO-pose label formatting & parsing (bpy-FREE, Docker-testable).

Label line (spec §3.9 + holo tag), one per labeled card:

    class cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4  |<card_id>|<holo_tag>

- All coords normalized to [0,1], image top-left origin.
- bbox (cx,cy,w,h) = min/max envelope of the four corner keypoints.
- Keypoints in FIXED order (config.KEYPOINT_ORDER, default TL,TR,BR,BL) in the
  card's own upright frame; visibility flag = 2 for all four.
- The trailing ` |<card_id>|<holo_tag>` is our extension. holo_tag is one of:
  'none' (no holo), 'full' (whole-card holo), 'holo' (picture-region holo),
  'reverse' (reverse holo). A "strictly standard" variant (no suffixes) can be
  emitted into a sibling folder via include_id=False.

The Blender side computes the four normalized corners (with Y already flipped to
top-left origin) and hands them here; this module never touches bpy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

try:
    import config
except ImportError:  # pragma: no cover
    from .. import config  # type: ignore

Corner = Tuple[float, float]  # (x, y) normalized, top-left origin


def _fmt(v: float) -> str:
    """6-decimal fixed format; keeps label files diff-stable and compact."""
    return f"{v:.6f}"


def bbox_from_corners(corners: Sequence[Corner]) -> Tuple[float, float, float, float]:
    """Return (cx, cy, w, h) normalized, as the min/max envelope of corners."""
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0)


HOLO_TAGS = ("none", "full", "holo", "reverse")


@dataclass(frozen=True)
class CardLabel:
    """One card instance's label data (already normalized, top-left origin).

    `corners` holds the keypoints in order: exactly 4 (TL,TR,BR,BL) for a full
    'card' (class 0), or 3-8 clipped-boundary points for a 'partial_card' (class 1).
    bbox is the min/max envelope of whatever points are present.
    """
    card_id: str
    corners: Tuple[Corner, ...]                      # 4 for card, 3-8 for partial_card
    holo_tag: str = "none"                           # none | full | holo | reverse
    class_id: int = config.YOLO_CLASS_ID
    visibility: int = config.KPT_VISIBILITY

    def to_line(self, include_id: bool = True) -> str:
        cx, cy, w, h = bbox_from_corners(self.corners)
        parts: List[str] = [str(self.class_id), _fmt(cx), _fmt(cy), _fmt(w), _fmt(h)]
        for (x, y) in self.corners:
            parts.extend([_fmt(x), _fmt(y), str(int(self.visibility))])
        line = " ".join(parts)
        if include_id:
            line += f" |{self.card_id}|{self.holo_tag}"
        return line


@dataclass(frozen=True)
class ParsedLabel:
    """Result of parsing one label line (for the visualizer / validation)."""
    class_id: int
    bbox: Tuple[float, float, float, float]           # cx, cy, w, h
    corners: List[Corner]
    visibilities: List[int]
    card_id: str        # "" if the line had no |id suffix
    holo_tag: str = ""  # "" if the line had no |tag suffix


def parse_pose_line(line: str) -> ParsedLabel:
    """Parse one label line, tolerating optional trailing ` |<card_id>|<holo_tag>`."""
    line = line.strip()
    card_id = ""
    holo_tag = ""
    if "|" in line:
        segs = line.split("|")
        line = segs[0]
        if len(segs) >= 2:
            card_id = segs[1].strip()
        if len(segs) >= 3:
            holo_tag = segs[2].strip()
    toks = line.split()
    if len(toks) < 14 or (len(toks) - 5) % 3 != 0:
        raise ValueError(f"Invalid pose label token count: {line!r}")
    class_id = int(float(toks[0]))
    cx, cy, w, h = (float(t) for t in toks[1:5])
    kpt_toks = toks[5:]
    corners: List[Corner] = []
    vis: List[int] = []
    for i in range(0, len(kpt_toks), 3):
        x, y = float(kpt_toks[i]), float(kpt_toks[i + 1])
        v = int(float(kpt_toks[i + 2]))
        corners.append((x, y))
        vis.append(v)
    return ParsedLabel(class_id, (cx, cy, w, h), corners, vis, card_id, holo_tag)


# --------------------------------------------------------------------------- #
# Occlusion-aware CUSTOM format (spec: 2026-07-22 user).
#
#   <class> <rb1x> <rb1y> <rb2x> <rb2y>  <x1> <y1> <f1> ... <xn> <yn> <fn>  |<id>|<holo>
#
# - <class>: 0 'card' | 1 'partial_card'.
# - (rb1x,rb1y),(rb2x,rb2y): the two opposite corners (min, max) of the axis-aligned
#   bounding rectangle of the bound points (replaces YOLO's cx,cy,w,h).
# - each bound point is (x, y, flag); flag 1=TL 2=TR 3=BL 4=BR original corner,
#   5=a non-corner bound point (frustum crossing / covered frame corner / occlusion
#   vertex). Points are in clockwise perimeter order; missing corners are simply absent.
#   The polygon may be CONCAVE (occlusion carving).
# --------------------------------------------------------------------------- #
TaggedPoint = Tuple[float, float, int]  # (x, y, flag)


@dataclass(frozen=True)
class PolyLabel:
    """One card's occlusion-aware bound in the custom format above."""
    card_id: str
    points: Tuple[TaggedPoint, ...]     # (x, y, flag) in clockwise perimeter order
    holo_tag: str = "none"
    class_id: int = config.YOLO_CLASS_ID

    def bbox2(self) -> Tuple[Corner, Corner]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys)), (max(xs), max(ys))

    def to_line(self, include_id: bool = True) -> str:
        (x0, y0), (x1, y1) = self.bbox2()
        parts: List[str] = [str(self.class_id), _fmt(x0), _fmt(y0), _fmt(x1), _fmt(y1)]
        for (x, y, f) in self.points:
            parts.extend([_fmt(x), _fmt(y), str(int(f))])
        line = " ".join(parts)
        if include_id:
            line += f" |{self.card_id}|{self.holo_tag}"
        return line


@dataclass(frozen=True)
class ParsedPoly:
    class_id: int
    bbox2: Tuple[Corner, Corner]          # (min_xy, max_xy)
    points: List[TaggedPoint]             # (x, y, flag)
    card_id: str
    holo_tag: str = ""


def parse_poly_line(line: str) -> ParsedPoly:
    """Parse one custom-format line (tolerating the optional ` |<id>|<holo>`)."""
    line = line.strip()
    card_id = holo_tag = ""
    if "|" in line:
        segs = line.split("|")
        line = segs[0]
        if len(segs) >= 2:
            card_id = segs[1].strip()
        if len(segs) >= 3:
            holo_tag = segs[2].strip()
    toks = line.split()
    if len(toks) < 14 or (len(toks) - 5) % 3 != 0:
        raise ValueError(f"Invalid polygon label token count: {line!r}")
    class_id = int(float(toks[0]))
    x0, y0, x1, y1 = (float(t) for t in toks[1:5])
    pt_toks = toks[5:]
    points: List[TaggedPoint] = []
    for i in range(0, len(pt_toks), 3):
        points.append((float(pt_toks[i]), float(pt_toks[i + 1]), int(float(pt_toks[i + 2]))))
    return ParsedPoly(class_id, ((x0, y0), (x1, y1)), points, card_id, holo_tag)


def write_poly_label_file(path: str, labels: Sequence[PolyLabel], include_id: bool = True) -> None:
    """Write one custom-format line per PolyLabel (empty list => empty file, still
    written so every image keeps a label pair)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for lab in labels:
            fh.write(lab.to_line(include_id=include_id) + "\n")


def write_label_file(path: str, labels: Sequence[CardLabel], include_id: bool = True) -> None:
    """Write one line per label. Empty list => empty file (still written, so a
    rendered image always has its label pair, even when 0 cards are labeled)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for lab in labels:
            fh.write(lab.to_line(include_id=include_id) + "\n")


def write_dataset_yaml(
    path: str,
    train: str = "images",
    val: str = "images",
    class_names: Sequence[str] = config.CLASS_NAMES,
) -> None:
    """Write an Ultralytics dataset.yaml with kpt_shape [4, 3] (spec §3.9).

    Written by hand (no PyYAML dependency) to keep this module dependency-free.
    kpt_shape describes the full 'card' class (4 corners); a 'partial_card' carries
    a variable 3-8 keypoints (see config.PARTIAL_KPT_RANGE) — noted here for humans.
    """
    kx, ky = config.KPT_SHAPE
    pmin, pmax = config.PARTIAL_KPT_RANGE
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    content = (
        f"# Auto-generated by labeltools/yolo_pose.py\n"
        f"path: .\n"
        f"train: {train}\n"
        f"val: {val}\n"
        f"kpt_shape: [{kx}, {ky}]\n"
        f"# keypoint order (class 0 'card'): {', '.join(config.KEYPOINT_ORDER)} "
        f"(card's own upright frame)\n"
        f"# class 1 'partial_card' carries {pmin}-{pmax} clipped-boundary points\n"
        f"names:\n{names}\n"
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
