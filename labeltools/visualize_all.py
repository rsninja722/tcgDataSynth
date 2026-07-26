"""Visualize every matching PNG/label pair in the project's out directory.

Usage:
    python3 labeltools/visualize_all.py
"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from labeltools.visualize import visualize_poly_label_file


def main() -> int:
    out_dir = _ROOT / "out"
    if not out_dir.is_dir():
        print(f"output directory does not exist: {out_dir}")
        return 1

    pairs = []
    for image_path in sorted(out_dir.glob("*.png")):
        if image_path.stem.endswith("_viz"):
            continue
        label_path = image_path.with_suffix(".txt")
        if label_path.is_file():
            pairs.append((image_path, label_path))

    if not pairs:
        print(f"no matching .png/.txt pairs found in {out_dir}")
        return 0

    for image_path, label_path in pairs:
        result = visualize_poly_label_file(str(image_path), str(label_path))
        print(f"wrote {result}")

    print(f"visualized {len(pairs)} pair(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
