"""
Project-wide configuration and physical constants (bpy-FREE, Docker-testable).

All Blender-side code imports these values instead of hardcoding them, so paths
and defaults live in exactly one place. Nothing here imports `bpy`.

UNITS: Blender scene works in METERS at real-world scale. Constants below give
both the millimetre spec value and the metre value actually used.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Root folder that is recursively searched for card face images. This is the
# USER's machine path (not visible from the Docker dev container). Override with
# the TCG_CARD_IMAGE_ROOT env var when running elsewhere.
DEFAULT_CARD_IMAGE_ROOT = r"C:\Code\React\CollectiblesApp\src\ai_dev\datasets\pokemon\data\images"

# Image extensions treated as card faces during recursive discovery.
CARD_IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# Optional central metadata file (card_id -> [x0, y0, x1, y1] picture region in
# normalized top-down coords). If present next to the image root or in the project
# root, it overrides the default region per card. Absent => default region used.
PICTURE_REGIONS_FILENAME = "picture_regions.json"

# The generic card-back texture lives in the image root as this filename. It is
# EXCLUDED from card-face discovery (it is not a selectable card) and is used as
# the back-of-card texture in layouts where backs face the camera.
BACK_IMAGE_FILENAME = "back.png"


def back_image_path() -> str:
    """Absolute path to the generic card-back texture, or '' if not present."""
    p = os.path.join(card_image_root(), BACK_IMAGE_FILENAME)
    return p if os.path.isfile(p) else ""


def card_image_root() -> str:
    """Resolved card image root: env override wins, else the baked-in default."""
    return os.environ.get("TCG_CARD_IMAGE_ROOT", DEFAULT_CARD_IMAGE_ROOT)


# --------------------------------------------------------------------------- #
# Physical card geometry (spec §3.1)
# --------------------------------------------------------------------------- #
MM = 0.001  # 1 mm in meters

CARD_W_MM, CARD_H_MM, CARD_T_MM = 63.0, 88.0, 0.45
CARD_CORNER_RADIUS_MM = 3.0

CARD_W_M = CARD_W_MM * MM          # 0.063
CARD_H_M = CARD_H_MM * MM          # 0.088
CARD_T_M = CARD_T_MM * MM          # 0.00045
CARD_CORNER_RADIUS_M = CARD_CORNER_RADIUS_MM * MM  # 0.003

# --------------------------------------------------------------------------- #
# Default picture region (normalized, top-down), spec §3.4 typical values.
# Per-card overrides come from picture_regions.json; never hardcode downstream.
# Order: (x0, y0, x1, y1) with y measured top-down (0 = top of image).
# --------------------------------------------------------------------------- #
DEFAULT_PICTURE_REGION: Tuple[float, float, float, float] = (0.080, 0.098, 0.920, 0.471)

# --------------------------------------------------------------------------- #
# Render / output (spec §3.7, §3.9)
# --------------------------------------------------------------------------- #
RENDER_W = 1280
RENDER_H = 1280
FOCAL_MM_RANGE: Tuple[float, float] = (15.0, 55.0)
CAMERA_MAX_OFFAXIS_DEG = 50.0

# Render engine. Default switched to CYCLES (2026-07-19): EEVEE-Next did not
# refract through Principled transmission (clear sleeves rendered solid), and the
# project is transmission-heavy (sleeves/holders/slab/binder/display-case). User
# has Cycles+CUDA configured. EEVEE stays selectable for the Phase 8 A/B compare.
RENDER_ENGINE = "CYCLES"               # "CYCLES" | "BLENDER_EEVEE"
CYCLES_SAMPLES = 128                    # with denoising; tune for speed in Phase 8
CYCLES_DEVICE = "GPU"                   # user configured CUDA
EEVEE_RENDER_SAMPLES = 64              # scene.eevee.taa_render_samples

# Color management view transform, LOCKED (Phase 0 confirmed 'AgX' is the 5.0
# default & active). postfx assumes this stays constant across the whole dataset.
VIEW_TRANSFORM = "AgX"

# --------------------------------------------------------------------------- #
# Label format (spec §3.9)
# --------------------------------------------------------------------------- #
YOLO_CLASS_ID = 0
# Keypoint order in the card's own upright frame (user decision 2026-07-19).
KEYPOINT_ORDER = ("TL", "TR", "BR", "BL")
KPT_SHAPE = (4, 3)  # 4 keypoints, (x, y, visibility)
KPT_VISIBILITY = 2  # all corners flagged visible per spec


@dataclass(frozen=True)
class OutputLayout:
    """Where renders, labels, and manifests are written under out/."""
    root: str = "out"
    images_subdir: str = "images"
    labels_subdir: str = "labels"
    # Optional sibling folder for strictly-standard labels (no |id suffix), spec §3.9.
    std_labels_subdir: str = "labels_standard"
    manifest_name: str = "manifest.jsonl"

    def images_dir(self) -> str:
        return os.path.join(self.root, self.images_subdir)

    def labels_dir(self) -> str:
        return os.path.join(self.root, self.labels_subdir)

    def std_labels_dir(self) -> str:
        return os.path.join(self.root, self.std_labels_subdir)


OUTPUT = OutputLayout()
