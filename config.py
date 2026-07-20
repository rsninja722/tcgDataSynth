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


# --------------------------------------------------------------------------- #
# Central runtime config (single hand-editable JSON; scripts load at runtime so
# it can be tuned without touching code). All tunable params live here.
# --------------------------------------------------------------------------- #
CONFIG_FILENAME = "config.json"
DEFAULT_CONFIG = {
    "holo": {
        "angle_gain": 12.0,    # flash-band motion speed vs view angle
        "pattern_gain": 4.0,   # spatial break-up of the flash phase by the pattern
        "sharpness": 4.0,      # flash threshold (higher = narrower/more selective)
        "emit": 1.6,           # peak flash brightness
        "darken": 0.18,        # base darkening between flashes
    },
    # Per-layout scene params (each scene type has its OWN set). Add entries here as
    # layouts are built (out_of_frustum: 'keep' = render but don't label | 'remove').
    "layouts": {
        "table": {"max_cards": 8, "allow_overlap": False, "out_of_frustum": "keep"},
        "floating": {"max_cards": 12, "max_shapes": 12, "allow_overlap": False,
                     "out_of_frustum": "keep"},
    },
}
# Back-compat alias (used by tests / older references).
DEFAULT_HOLO_TUNING = DEFAULT_CONFIG["holo"]
_OUT_OF_FRUSTUM_CHOICES = ("keep", "remove")


def _coerce(default_val, new):
    if isinstance(default_val, bool):
        return bool(new)
    if isinstance(default_val, int) and not isinstance(default_val, bool):
        return int(new)
    if isinstance(default_val, float):
        return float(new)
    return new  # string


def _deep_merge(defaults: dict, raw) -> dict:
    """Recursively merge `raw` over `defaults`, coercing leaves to the default type;
    unknown/malformed entries fall back to defaults."""
    out = {}
    for k, dv in defaults.items():
        if isinstance(dv, dict):
            sub = raw.get(k) if isinstance(raw, dict) else None
            out[k] = _deep_merge(dv, sub if isinstance(sub, dict) else {})
        elif isinstance(raw, dict) and k in raw:
            try:
                out[k] = _coerce(dv, raw[k])
            except (TypeError, ValueError):
                out[k] = dv
        else:
            out[k] = dv
    return out


def load_config(path: str = None) -> dict:
    """Full config (JSON deep-merged over DEFAULT_CONFIG, type-coerced). A bad edit
    never crashes a render."""
    import json
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)
    raw = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception:  # noqa: BLE001
            raw = {}
    cfg = _deep_merge(DEFAULT_CONFIG, raw)
    for lp in cfg.get("layouts", {}).values():   # validate enums
        if lp.get("out_of_frustum") not in _OUT_OF_FRUSTUM_CHOICES:
            lp["out_of_frustum"] = "keep"
    return cfg


def load_holo_tuning(path: str = None) -> dict:
    """Holo tuning section (back-compat for finishes.py)."""
    return load_config(path)["holo"]


def load_layout_params(layout: str, path: str = None) -> dict:
    """Per-layout scene params (e.g. 'table', 'floating')."""
    return load_config(path)["layouts"].get(layout, {})
