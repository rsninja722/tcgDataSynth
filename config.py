"""
Project-wide configuration and physical constants (bpy-FREE, Docker-testable).

All Blender-side code imports these values instead of hardcoding them, so paths
and defaults live in exactly one place. Nothing here imports `bpy`.

UNITS: Blender scene works in METERS at real-world scale. Constants below give
both the millimetre spec value and the metre value actually used.
"""
from __future__ import annotations

import os
import math
import json
from dataclasses import dataclass, field
from typing import Tuple

# Compact CC0 left/right rig library bundled with the project. It contains only the
# hand meshes, armatures, and their required dependencies. TCG_HAND_ASSET remains an
# explicit override for diagnostics with another compatible library.
HAND_ASSET_FILENAME = "hand_rig.blend"
DEFAULT_HAND_ASSET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", HAND_ASSET_FILENAME)


def hand_asset_path() -> str:
    """Resolved hand library path: env override wins, else the bundled asset."""
    return os.environ.get("TCG_HAND_ASSET", DEFAULT_HAND_ASSET_PATH)

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
    root = card_image_root()
    if not root:
        return ""
    p = os.path.join(root, BACK_IMAGE_FILENAME)
    return p if os.path.isfile(p) else ""

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
CYCLES_SAMPLES = 32                    # with denoising; tune for speed in Phase 8
CYCLES_DEVICE = "GPU"                   # user configured CUDA
EEVEE_RENDER_SAMPLES = 32              # scene.eevee.taa_render_samples

# Color management view transform, LOCKED (Phase 0 confirmed 'AgX' is the 5.0
# default & active). postfx assumes this stays constant across the whole dataset.
VIEW_TRANSFORM = "AgX"

# --------------------------------------------------------------------------- #
# Label format (spec §3.9)
# --------------------------------------------------------------------------- #
YOLO_CLASS_ID = 0        # 'card': fully in-frustum, exactly 4 corner keypoints
PARTIAL_CLASS_ID = 1     # 'partial_card': partially in-frustum, 3-8 boundary points
CLASS_NAMES = ("card", "partial_card")
# Keypoint order in the card's own upright frame (user decision 2026-07-19).
KEYPOINT_ORDER = ("TL", "TR", "BR", "BL")
KPT_SHAPE = (4, 3)  # 4 keypoints, (x, y, visibility) for a full 'card'
# A 'partial_card' carries a VARIABLE number of keypoints: the exact outline of the
# card's VISIBLE region = the polygon (card quad) ∩ (frustum square). That polygon's
# vertices are the in-frustum card corners, the points where card edges cross the
# frustum boundary, AND any frustum CORNER the card covers (an interior-of-card point
# that sits on the frame corner). A quad ∩ square gives up to 8 vertices; boundary
# vertices always have a normalized component == 0 or 1.
PARTIAL_KPT_RANGE = (3, 8)
KPT_VISIBILITY = 2  # all kept keypoints flagged visible per spec


@dataclass(frozen=True)
class OutputLayout:
    """Where renders, labels, and manifests are written under out/."""
    root: str = "out"
    images_subdir: str = "images"
    labels_subdir: str = "labels"
    yolo_labels_subdir: str = "labels_yolo"
    extra_labels_subdir: str = "extra_label"
    manifest_name: str = "manifest.jsonl"
    refraction_failures_name: str = "refraction_failures.txt"

    def images_dir(self) -> str:
        return os.path.join(self.root, self.images_subdir)

    def labels_dir(self) -> str:
        return os.path.join(self.root, self.labels_subdir)

    def yolo_labels_dir(self) -> str:
        return os.path.join(self.root, self.yolo_labels_subdir)

    def extra_labels_dir(self) -> str:
        return os.path.join(self.root, self.extra_labels_subdir)

    def refraction_failures_path(self) -> str:
        return os.path.join(self.root, self.refraction_failures_name)


OUTPUT = OutputLayout()


# --------------------------------------------------------------------------- #
# Central runtime config (single hand-editable JSON; scripts load at runtime so
# it can be tuned without touching code). All tunable params live here.
# --------------------------------------------------------------------------- #
CONFIG_FILENAME = "config.json"
DEFAULT_CONFIG = {
    "blender_executable": r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
    "table_texture_dir": "",
    "card_image_root": "",
    "holo": {
        "angle_gain": 12.0,    # flash-band motion speed vs view angle
        "pattern_gain": 4.0,   # spatial break-up of the flash phase by the pattern
        "sharpness": 4.0,      # flash threshold (higher = narrower/more selective)
        "emit": 1.6,           # peak flash brightness
        "darken": 0.18,        # base darkening between flashes
    },
    "camera": {
        "aperture_fstop_range": [1.8, 8.0],
    },
    "lighting": {
        "shadow_plane_opacity": 0.95,
    },
    # Each effect has a per-image enable probability and all of its sampled-value
    # ranges. config.json is the user-editable source; these are safe fallbacks.
    "postfx": {
        "motion_blur": {
            "probability": 1.0,
            "strength_range": [0.12, 0.22],
        },
        "sensor_noise": {
            "probability": 0.45,
            "luma_sigma_range": [0.002, 0.012],
            "chroma_sigma_range": [0.001, 0.006],
        },
        "compression": {
            "probability": 0.40,
            "jpeg_quality_range": [25, 70],
            "cycles_range": [1, 3],
        },
        "pixel_melt": {
            "probability": 0.35,
            "blur_sigma_range": [0.35, 1.15],
            "scale_range": [0.55, 0.85],
        },
        "white_balance": {
            "probability": 0.40,
            "temperature_shift_k_range": [-450.0, 450.0],
        },
        "tint": {
            "probability": 0.35,
            "green_magenta_shift_range": [-0.04, 0.04],
        },
        "chromatic_aberration": {
            "probability": 0.30,
            "shift_px_range": [0.25, 1.5],
        },
        "contrast": {
            "probability": 0.40,
            "reduction_range": [0.0, 0.50],
        },
        "haze": {
            "probability": 0.30,
            "strength_range": [0.02, 0.18],
            "brightness_range": [0.75, 1.0],
        },
    },
    # Persisted by the Phase 7 add-on. These keys map directly to
    # rules.combinations.default_enabled_options() without importing that module.
    "generation": {
        "count": 10,
        "base_seed": 20260731,
        "export_yolo_segmentation": False,
        "enabled_options": {
            "layouts": ["table", "floating", "binder", "display_case", "hand", "stack"],
            "protections": ["none", "sleeve", "semi_rigid", "toploader", "slab"],
            "sleeve_types": ["clear", "opaque_back"],
            "sleeve_sizes": ["1mm", "2.5mm"],
            "finishes": ["normal", "holo"],
            "holo_regions": ["entire", "picture", "reverse"],
            "holo_patterns": ["none", "cosmos", "horizontal_lines", "water_web"],
            "physical_texture": True,
            "damage": ["dirt", "scratches", "surface"],
            "binder_grids": ["1x1", "2x2", "3x3", "4x3"],
            "binder_contents": ["sleeved", "toploader", "slab"],
            "lighting": {"spotlight": True, "point_lights": True, "occluders": True},
            "post_effects": ["sensor_noise", "compression", "pixel_melt", "white_balance",
                             "tint", "chromatic_aberration", "contrast", "haze",
                             "motion_blur"],
            "back_to_camera_prob": 0.15,
            "cardless_scene_prob": 0.0,
        },
    },
    # Per-layout scene params (each scene type has its OWN set). Add entries here as
    # layouts are built (out_of_frustum: 'keep' = render but don't label | 'remove').
    "layouts": {
        "table": {"max_cards": 8, "allow_overlap": False, "out_of_frustum": "keep",
                  "camera_max_offaxis_deg": 50.0},
        "floating": {"max_cards": 12, "max_shapes": 12, "allow_overlap": False,
                     "out_of_frustum": "keep", "camera_max_offaxis_deg": 50.0},
        "binder": {"max_cards": 12, "out_of_frustum": "keep",
                   "camera_max_offaxis_deg": 50.0},
        "display_case": {"max_cards": 24, "out_of_frustum": "keep",
                         "camera_max_offaxis_deg": 30.0},
        "hand": {"max_cards": 1, "out_of_frustum": "keep",
                  "camera_max_offaxis_deg": 50.0},
        "stack": {"max_cards": 10, "out_of_frustum": "keep",
                  "camera_max_offaxis_deg": 50.0},
    },
}
# Back-compat alias (used by tests / older references).
DEFAULT_HOLO_TUNING = DEFAULT_CONFIG["holo"]
_OUT_OF_FRUSTUM_CHOICES = ("keep", "remove")
_POSTFX_INTEGER_RANGES = {("compression", "jpeg_quality_range"),
                          ("compression", "cycles_range")}
_POSTFX_RANGE_BOUNDS = {
    ("motion_blur", "strength_range"): (0.0, 1.0),
    ("sensor_noise", "luma_sigma_range"): (0.0, 1.0),
    ("sensor_noise", "chroma_sigma_range"): (0.0, 1.0),
    ("pixel_melt", "blur_sigma_range"): (0.0, 50.0),
    ("pixel_melt", "scale_range"): (0.01, 1.0),
    ("white_balance", "temperature_shift_k_range"): (-10000.0, 10000.0),
    ("tint", "green_magenta_shift_range"): (-1.0, 1.0),
    ("chromatic_aberration", "shift_px_range"): (0.0, 100.0),
    ("contrast", "reduction_range"): (0.0, 0.5),
    ("haze", "strength_range"): (0.0, 1.0),
    ("haze", "brightness_range"): (0.0, 1.0),
}
_CAMERA_FSTOP_ALLOWED_RANGE = (0.1, 64.0)
_CAMERA_OFFAXIS_ALLOWED_RANGE = (0.0, 89.0)
GENERATION_SEED_MAX = 2 ** 63 - 1


def _valid_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _validated_postfx_tuning(tuning: dict) -> dict:
    """Return safe post-effect tuning with probabilities and numeric ranges normalized."""
    defaults = DEFAULT_CONFIG["postfx"]
    out = {}
    for effect, default_values in defaults.items():
        candidate = tuning.get(effect, {}) if isinstance(tuning, dict) else {}
        values = {}
        probability = candidate.get("probability", default_values["probability"])
        if not _valid_number(probability):
            probability = default_values["probability"]
        values["probability"] = min(1.0, max(0.0, float(probability)))
        for key, default_range in default_values.items():
            if key == "probability":
                continue
            supplied = candidate.get(key, default_range)
            if not isinstance(supplied, (list, tuple)) or len(supplied) != 2 \
                    or not all(_valid_number(value) for value in supplied):
                supplied = default_range
            low, high = sorted(float(value) for value in supplied)
            if (effect, key) in _POSTFX_INTEGER_RANGES:
                low, high = int(round(low)), int(round(high))
                if key == "jpeg_quality_range":
                    low, high = max(1, low), min(100, high)
                else:
                    low, high = max(1, low), max(1, high)
                if low > high:
                    low, high = default_range
            else:
                allowed_low, allowed_high = _POSTFX_RANGE_BOUNDS[(effect, key)]
                if low < allowed_low or high > allowed_high:
                    low, high = default_range
            values[key] = [low, high]
        out[effect] = values
    return out


def _validated_camera_tuning(tuning: dict) -> dict:
    """Return a physically valid, ordered aperture sampling range."""
    default = DEFAULT_CONFIG["camera"]["aperture_fstop_range"]
    supplied = tuning.get("aperture_fstop_range", default) if isinstance(tuning, dict) else default
    if not isinstance(supplied, (list, tuple)) or len(supplied) != 2 \
            or not all(_valid_number(value) for value in supplied):
        supplied = default
    low, high = sorted(float(value) for value in supplied)
    if low < _CAMERA_FSTOP_ALLOWED_RANGE[0] or high > _CAMERA_FSTOP_ALLOWED_RANGE[1]:
        low, high = default
    return {"aperture_fstop_range": [low, high]}


def _validated_lighting_tuning(tuning: dict) -> dict:
    """Return normalized lighting values used directly by generated scenes."""
    default = DEFAULT_CONFIG["lighting"]["shadow_plane_opacity"]
    supplied = tuning.get("shadow_plane_opacity", default) if isinstance(tuning, dict) else default
    if not _valid_number(supplied):
        supplied = default
    return {"shadow_plane_opacity": min(1.0, max(0.0, float(supplied)))}


def _validated_generation(settings: dict) -> dict:
    """Return safe persisted GUI settings while preserving deliberate empty toggles."""
    defaults = DEFAULT_CONFIG["generation"]
    settings = settings if isinstance(settings, dict) else {}
    out = {}
    for key in ("count", "base_seed"):
        value = settings.get(key, defaults[key])
        if not isinstance(value, int) or isinstance(value, bool):
            value = defaults[key]
        out[key] = int(value)
    if not 1 <= out["count"] <= GENERATION_SEED_MAX:
        out["count"] = defaults["count"]
    if not 0 <= out["base_seed"] <= GENERATION_SEED_MAX - out["count"]:
        out["base_seed"] = defaults["base_seed"]
    export_yolo = settings.get(
        "export_yolo_segmentation", defaults["export_yolo_segmentation"])
    out["export_yolo_segmentation"] = (
        export_yolo if isinstance(export_yolo, bool) else defaults["export_yolo_segmentation"])

    supplied = settings.get("enabled_options", {})
    supplied = supplied if isinstance(supplied, dict) else {}
    option_defaults = defaults["enabled_options"]
    options = {}
    for key, allowed in option_defaults.items():
        value = supplied.get(key, allowed)
        if isinstance(allowed, list):
            if not isinstance(value, list):
                value = allowed
            options[key] = [item for item in allowed if item in value]
        elif isinstance(allowed, dict):
            value = value if isinstance(value, dict) else {}
            options[key] = {name: bool(value.get(name, default))
                            for name, default in allowed.items()}
        elif isinstance(allowed, bool):
            options[key] = bool(value)
        else:
            if not _valid_number(value):
                value = allowed
            options[key] = min(1.0, max(0.0, float(value)))
    out["enabled_options"] = options
    return out


def _coerce(default_val, new):
    if isinstance(default_val, bool):
        if isinstance(new, bool):
            return new
        if isinstance(new, (int, float)) and new in (0, 1):
            return bool(new)
        return default_val
    if isinstance(default_val, int) and not isinstance(default_val, bool):
        return int(new)
    if isinstance(default_val, float):
        return float(new)
    if isinstance(default_val, str):
        return new if isinstance(new, str) else default_val
    return new


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
    cfg["camera"] = _validated_camera_tuning(cfg["camera"])
    cfg["lighting"] = _validated_lighting_tuning(cfg["lighting"])
    cfg["postfx"] = _validated_postfx_tuning(cfg["postfx"])
    cfg["generation"] = _validated_generation(cfg["generation"])
    for name, lp in cfg.get("layouts", {}).items():
        if not isinstance(lp.get("max_cards"), int) or isinstance(lp.get("max_cards"), bool) \
                or lp["max_cards"] < 1:
            lp["max_cards"] = DEFAULT_CONFIG["layouts"][name]["max_cards"]
        if lp.get("out_of_frustum") not in _OUT_OF_FRUSTUM_CHOICES:
            lp["out_of_frustum"] = "keep"
        maximum = lp.get("camera_max_offaxis_deg")
        if not _valid_number(maximum) or not (
                _CAMERA_OFFAXIS_ALLOWED_RANGE[0] <= maximum
                <= _CAMERA_OFFAXIS_ALLOWED_RANGE[1]):
            lp["camera_max_offaxis_deg"] = \
                DEFAULT_CONFIG["layouts"][name]["camera_max_offaxis_deg"]
    return cfg


def load_holo_tuning(path: str = None) -> dict:
    """Holo tuning section (back-compat for finishes.py)."""
    return load_config(path)["holo"]


def load_layout_params(layout: str, path: str = None) -> dict:
    """Per-layout scene params (e.g. 'table', 'floating')."""
    return load_config(path)["layouts"].get(layout, {})


def load_postfx_tuning(path: str = None) -> dict:
    """Validated post-effect probabilities and sampled-value ranges."""
    return load_config(path)["postfx"]


def load_camera_tuning(path: str = None) -> dict:
    """Validated camera sampling tuning."""
    return load_config(path)["camera"]


def load_lighting_tuning(path: str = None) -> dict:
    """Validated lighting tuning."""
    return load_config(path)["lighting"]


def load_generation_settings(path: str = None) -> dict:
    """Validated persisted Phase 7 GUI settings."""
    return load_config(path)["generation"]


def load_blender_executable(path: str = None) -> str:
    """Configured Blender executable for the standalone Phase 7 GUI."""
    return str(load_config(path)["blender_executable"])


def load_table_texture_dir(path: str = None) -> str:
    """Configured image directory used for table surface textures."""
    value = str(load_config(path)["table_texture_dir"]).strip()
    if not value or os.path.isabs(value):
        return value
    base = os.path.dirname(os.path.abspath(path)) if path else os.path.dirname(
        os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, value))

def card_image_root(path: str = None) -> str:
    """Configured card-library directory, resolved relative to config.json."""
    value = str(load_config(path)["card_image_root"]).strip()
    if not value or os.path.isabs(value):
        return value
    base = os.path.dirname(os.path.abspath(path)) if path else os.path.dirname(
        os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, value))
    

def _read_raw_config(path: str) -> dict:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if isinstance(raw, dict):
                return raw
        except Exception:  # noqa: BLE001
            pass
    return {}


def _write_raw_config(path: str, raw: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    staged = path + ".generation.tmp"
    with open(staged, "w", encoding="utf-8") as handle:
        json.dump(raw, handle, indent=2)
        handle.write("\n")
    os.replace(staged, path)


def save_generation_settings(settings: dict, path: str = None) -> None:
    """Atomically persist the GUI's generation controls without rewriting other config."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)
    raw = _read_raw_config(path)
    raw["generation"] = _validated_generation(settings)
    _write_raw_config(path, raw)


def save_blender_executable(executable: str, path: str = None) -> None:
    """Atomically persist the standalone GUI's Blender executable path."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)
    if not isinstance(executable, str) or not executable.strip():
        raise ValueError("Blender executable path must be a non-empty string")
    raw = _read_raw_config(path)
    raw["blender_executable"] = executable.strip()
    _write_raw_config(path, raw)


def save_table_texture_dir(directory: str, path: str = None) -> None:
    """Atomically persist the table texture image directory."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)
    if not isinstance(directory, str):
        raise ValueError("Table texture directory must be a string")
    raw = _read_raw_config(path)
    raw["table_texture_dir"] = directory.strip()
    _write_raw_config(path, raw)
