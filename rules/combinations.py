"""
Combination rules & scene-config sampling (bpy-FREE, Docker-tested).

Single entry point:  sample_scene_config(enabled_options, rng_seed) -> SceneConfig

This module owns EVERY legality decision (spec §3.2/§3.4/§3.5/§3.6). Blender code
consumes a validated SceneConfig (via .to_dict()/JSON) and never decides legality.
`validate_scene_config` re-checks all rules and is used both as a runtime safety
net and by the Docker audit test.

Card *image* selection is intentionally NOT done here (it needs disk access that
the Docker container lacks). Each CardConfig carries abstract protection/finish/
damage; the Blender side assigns actual card images with the same seed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

import config as project_config

# --------------------------------------------------------------------------- #
# Option value vocabularies (the GUI toggles enable/disable these)
# --------------------------------------------------------------------------- #
LAYOUTS = ("table", "floating", "binder", "display_case", "hand")
PROTECTIONS = ("none", "sleeve", "semi_rigid", "toploader", "slab")
SLEEVE_TYPES = ("clear", "opaque_back")
SLEEVE_SIZES = ("1mm", "2.5mm")
FINISHES = ("normal", "holo")
HOLO_REGIONS = ("entire", "picture", "reverse")   # reverse = everything EXCEPT picture
HOLO_PATTERNS = ("none", "cosmos", "horizontal_lines", "water_web")
DAMAGE_KINDS = ("dirt", "scratches", "surface")
POST_EFFECTS = (
    "motion_blur", "sensor_noise", "compression", "pixel_melt", "white_balance", "tint",
    "chromatic_aberration", "contrast", "haze",
)
BINDER_GRIDS = ("1x1", "2x2", "3x3", "4x3")
BINDER_CONTENTS = ("sleeved", "toploader", "slab")  # what the binder is sized for
HAND_GRIPS = ("side", "pinch")
HAND_SIDES = ("left", "right")
HAND_DEPTH_RANGE = (0.1, 0.34)  # deeper contacts clipped into the card during t14 review
DISPLAY_CASE_MAX_CARDS = 24   # cap on the display-case grid (user)
# A decorative card may rest ON TOP of a display case; it can be any of these
# (NOT semi-rigid), independent of the grid's toploader/slab restriction (user).
DISPLAY_CASE_TOP_PROTECTIONS = ("none", "sleeve", "toploader", "slab")

HOLDER_MAX_ROT_DEG = 1.0    # max card rotation inside a semi-rigid/toploader (user)
HOLDER_MAX_OFFSET_MM = 2.0  # max card offset inside a holder

SUN_ELEVATION_RANGE = (15.0, 75.0)
SUN_AZIMUTH_RANGE = (-60.0, 60.0)
SUN_ENERGY_RANGE = (0.028125, 0.16875)  # low ambient fill; non-sun lights provide key light
# About twice the reciprocal color warmth of t16's warm_points tile (3962.6 K).
POINT_TEMP_RANGE = (2000.0, 9000.0)
POINT_ENERGY_RANGE = (1.125, 22.5)
POINT_ENERGY_MAX_REDUCTION = {2: 8.0, 3: 15.0}
POINT_XY_RANGE = (-0.30, 0.30)
POINT_Z_RANGE = (0.05, 0.40)
NON_SUN_SHADOW_MASK_PROB = 0.25
# Blender ID custom properties store Python ints as signed C ints. This is an
# exclusive upper bound, so generated values are always 0..INT32_MAX.
SHADOW_MASK_SEED_MAX = 2 ** 31
CAMERA_FOCAL_RANGE = (15.0, 55.0)
CAMERA_OFFAXIS_RANGE = (0.0, 50.0)
CAMERA_OFFAXIS_ALLOWED_RANGE = (0.0, 89.0)
CAMERA_ORBIT_RANGE = (0.0, 360.0)
CAMERA_FSTOP_RANGE = (1.8, 8.0)
CAMERA_FSTOP_ALLOWED_RANGE = (0.1, 64.0)

# Layout -> the set of per-card protections that layout allows (spec §3.5).
_LAYOUT_PROTECTIONS: Dict[str, tuple] = {
    "table": PROTECTIONS,                       # anything, or bare
    "floating": PROTECTIONS,                    # anything
    "display_case": ("toploader", "slab"),      # tight grid of toploadered/slabbed
    "hand": ("none", "sleeve", "toploader"),    # side grip: sleeved/toploadered; pinch: also bare
    # binder is handled specially: its content_type fixes the protection.
}
_BINDER_CONTENT_PROTECTION = {"sleeved": "sleeve", "toploader": "toploader", "slab": "slab"}


class ConfigError(ValueError):
    """Raised when enabled_options make a legal scene impossible."""


# --------------------------------------------------------------------------- #
# Config dataclasses (all JSON-serializable via asdict)
# --------------------------------------------------------------------------- #
@dataclass
class SleeveConfig:
    sleeve_type: str          # SLEEVE_TYPES
    size: str                 # SLEEVE_SIZES


@dataclass
class ProtectionConfig:
    kind: str                             # PROTECTIONS
    sleeve: Optional[SleeveConfig] = None  # present iff the card is sleeved
    # semi-rigid / toploader: card offset ±2mm and rotated ±2deg inside (spec §3.2)
    inner_offset_mm: Optional[List[float]] = None  # [dx, dy]
    inner_rot_deg: Optional[float] = None


@dataclass
class FinishConfig:
    kind: str                          # FINISHES
    holo_region: Optional[str] = None  # HOLO_REGIONS when holo
    holo_pattern: Optional[str] = None  # HOLO_PATTERNS when holo
    physical_texture: bool = False     # etched-foil normal map (holo only)


@dataclass
class DamageConfig:
    dirt: bool = False
    scratches: bool = False
    surface: bool = False


@dataclass
class CardConfig:
    slot_index: int
    protection: ProtectionConfig
    finish: FinishConfig
    damage: DamageConfig
    back_to_camera: bool = False  # if True this card shows its back (NOT labeled)


@dataclass
class PointLightConfig:
    color_temp: float   # kelvin-ish, warm(3000)..cold(9000)
    intensity: float    # watts
    position: List[float]  # xyz, front hemisphere
    shadow_mask_seed: Optional[int] = None


@dataclass
class LightingConfig:
    sun_angle_deg: List[float]           # [elevation, azimuth], front hemisphere
    sun_energy: float
    spotlight_beside_camera: bool
    point_lights: List[PointLightConfig] = field(default_factory=list)
    spotlight_shadow_mask_seed: Optional[int] = None
    shadow_plane_opacity: float = 0.95


@dataclass
class CameraConfig:
    focal_mm: float
    offaxis_deg: float
    orbit_deg: float
    dof_enabled: bool
    aperture_fstop: float


@dataclass
class SensorNoiseConfig:
    luma_sigma: float
    chroma_sigma: float
    seed: int


@dataclass
class MotionBlurConfig:
    direction_index: int
    strength: float


@dataclass
class CompressionConfig:
    jpeg_quality: int
    cycles: int


@dataclass
class PixelMeltConfig:
    blur_sigma: float
    scale: float


@dataclass
class WhiteBalanceConfig:
    temperature_shift_k: float


@dataclass
class TintConfig:
    green_magenta_shift: float


@dataclass
class ChromaticAberrationConfig:
    shift_px: float
    angle_deg: float


@dataclass
class ContrastConfig:
    reduction: float


@dataclass
class HazeConfig:
    strength: float
    brightness: float


@dataclass
class PostFxConfig:
    """Fully sampled image effects. ``None`` means that effect is disabled."""
    motion_blur: Optional[MotionBlurConfig] = None
    sensor_noise: Optional[SensorNoiseConfig] = None
    compression: Optional[CompressionConfig] = None
    pixel_melt: Optional[PixelMeltConfig] = None
    white_balance: Optional[WhiteBalanceConfig] = None
    tint: Optional[TintConfig] = None
    chromatic_aberration: Optional[ChromaticAberrationConfig] = None
    contrast: Optional[ContrastConfig] = None
    haze: Optional[HazeConfig] = None


@dataclass
class LayoutConfig:
    kind: str                                  # LAYOUTS
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneConfig:
    seed: int
    layout: LayoutConfig
    cards: List[CardConfig]
    lighting: LightingConfig
    camera: CameraConfig
    postfx: PostFxConfig

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)


# --------------------------------------------------------------------------- #
# Enabled-options handling
# --------------------------------------------------------------------------- #
def default_enabled_options() -> Dict[str, Any]:
    """Everything on. The GUI overrides subsets of this."""
    return {
        "layouts": list(LAYOUTS),
        "protections": list(PROTECTIONS),
        "sleeve_types": list(SLEEVE_TYPES),
        "sleeve_sizes": list(SLEEVE_SIZES),
        "finishes": list(FINISHES),
        "holo_regions": list(HOLO_REGIONS),
        "holo_patterns": list(HOLO_PATTERNS),
        "physical_texture": True,
        "damage": list(DAMAGE_KINDS),
        "binder_grids": list(BINDER_GRIDS),
        "binder_contents": list(BINDER_CONTENTS),
        "lighting": {"spotlight": True, "point_lights": True, "occluders": True},
        "post_effects": list(POST_EFFECTS),
        "back_to_camera_prob": 0.15,  # chance a given card faces away (table/floating)
    }


def _resolve(enabled_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge caller options over defaults (shallow, one level for nested dicts)."""
    opts = default_enabled_options()
    if enabled_options:
        for k, v in enabled_options.items():
            if isinstance(v, dict) and isinstance(opts.get(k), dict):
                opts[k] = {**opts[k], **v}
            else:
                opts[k] = v
    return opts


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #
def _choice(rng: np.random.Generator, seq):
    seq = list(seq)
    return seq[int(rng.integers(0, len(seq)))]


def _maybe(rng: np.random.Generator, p: float) -> bool:
    return bool(rng.random() < p)


def _require(seq, label: str):
    seq = list(seq)
    if not seq:
        raise ConfigError(f"No enabled options for {label}.")
    return seq


def _sleeve(rng, opts) -> SleeveConfig:
    return SleeveConfig(
        sleeve_type=_choice(rng, _require(opts["sleeve_types"], "sleeve_types")),
        size=_choice(rng, _require(opts["sleeve_sizes"], "sleeve_sizes")),
    )


def _sample_protection(rng, opts, kind: str) -> ProtectionConfig:
    """Build a ProtectionConfig for a fixed `kind`, applying implication rules:
    toploader/semi_rigid ALWAYS sleeved; slab/none never; sleeve is sleeved."""
    sleeve = None
    inner_offset = None
    inner_rot = None
    if kind in ("sleeve", "toploader", "semi_rigid"):
        sleeve = _sleeve(rng, opts)
    if kind in ("semi_rigid", "toploader"):
        # card offset ±2mm and rotated up to ±1deg inside the holder (user tightened
        # from the spec's ±2deg).
        inner_offset = [float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-2.0, 2.0))]
        inner_rot = float(rng.uniform(-HOLDER_MAX_ROT_DEG, HOLDER_MAX_ROT_DEG))
    return ProtectionConfig(kind=kind, sleeve=sleeve,
                            inner_offset_mm=inner_offset, inner_rot_deg=inner_rot)


def _sample_finish(rng, opts) -> FinishConfig:
    kind = _choice(rng, _require(opts["finishes"], "finishes"))
    if kind != "holo":
        return FinishConfig(kind="normal")
    patterns = _require(opts["holo_patterns"], "holo_patterns")
    physical = bool(opts.get("physical_texture", True)) and _maybe(rng, 0.5)
    # Rule: a physical-texture card is ALWAYS paired with the 'none' holo pattern
    # (the etched lines are the structure; other patterns don't co-occur with it).
    if physical and "none" in patterns:
        pattern = "none"
    else:
        physical = False
        pattern = _choice(rng, patterns)
    return FinishConfig(
        kind="holo",
        holo_region=_choice(rng, _require(opts["holo_regions"], "holo_regions")),
        holo_pattern=pattern,
        physical_texture=physical,
    )


def _sample_damage(rng, opts) -> DamageConfig:
    enabled = set(opts.get("damage", []))
    # Each kind independently on/off (spec §3.3); ~35% each when enabled.
    return DamageConfig(
        dirt=("dirt" in enabled) and _maybe(rng, 0.35),
        scratches=("scratches" in enabled) and _maybe(rng, 0.35),
        surface=("surface" in enabled) and _maybe(rng, 0.35),
    )


def _allowed_protections_for_layout(layout_kind: str, opts) -> List[str]:
    """Intersect the layout's allowed protections with the globally-enabled set."""
    enabled = set(opts["protections"])
    allowed = _LAYOUT_PROTECTIONS.get(layout_kind, PROTECTIONS)
    return [p for p in allowed if p in enabled]


def _selectable_layouts(opts) -> List[str]:
    """Layouts that are enabled AND can be satisfied given enabled protections."""
    out = []
    for lk in opts["layouts"]:
        if lk == "binder":
            contents = [c for c in opts.get("binder_contents", [])
                        if _BINDER_CONTENT_PROTECTION[c] in opts["protections"]]
            if contents and opts.get("binder_grids"):
                out.append(lk)
        else:
            if _allowed_protections_for_layout(lk, opts):
                out.append(lk)
    return out


# --------------------------------------------------------------------------- #
# Layout + card sampling
# --------------------------------------------------------------------------- #
def _grid_capacity(grid: str) -> int:
    a, b = grid.split("x")
    return int(a) * int(b)


def _sample_layout_and_cards(rng, opts):
    layouts = _selectable_layouts(opts)
    if not layouts:
        raise ConfigError("No layout can be satisfied with the enabled options "
                          "(check that some protection is enabled for each layout).")
    kind = _choice(rng, layouts)

    if kind == "binder":
        return _binder(rng, opts)
    if kind == "display_case":
        return _display_case(rng, opts)
    if kind == "hand":
        return _hand(rng, opts)
    return _loose(rng, opts, kind)  # table / floating


def _make_card(rng, opts, slot_index, protection_kind, allow_back=True) -> CardConfig:
    back = bool(allow_back and _maybe(rng, opts.get("back_to_camera_prob", 0.15)))
    # A card showing its back is not sleeved/holo-visible in practice, but we keep
    # its protection/finish for geometry; labeling simply skips it (back_to_camera).
    return CardConfig(
        slot_index=slot_index,
        protection=_sample_protection(rng, opts, protection_kind),
        finish=_sample_finish(rng, opts),
        damage=_sample_damage(rng, opts),
        back_to_camera=back,
    )


def _binder(rng, opts):
    contents = [c for c in opts["binder_contents"]
                if _BINDER_CONTENT_PROTECTION[c] in opts["protections"]]
    content = _choice(rng, contents)
    grid = _choice(rng, opts["binder_grids"])
    cap = _grid_capacity(grid)
    protection_kind = _BINDER_CONTENT_PROTECTION[content]
    # Not every slot filled (spec §3.5.1); at least one card.
    n_filled = int(rng.integers(1, cap + 1))
    filled_slots = sorted(rng.choice(cap, size=n_filled, replace=False).tolist())
    cards = [_make_card(rng, opts, s, protection_kind, allow_back=False) for s in filled_slots]
    params = {
        "grid": grid, "content_type": content,
        "page_color": "clear" if _maybe(rng, 0.5) else "solid",
        "slot_gap_mm": float(rng.uniform(7.0, 18.0)),
        "two_pages": _maybe(rng, 0.4),
        "side": "left" if _maybe(rng, 0.5) else "right",
        "capacity": cap, "filled_slots": filled_slots,
    }
    return LayoutConfig("binder", params), cards


def _display_case(rng, opts):
    allowed = _allowed_protections_for_layout("display_case", opts)
    cols = int(rng.integers(2, 6))
    rows = int(rng.integers(2, 6))
    # Tight grid capped at DISPLAY_CASE_MAX_CARDS (user). Keep `cols`, trim card
    # count, and recompute `rows` so the last row may be partial (an empty corner).
    n = min(cols * rows, DISPLAY_CASE_MAX_CARDS)
    rows = (n + cols - 1) // cols
    tilt = _maybe(rng, 0.5)
    cards = [_make_card(rng, opts, i, _choice(rng, allowed), allow_back=False)
             for i in range(n)]
    params = {"cols": cols, "rows": rows, "tilt_forward": tilt,
              "tilt_deg": 25.0 if tilt else 0.0, "cover_scratches": True}
    return LayoutConfig("display_case", params), cards


def sample_top_card(rng, enabled_options=None) -> CardConfig:
    """Sample ONE decorative card (none/sleeve/toploader/slab) to rest on top of a
    display case. Honors per-card legality (sleeve presence/type/size) via the same
    path as scene cards, but is NOT limited to the grid's toploader/slab set.
    Determinism comes entirely from the caller's `rng`; always front-facing."""
    opts = _resolve(enabled_options)
    choices = [p for p in DISPLAY_CASE_TOP_PROTECTIONS if p in opts["protections"]]
    if not choices:
        raise ConfigError("No enabled protection is valid for a display-case top card.")
    kind = _choice(rng, choices)
    return _make_card(rng, opts, slot_index=-1, protection_kind=kind, allow_back=False)


def _hand(rng, opts):
    allowed = _allowed_protections_for_layout("hand", opts)
    protection_kind = _choice(rng, allowed)
    # Bare card => pinch grip only; sleeved/toploadered => either grip.
    if protection_kind == "none":
        grip = "pinch"
    else:
        grip = _choice(rng, HAND_GRIPS)
    card = _make_card(rng, opts, 0, protection_kind, allow_back=False)
    params = {
        "grip": grip,
        "handedness": _choice(rng, HAND_SIDES),
        "approach_deg": float(rng.uniform(0.0, 360.0)),
        # Fraction controlling how far the contact point moves inward from the
        # protection boundary. This is dimensionless, not a world-space distance.
        "depth": float(rng.uniform(*HAND_DEPTH_RANGE)),
    }
    return LayoutConfig("hand", params), [card]


def _loose(rng, opts, kind):
    """table / floating: 1..N cards, any enabled protection for that layout."""
    allowed = _allowed_protections_for_layout(kind, opts)
    n = int(rng.integers(1, 6))
    cards = [_make_card(rng, opts, i, _choice(rng, allowed), allow_back=True)
             for i in range(n)]
    if kind == "table":
        params = {"max_overlap_frac": 0.15, "clutter_rects": int(rng.integers(0, 6))}
    else:
        params = {"bg_prisms": int(rng.integers(1, 6)), "bg_cylinders": int(rng.integers(0, 5))}
    return LayoutConfig(kind, params), cards


# --------------------------------------------------------------------------- #
# Lighting / camera / postfx
# --------------------------------------------------------------------------- #
def _point_energy_range(n_points: int) -> tuple[float, float]:
    reduction = POINT_ENERGY_MAX_REDUCTION.get(n_points, 0.0)
    return POINT_ENERGY_RANGE[0], POINT_ENERGY_RANGE[1] - reduction


def _sample_lighting(rng, opts, tuning: Dict[str, Any]) -> LightingConfig:
    lopt = opts.get("lighting", {})
    spotlight = bool(lopt.get("spotlight", True)) and _maybe(rng, 0.5)
    n_points = int(rng.integers(0, 5)) if lopt.get("point_lights", True) else 0
    # Constraint (spec §3.6): at least one NON-sun light must exist.
    if not spotlight and n_points == 0:
        if lopt.get("spotlight", True):
            spotlight = True
        else:
            n_points = 1
    point_energy_range = _point_energy_range(n_points)
    points = [
        PointLightConfig(
            color_temp=float(rng.uniform(*POINT_TEMP_RANGE)),
            intensity=float(rng.uniform(*point_energy_range)),
            position=[float(rng.uniform(*POINT_XY_RANGE)),
                      float(rng.uniform(*POINT_XY_RANGE)),
                      float(rng.uniform(*POINT_Z_RANGE))],
        )
        for _ in range(n_points)
    ]
    sun_angle = [float(rng.uniform(*SUN_ELEVATION_RANGE)),
                 float(rng.uniform(*SUN_AZIMUTH_RANGE))]
    sun_energy = float(rng.uniform(*SUN_ENERGY_RANGE))

    masks_enabled = bool(lopt.get("occluders", True))
    # Draw every non-sun source's activation before any pattern seeds so one enabled
    # mask cannot alter another source's activation decision.
    spotlight_has_mask = (masks_enabled and spotlight
                           and _maybe(rng, NON_SUN_SHADOW_MASK_PROB))
    point_has_masks = [masks_enabled and _maybe(rng, NON_SUN_SHADOW_MASK_PROB)
                       for _point in points]

    def mask_seed(enabled: bool) -> Optional[int]:
        if not enabled:
            return None
        return int(rng.integers(0, SHADOW_MASK_SEED_MAX))

    for point, enabled in zip(points, point_has_masks):
        point.shadow_mask_seed = mask_seed(enabled)
    return LightingConfig(
        sun_angle_deg=sun_angle,
        sun_energy=sun_energy,
        spotlight_beside_camera=spotlight,
        point_lights=points,
        spotlight_shadow_mask_seed=mask_seed(spotlight_has_mask),
        shadow_plane_opacity=float(tuning["shadow_plane_opacity"]),
    )


def _sample_camera(rng, opts, tuning: Dict[str, Any], max_offaxis_deg: float) -> CameraConfig:
    return CameraConfig(
        focal_mm=float(rng.uniform(*CAMERA_FOCAL_RANGE)),
        offaxis_deg=float(rng.uniform(CAMERA_OFFAXIS_RANGE[0], max_offaxis_deg)),
        orbit_deg=float(rng.uniform(*CAMERA_ORBIT_RANGE)),
        dof_enabled=True,
        aperture_fstop=float(rng.uniform(*tuning["aperture_fstop_range"])),
    )


def _sample_postfx(rng, opts, tuning: Dict[str, Any]) -> PostFxConfig:
    enabled = set(opts.get("post_effects", []))
    def active(name: str) -> bool:
        return name in enabled and _maybe(rng, float(tuning[name]["probability"]))

    def uniform(name: str, key: str) -> float:
        return float(rng.uniform(*tuning[name][key]))

    def integer(name: str, key: str) -> int:
        low, high = tuning[name][key]
        return int(rng.integers(int(low), int(high) + 1))

    motion_blur = None
    if active("motion_blur"):
        motion_blur = MotionBlurConfig(
            direction_index=int(rng.integers(0, 16)),
            strength=uniform("motion_blur", "strength_range"),
        )
    noise = None
    if active("sensor_noise"):
        noise = SensorNoiseConfig(
            luma_sigma=uniform("sensor_noise", "luma_sigma_range"),
            chroma_sigma=uniform("sensor_noise", "chroma_sigma_range"),
            seed=int(rng.integers(0, SHADOW_MASK_SEED_MAX)),
        )
    compression = None
    if active("compression"):
        compression = CompressionConfig(
            jpeg_quality=integer("compression", "jpeg_quality_range"),
            cycles=integer("compression", "cycles_range"),
        )
    pixel_melt = None
    if active("pixel_melt"):
        pixel_melt = PixelMeltConfig(
            blur_sigma=uniform("pixel_melt", "blur_sigma_range"),
            scale=uniform("pixel_melt", "scale_range"),
        )
    white_balance = None
    if active("white_balance"):
        white_balance = WhiteBalanceConfig(
            temperature_shift_k=uniform("white_balance", "temperature_shift_k_range"))
    tint = None
    if active("tint"):
        tint = TintConfig(green_magenta_shift=uniform("tint", "green_magenta_shift_range"))
    chromatic = None
    if active("chromatic_aberration"):
        chromatic = ChromaticAberrationConfig(
            shift_px=uniform("chromatic_aberration", "shift_px_range"),
            angle_deg=float(rng.uniform(0.0, 360.0)),
        )
    contrast = None
    if active("contrast"):
        contrast = ContrastConfig(reduction=uniform("contrast", "reduction_range"))
    haze = None
    if active("haze"):
        haze = HazeConfig(
            strength=uniform("haze", "strength_range"),
            brightness=uniform("haze", "brightness_range"),
        )
    return PostFxConfig(
        motion_blur=motion_blur,
        sensor_noise=noise,
        compression=compression,
        pixel_melt=pixel_melt,
        white_balance=white_balance,
        tint=tint,
        chromatic_aberration=chromatic,
        contrast=contrast,
        haze=haze,
    )


# --------------------------------------------------------------------------- #
# Entry point + validation
# --------------------------------------------------------------------------- #
def sample_scene_config(enabled_options: Optional[Dict[str, Any]], rng_seed: int,
                        max_cards: Optional[int] = None, config_path: Optional[str] = None,
                        rng: Optional[np.random.Generator] = None) -> SceneConfig:
    """Sample ONE validated scene config from a seed. Deterministic per seed.
    `max_cards` (if given) caps the number of cards in the scene. `config_path`
    selects the runtime config.json used for post-effect tuning. Pass ``rng`` to
    continue the same scene generator into downstream Blender construction."""
    if max_cards is not None and int(max_cards) < 1:
        raise ConfigError("max_cards must be at least 1.")
    opts = _resolve(enabled_options)
    rng = np.random.default_rng(rng_seed) if rng is None else rng
    layout, cards = _sample_layout_and_cards(rng, opts)
    if max_cards is not None and len(cards) > max_cards:
        cards = cards[:int(max_cards)]
        if layout.kind == "binder":
            layout.params["filled_slots"] = layout.params["filled_slots"][:len(cards)]
        elif layout.kind == "display_case":
            cols = int(layout.params["cols"])
            layout.params["rows"] = (len(cards) + cols - 1) // cols
    # Apply this after external truncation so the retained prefix always contains a
    # potential camera/DoF focus target.
    if all(card.back_to_camera for card in cards):
        cards[0].back_to_camera = False
    runtime_tuning = project_config.load_config(config_path)
    max_offaxis_deg = float(
        runtime_tuning["layouts"][layout.kind]["camera_max_offaxis_deg"])
    cfg = SceneConfig(
        seed=int(rng_seed),
        layout=layout,
        cards=cards,
        lighting=_sample_lighting(rng, opts, runtime_tuning["lighting"]),
        camera=_sample_camera(rng, opts, runtime_tuning["camera"], max_offaxis_deg),
        postfx=_sample_postfx(rng, opts, runtime_tuning["postfx"]),
    )
    validate_scene_config(cfg)  # never emit an illegal config
    return cfg


def validate_scene_config(cfg: SceneConfig) -> None:
    """Assert every combination rule holds. Raises AssertionError on violation."""
    assert cfg.layout.kind in LAYOUTS, cfg.layout.kind
    assert len(cfg.cards) >= 1, "scene must have >=1 card"

    for c in cfg.cards:
        p = c.protection
        assert p.kind in PROTECTIONS, p.kind
        # sleeve presence implication rules (spec §3.2)
        if p.kind in ("toploader", "semi_rigid"):
            assert p.sleeve is not None, f"{p.kind} must contain a sleeve"
        if p.kind == "sleeve":
            assert p.sleeve is not None, "sleeve protection must have a sleeve"
        if p.kind in ("slab", "none"):
            assert p.sleeve is None, f"{p.kind} must NOT have a sleeve"
        # sleeve type/size mutual exclusivity (exactly one each)
        if p.sleeve is not None:
            assert p.sleeve.sleeve_type in SLEEVE_TYPES, p.sleeve.sleeve_type
            assert p.sleeve.size in SLEEVE_SIZES, p.sleeve.size
        # holder inner placement only for semi_rigid/toploader
        if p.kind in ("semi_rigid", "toploader"):
            assert p.inner_offset_mm is not None and p.inner_rot_deg is not None
            assert abs(p.inner_rot_deg) <= HOLDER_MAX_ROT_DEG + 1e-9
            assert all(abs(v) <= HOLDER_MAX_OFFSET_MM + 1e-9 for v in p.inner_offset_mm)

        # finish rules (spec §3.4)
        f = c.finish
        assert f.kind in FINISHES, f.kind
        if f.kind == "holo":
            assert f.holo_region in HOLO_REGIONS, f.holo_region
            assert f.holo_pattern in HOLO_PATTERNS, f.holo_pattern
            if f.physical_texture:
                assert f.holo_pattern == "none", \
                    "physical texture must pair with the 'none' holo pattern"
        else:
            assert f.holo_region is None and f.holo_pattern is None
            assert f.physical_texture is False

    assert any(not card.back_to_camera for card in cfg.cards), \
        "scene needs at least one front-facing focus candidate"

    # layout-specific protection constraints (spec §3.5)
    lk = cfg.layout.kind
    kinds = {c.protection.kind for c in cfg.cards}
    if lk == "display_case":
        assert kinds <= {"toploader", "slab"}, f"display_case has {kinds}"
        assert len(cfg.cards) <= DISPLAY_CASE_MAX_CARDS, \
            f"display_case has {len(cfg.cards)} cards (max {DISPLAY_CASE_MAX_CARDS})"
    elif lk == "hand":
        assert kinds <= {"none", "sleeve", "toploader"}, f"hand has {kinds}"
        assert len(cfg.cards) == 1, "hand layout requires exactly one card"
        assert not cfg.cards[0].back_to_camera, "hand card must face the camera"
        grip = cfg.layout.params.get("grip")
        assert grip in HAND_GRIPS
        assert cfg.layout.params.get("handedness") in HAND_SIDES
        if cfg.cards[0].protection.kind == "none":
            assert grip == "pinch", "bare hand-held cards require a pinch grip"
        if grip == "side":
            assert cfg.cards[0].protection.kind in {"sleeve", "toploader"}
        approach = float(cfg.layout.params.get("approach_deg", -1.0))
        depth = float(cfg.layout.params.get("depth", -1.0))
        assert 0.0 <= approach < 360.0
        assert HAND_DEPTH_RANGE[0] <= depth <= HAND_DEPTH_RANGE[1]
    elif lk == "binder":
        content = cfg.layout.params["content_type"]
        expected = _BINDER_CONTENT_PROTECTION[content]
        assert kinds == {expected}, f"binder({content}) expects {{{expected}}}, got {kinds}"

    # Lighting ranges and hard rules (spec §3.6).
    lit = cfg.lighting
    assert lit.spotlight_beside_camera or len(lit.point_lights) >= 1, \
        "at least one non-sun light must exist"
    assert math.isfinite(lit.shadow_plane_opacity)
    assert 0.0 <= lit.shadow_plane_opacity <= 1.0
    assert len(lit.sun_angle_deg) == 2
    elevation, azimuth = (float(value) for value in lit.sun_angle_deg)
    assert all(math.isfinite(value) for value in (elevation, azimuth, lit.sun_energy))
    assert SUN_ELEVATION_RANGE[0] <= elevation <= SUN_ELEVATION_RANGE[1]
    assert SUN_AZIMUTH_RANGE[0] <= azimuth <= SUN_AZIMUTH_RANGE[1]
    assert SUN_ENERGY_RANGE[0] <= lit.sun_energy <= SUN_ENERGY_RANGE[1]
    assert 0 <= len(lit.point_lights) <= 4
    assert lit.spotlight_beside_camera or lit.spotlight_shadow_mask_seed is None, \
        "a spotlight shadow mask requires a spotlight"

    def validate_mask_seed(seed) -> None:
        assert seed is None or (
            isinstance(seed, int) and not isinstance(seed, bool)
            and 0 <= seed < SHADOW_MASK_SEED_MAX
        ), f"invalid shadow mask seed: {seed!r}"

    validate_mask_seed(lit.spotlight_shadow_mask_seed)
    point_energy_range = _point_energy_range(len(lit.point_lights))
    for point in lit.point_lights:
        assert len(point.position) == 3
        values = (point.color_temp, point.intensity, *point.position)
        assert all(math.isfinite(float(value)) for value in values)
        assert POINT_TEMP_RANGE[0] <= point.color_temp <= POINT_TEMP_RANGE[1]
        assert point_energy_range[0] <= point.intensity <= point_energy_range[1]
        assert all(POINT_XY_RANGE[0] <= value <= POINT_XY_RANGE[1]
                   for value in point.position[:2])
        assert POINT_Z_RANGE[0] <= point.position[2] <= POINT_Z_RANGE[1]
        validate_mask_seed(point.shadow_mask_seed)

    # Camera ranges and required DoF (spec §3.7).
    cam = cfg.camera
    assert all(math.isfinite(value) for value in
               (cam.focal_mm, cam.offaxis_deg, cam.orbit_deg, cam.aperture_fstop))
    assert CAMERA_FOCAL_RANGE[0] <= cam.focal_mm <= CAMERA_FOCAL_RANGE[1]
    assert CAMERA_OFFAXIS_ALLOWED_RANGE[0] <= cam.offaxis_deg <= CAMERA_OFFAXIS_ALLOWED_RANGE[1]
    assert CAMERA_ORBIT_RANGE[0] <= cam.orbit_deg < CAMERA_ORBIT_RANGE[1]
    assert cam.dof_enabled is True
    assert CAMERA_FSTOP_ALLOWED_RANGE[0] <= cam.aperture_fstop <= CAMERA_FSTOP_ALLOWED_RANGE[1]

    # Post effects are image-space only; all randomness was sampled into SceneConfig.
    fx = cfg.postfx
    def finite(*values):
        return all(math.isfinite(float(value)) for value in values)

    if fx.motion_blur is not None:
        assert isinstance(fx.motion_blur.direction_index, int) \
            and not isinstance(fx.motion_blur.direction_index, bool)
        assert 0 <= fx.motion_blur.direction_index < 16
        assert finite(fx.motion_blur.strength)
        assert 0.0 <= fx.motion_blur.strength <= 1.0
    if fx.sensor_noise is not None:
        assert finite(fx.sensor_noise.luma_sigma, fx.sensor_noise.chroma_sigma)
        assert fx.sensor_noise.luma_sigma >= 0.0 and fx.sensor_noise.chroma_sigma >= 0.0
        validate_mask_seed(fx.sensor_noise.seed)
    if fx.compression is not None:
        assert 1 <= fx.compression.jpeg_quality <= 100 and fx.compression.cycles >= 1
    if fx.pixel_melt is not None:
        assert finite(fx.pixel_melt.blur_sigma, fx.pixel_melt.scale)
        assert fx.pixel_melt.blur_sigma >= 0.0 and 0.0 < fx.pixel_melt.scale <= 1.0
    if fx.white_balance is not None:
        assert finite(fx.white_balance.temperature_shift_k)
    if fx.tint is not None:
        assert finite(fx.tint.green_magenta_shift)
    if fx.chromatic_aberration is not None:
        assert finite(fx.chromatic_aberration.shift_px, fx.chromatic_aberration.angle_deg)
        assert fx.chromatic_aberration.shift_px >= 0.0
    if fx.contrast is not None:
        assert 0.0 <= fx.contrast.reduction <= 0.5
    if fx.haze is not None:
        assert finite(fx.haze.strength, fx.haze.brightness)
        assert 0.0 <= fx.haze.strength <= 1.0 and 0.0 <= fx.haze.brightness <= 1.0
