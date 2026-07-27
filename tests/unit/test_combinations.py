"""
Docker unit tests for rules/combinations.py — every exclusion/implication rule
(spec §3.2/§3.4/§3.5/§3.6) plus the §7 definition-of-done 500-sample audit.

Run:  python3 tests/unit/test_combinations.py
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rules import combinations as C  # noqa: E402


def _sample(seed, **opts):
    return C.sample_scene_config(opts or None, seed)


# --------------------------------------------------------------------------- #
# §7 audit: 500 configs, all options enabled, every rule must hold + coverage
# --------------------------------------------------------------------------- #
def test_audit_500_samples_all_rules_hold_and_coverage():
    seen_layouts, seen_prot, seen_finish = set(), set(), set()
    for seed in range(500):
        cfg = C.sample_scene_config(None, seed)
        C.validate_scene_config(cfg)  # explicit re-check (sample already validates)
        seen_layouts.add(cfg.layout.kind)
        for c in cfg.cards:
            seen_prot.add(c.protection.kind)
            seen_finish.add(c.finish.kind)
    # Coverage: the audit should exercise every layout, protection, and finish.
    assert seen_layouts == set(C.LAYOUTS), f"missing layouts: {set(C.LAYOUTS) - seen_layouts}"
    assert seen_prot == set(C.PROTECTIONS), f"missing protections: {set(C.PROTECTIONS) - seen_prot}"
    assert seen_finish == set(C.FINISHES), f"missing finishes: {set(C.FINISHES) - seen_finish}"


def test_determinism_same_seed_same_config():
    a = C.sample_scene_config(None, 42)
    b = C.sample_scene_config(None, 42)
    assert a.to_json() == b.to_json()
    c = C.sample_scene_config(None, 43)
    assert c.to_json() != a.to_json()  # different seed => (almost surely) different


def test_json_serializable():
    cfg = C.sample_scene_config(None, 7)
    s = json.dumps(cfg.to_dict())  # must not raise
    assert '"seed": 7' in s or '"seed":7' in s


# --------------------------------------------------------------------------- #
# Protection implication rules (force each via single-protection option sets)
# --------------------------------------------------------------------------- #
def test_toploader_always_sleeved():
    for seed in range(60):
        cfg = _sample(seed, protections=["toploader"], layouts=["table"])
        for c in cfg.cards:
            assert c.protection.kind == "toploader"
            assert c.protection.sleeve is not None


def test_semi_rigid_always_contains_sleeve():
    for seed in range(60):
        cfg = _sample(seed, protections=["semi_rigid"], layouts=["table"])
        for c in cfg.cards:
            assert c.protection.kind == "semi_rigid"
            assert c.protection.sleeve is not None


def test_slab_never_sleeved():
    for seed in range(60):
        cfg = _sample(seed, protections=["slab"], layouts=["table"])
        for c in cfg.cards:
            assert c.protection.kind == "slab"
            assert c.protection.sleeve is None


def test_none_never_sleeved():
    for seed in range(60):
        cfg = _sample(seed, protections=["none"], layouts=["table"])
        for c in cfg.cards:
            assert c.protection.sleeve is None


def test_holders_mutually_exclusive_by_construction():
    # A single card has exactly one protection.kind, so semi_rigid/toploader/slab
    # can never co-occur on one card. Confirm across the audit.
    for seed in range(200):
        cfg = C.sample_scene_config(None, seed)
        for c in cfg.cards:
            assert c.protection.kind in C.PROTECTIONS


def test_inner_offset_and_rotation_bounds():
    for seed in range(80):
        for prot in ("semi_rigid", "toploader"):
            cfg = _sample(seed, protections=[prot], layouts=["table"])
            for c in cfg.cards:
                assert abs(c.protection.inner_rot_deg) <= C.HOLDER_MAX_ROT_DEG + 1e-9
                assert all(abs(v) <= C.HOLDER_MAX_OFFSET_MM + 1e-9
                           for v in c.protection.inner_offset_mm)


# --------------------------------------------------------------------------- #
# Sleeve type/size exclusivity
# --------------------------------------------------------------------------- #
def test_sleeve_size_restriction_respected():
    for seed in range(60):
        cfg = _sample(seed, protections=["sleeve"], sleeve_sizes=["2.5mm"], layouts=["table"])
        for c in cfg.cards:
            assert c.protection.sleeve.size == "2.5mm"


def test_sleeve_type_restriction_respected():
    for seed in range(60):
        cfg = _sample(seed, protections=["sleeve"], sleeve_types=["opaque_back"], layouts=["table"])
        for c in cfg.cards:
            assert c.protection.sleeve.sleeve_type == "opaque_back"


# --------------------------------------------------------------------------- #
# Finish rules
# --------------------------------------------------------------------------- #
def test_holo_only_has_region_and_pattern():
    for seed in range(80):
        cfg = _sample(seed, finishes=["holo"], layouts=["table"])
        for c in cfg.cards:
            assert c.finish.kind == "holo"
            assert c.finish.holo_region in C.HOLO_REGIONS
            assert c.finish.holo_pattern in C.HOLO_PATTERNS


def test_normal_finish_has_no_holo_fields():
    for seed in range(80):
        cfg = _sample(seed, finishes=["normal"], layouts=["table"])
        for c in cfg.cards:
            assert c.finish.kind == "normal"
            assert c.finish.holo_region is None and c.finish.holo_pattern is None
            assert c.finish.physical_texture is False


def test_physical_texture_pairs_with_none_pattern():
    seen_physical = False
    for seed in range(300):
        cfg = _sample(seed, finishes=["holo"], layouts=["table"])
        for c in cfg.cards:
            if c.finish.physical_texture:
                seen_physical = True
                assert c.finish.holo_pattern == "none", c.finish
    assert seen_physical, "expected some physical-texture cards in the sample"


def test_holo_pattern_restriction():
    for seed in range(60):
        cfg = _sample(seed, finishes=["holo"], holo_patterns=["cosmos"], layouts=["table"])
        for c in cfg.cards:
            assert c.finish.holo_pattern == "cosmos"


# --------------------------------------------------------------------------- #
# Layout-specific constraints
# --------------------------------------------------------------------------- #
def test_display_case_only_toploader_or_slab():
    for seed in range(100):
        cfg = _sample(seed, layouts=["display_case"])
        kinds = {c.protection.kind for c in cfg.cards}
        assert kinds <= {"toploader", "slab"}, kinds


def test_display_case_capped_at_24_cards():
    seen_max = 0
    for seed in range(300):
        cfg = _sample(seed, layouts=["display_case"])
        n = len(cfg.cards)
        assert n <= C.DISPLAY_CASE_MAX_CARDS, f"seed {seed}: {n} cards"
        # grid params stay consistent with the (possibly capped) card count.
        cols = cfg.layout.params["cols"]
        rows = cfg.layout.params["rows"]
        assert n <= cols * rows and n > cols * (rows - 1), (n, cols, rows)
        seen_max = max(seen_max, n)
    assert seen_max == C.DISPLAY_CASE_MAX_CARDS, f"cap never reached (max {seen_max})"


def test_display_case_external_cap_keeps_grid_metadata_consistent():
    for seed in range(30):
        cfg = C.sample_scene_config({"layouts": ["display_case"]}, seed, max_cards=3)
        cols = cfg.layout.params["cols"]
        rows = cfg.layout.params["rows"]
        assert len(cfg.cards) == 3
        assert rows == (len(cfg.cards) + cols - 1) // cols


def test_binder_cards_match_content_type():
    for seed in range(150):
        cfg = _sample(seed, layouts=["binder"])
        content = cfg.layout.params["content_type"]
        expected = C._BINDER_CONTENT_PROTECTION[content]
        kinds = {c.protection.kind for c in cfg.cards}
        assert kinds == {expected}, (content, kinds)
        # not every slot filled
        assert len(cfg.cards) <= cfg.layout.params["capacity"]


def test_hand_grip_and_protection():
    for seed in range(120):
        cfg = _sample(seed, layouts=["hand"])
        assert len(cfg.cards) == 1
        assert cfg.layout.params["grip"] in C.HAND_GRIPS
        assert cfg.layout.params["handedness"] in C.HAND_SIDES
        assert 0.0 <= cfg.layout.params["approach_deg"] < 360.0
        assert C.HAND_DEPTH_RANGE[0] <= cfg.layout.params["depth"] <= C.HAND_DEPTH_RANGE[1]
        assert cfg.cards[0].back_to_camera is False
        kind = cfg.cards[0].protection.kind
        assert kind in ("none", "sleeve", "toploader")
        if kind == "none":
            assert cfg.layout.params["grip"] == "pinch"


def test_hand_validation_rejects_illegal_manual_config():
    cfg = _sample(4, layouts=["hand"], protections=["sleeve"])
    cfg.layout.params["depth"] = 0.9
    try:
        C.validate_scene_config(cfg)
    except AssertionError:
        pass
    else:
        raise AssertionError("invalid hand depth must be rejected")


def test_unsatisfiable_layout_raises():
    # display_case needs toploader/slab; enabling only sleeve => impossible.
    try:
        _sample(1, layouts=["display_case"], protections=["sleeve"])
    except C.ConfigError:
        pass
    else:
        raise AssertionError("expected ConfigError for unsatisfiable display_case")


def test_top_card_does_not_reenable_disabled_protections():
    import numpy as np
    try:
        C.sample_top_card(np.random.default_rng(1), {"protections": ["semi_rigid"]})
    except C.ConfigError:
        pass
    else:
        raise AssertionError("top-card sampling must honor disabled protections")


def test_max_cards_must_be_positive():
    try:
        C.sample_scene_config(None, 1, max_cards=0)
    except C.ConfigError:
        pass
    else:
        raise AssertionError("max_cards=0 must be rejected")


# --------------------------------------------------------------------------- #
# Lighting and camera rules
# --------------------------------------------------------------------------- #
def test_at_least_one_non_sun_light():
    for seed in range(200):
        cfg = C.sample_scene_config(None, seed)
        assert cfg.lighting.spotlight_beside_camera or len(cfg.lighting.point_lights) >= 1


def test_non_sun_light_forced_even_if_disabled():
    # Even if the user disables spotlight AND point lights, the hard §3.6 rule holds.
    for seed in range(40):
        cfg = _sample(seed, lighting={"spotlight": False, "point_lights": False})
        assert cfg.lighting.spotlight_beside_camera or len(cfg.lighting.point_lights) >= 1


def test_lighting_camera_ranges_and_focus_candidate():
    for seed in range(200):
        cfg = C.sample_scene_config(None, seed)
        lighting = cfg.lighting
        camera = cfg.camera
        assert C.SUN_ELEVATION_RANGE[0] <= lighting.sun_angle_deg[0] <= C.SUN_ELEVATION_RANGE[1]
        assert C.SUN_AZIMUTH_RANGE[0] <= lighting.sun_angle_deg[1] <= C.SUN_AZIMUTH_RANGE[1]
        assert C.SUN_ENERGY_RANGE[0] <= lighting.sun_energy <= C.SUN_ENERGY_RANGE[1]
        assert len(lighting.point_lights) <= 4
        point_energy_range = C._point_energy_range(len(lighting.point_lights))
        for point in lighting.point_lights:
            assert C.POINT_TEMP_RANGE[0] <= point.color_temp <= C.POINT_TEMP_RANGE[1]
            assert point_energy_range[0] <= point.intensity <= point_energy_range[1]
            assert all(C.POINT_XY_RANGE[0] <= value <= C.POINT_XY_RANGE[1]
                       for value in point.position[:2])
            assert C.POINT_Z_RANGE[0] <= point.position[2] <= C.POINT_Z_RANGE[1]
        assert C.CAMERA_FOCAL_RANGE[0] <= camera.focal_mm <= C.CAMERA_FOCAL_RANGE[1]
        assert C.CAMERA_OFFAXIS_RANGE[0] <= camera.offaxis_deg <= C.CAMERA_OFFAXIS_RANGE[1]
        assert C.CAMERA_ORBIT_RANGE[0] <= camera.orbit_deg < C.CAMERA_ORBIT_RANGE[1]
        assert camera.dof_enabled is True
        assert C.CAMERA_FSTOP_RANGE[0] <= camera.aperture_fstop <= C.CAMERA_FSTOP_RANGE[1]
        assert any(not card.back_to_camera for card in cfg.cards)


def test_point_energy_max_reduces_for_two_and_three_lights():
    assert C._point_energy_range(1) == (1.125, 22.5)
    assert C._point_energy_range(2) == (1.125, 14.5)
    assert C._point_energy_range(3) == (1.125, 7.5)
    assert C._point_energy_range(4) == (1.125, 22.5)

    for count in (2, 3):
        cfg = None
        for seed in range(100):
            candidate = C.sample_scene_config(None, seed)
            if len(candidate.lighting.point_lights) == count:
                cfg = candidate
                break
        assert cfg is not None
        cfg.lighting.point_lights[0].intensity = C._point_energy_range(count)[1] + 0.01
        _expect_invalid(cfg)


def _shadow_mask_count(lighting):
    return int(lighting.spotlight_shadow_mask_seed is not None) + \
        sum(point.shadow_mask_seed is not None for point in lighting.point_lights)


def test_shadow_masks_are_deterministic_and_master_disable_works():
    for seed in range(100):
        first = C.sample_scene_config(None, seed).lighting
        second = C.sample_scene_config(None, seed).lighting
        assert first == second
        disabled = _sample(seed, lighting={"occluders": False}).lighting
        assert _shadow_mask_count(disabled) == 0


def test_shadow_mask_seeds_fit_blender_custom_property_c_ints():
    assert C.SHADOW_MASK_SEED_MAX == 2 ** 31
    for seed in range(500):
        lighting = C.sample_scene_config(None, seed).lighting
        seeds = [lighting.spotlight_shadow_mask_seed]
        seeds.extend(point.shadow_mask_seed for point in lighting.point_lights)
        for mask_seed in seeds:
            if mask_seed is not None:
                assert 0 <= mask_seed < C.SHADOW_MASK_SEED_MAX


def test_shadow_masks_sample_independently_per_existing_non_sun_light():
    spot_eligible = spot_draws = 0
    point_eligible = point_draws = 0
    saw_multiple = False
    for seed in range(5000):
        lighting = C.sample_scene_config(None, seed).lighting
        if lighting.spotlight_beside_camera:
            spot_eligible += 1
            spot_draws += lighting.spotlight_shadow_mask_seed is not None
        else:
            assert lighting.spotlight_shadow_mask_seed is None
        point_eligible += len(lighting.point_lights)
        point_draws += sum(point.shadow_mask_seed is not None
                           for point in lighting.point_lights)
        saw_multiple |= _shadow_mask_count(lighting) >= 2

    assert 0.22 <= spot_draws / spot_eligible <= 0.28
    assert 0.22 <= point_draws / point_eligible <= 0.28
    assert saw_multiple, "sources must not be reduced to one selected shadow mask"


def test_loose_layout_always_has_front_facing_card():
    for layout in ("table", "floating"):
        for seed in range(50):
            cfg = _sample(seed, layouts=[layout], back_to_camera_prob=1.0)
            assert any(not card.back_to_camera for card in cfg.cards)


def test_back_facing_cards_remain_legal_with_focus_candidate():
    saw_mixed_scene = False
    for seed in range(200):
        cfg = _sample(seed, layouts=["table"], back_to_camera_prob=0.5)
        backs = [card.back_to_camera for card in cfg.cards]
        if any(backs) and not all(backs):
            saw_mixed_scene = True
            C.validate_scene_config(cfg)
    assert saw_mixed_scene, "sampling should retain legal front/back card mixtures"


def test_max_cards_retains_front_facing_focus_candidate():
    for layout in ("table", "floating"):
        for seed in range(100):
            cfg = C.sample_scene_config(
                {"layouts": [layout], "back_to_camera_prob": 0.75}, seed, max_cards=1)
            assert len(cfg.cards) == 1
            assert cfg.cards[0].back_to_camera is False


def _expect_invalid(cfg):
    try:
        C.validate_scene_config(cfg)
    except AssertionError:
        return
    raise AssertionError("invalid scene config must be rejected")


def test_validation_rejects_invalid_lighting():
    cfg = _sample(7, layouts=["table"],
                  lighting={"spotlight": False, "point_lights": True})
    cfg.lighting.sun_energy = C.SUN_ENERGY_RANGE[1] + 1.0
    _expect_invalid(cfg)

    cfg = _sample(7, layouts=["table"],
                  lighting={"spotlight": False, "point_lights": True})
    cfg.lighting.point_lights[0].position[2] = -0.1
    _expect_invalid(cfg)

    cfg = _sample(7, layouts=["table"], lighting={"spotlight": False})
    cfg.lighting.spotlight_beside_camera = False
    cfg.lighting.spotlight_shadow_mask_seed = 12
    _expect_invalid(cfg)

    cfg = _sample(7, layouts=["table"])
    cfg.lighting.point_lights[0].shadow_mask_seed = True
    _expect_invalid(cfg)


def test_validation_rejects_invalid_camera_or_focus_candidate():
    cfg = _sample(9, layouts=["table"])
    cfg.camera.dof_enabled = False
    _expect_invalid(cfg)

    cfg = _sample(9, layouts=["table"])
    cfg.camera.orbit_deg = 360.0
    _expect_invalid(cfg)

    cfg = _sample(9, layouts=["table"])
    for card in cfg.cards:
        card.back_to_camera = True
    _expect_invalid(cfg)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
