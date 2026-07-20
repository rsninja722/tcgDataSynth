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
        kind = cfg.cards[0].protection.kind
        assert kind in ("none", "sleeve", "toploader")
        if kind == "none":
            assert cfg.layout.params["grip"] == "pinch"


def test_unsatisfiable_layout_raises():
    # display_case needs toploader/slab; enabling only sleeve => impossible.
    try:
        _sample(1, layouts=["display_case"], protections=["sleeve"])
    except C.ConfigError:
        pass
    else:
        raise AssertionError("expected ConfigError for unsatisfiable display_case")


# --------------------------------------------------------------------------- #
# Lighting rule
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


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
