"""Docker unit tests for the unified config.json loader (per-layout params)."""
from __future__ import annotations

import json
import os
import sys
import tempfile

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402


def test_project_config_json_loads():
    cfg = config.load_config()
    assert set(cfg) == set(config.DEFAULT_CONFIG)
    assert set(cfg["holo"]) == set(config.DEFAULT_CONFIG["holo"])
    # per-layout sections
    assert "table" in cfg["layouts"] and "floating" in cfg["layouts"]
    assert config.load_layout_params("hand")["max_cards"] == 1
    assert "max_shapes" in config.load_layout_params("floating")
    assert config.load_holo_tuning()["angle_gain"] == cfg["holo"]["angle_gain"]
    assert config.load_layout_params("table")["max_cards"] == cfg["layouts"]["table"]["max_cards"]
    assert config.load_layout_params("nonexistent") == {}


def test_hand_asset_path_default_and_env_override():
    old = os.environ.get("TCG_HAND_ASSET")
    try:
        os.environ.pop("TCG_HAND_ASSET", None)
        expected = os.path.join(_ROOT, "assets", config.HAND_ASSET_FILENAME)
        assert os.path.normcase(config.hand_asset_path()) == os.path.normcase(expected)
        assert os.path.isabs(config.hand_asset_path())
        override = os.path.join("custom", "hand.blend")
        os.environ["TCG_HAND_ASSET"] = override
        assert config.hand_asset_path() == override
    finally:
        if old is None:
            os.environ.pop("TCG_HAND_ASSET", None)
        else:
            os.environ["TCG_HAND_ASSET"] = old


def test_type_coercion_and_partial_merge():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "config.json")
        with open(p, "w") as fh:
            json.dump({"holo": {"sharpness": "9"},
                       "layouts": {"floating": {"max_cards": "3", "allow_overlap": 1}}}, fh)
        cfg = config.load_config(p)
        assert cfg["holo"]["sharpness"] == 9.0 and isinstance(cfg["holo"]["sharpness"], float)
        fl = cfg["layouts"]["floating"]
        assert fl["max_cards"] == 3 and isinstance(fl["max_cards"], int)
        assert fl["allow_overlap"] is True
        # unspecified keep defaults (both within and across layouts)
        assert fl["max_shapes"] == config.DEFAULT_CONFIG["layouts"]["floating"]["max_shapes"]
        assert cfg["layouts"]["table"] == config.DEFAULT_CONFIG["layouts"]["table"]
        assert cfg["holo"]["emit"] == config.DEFAULT_CONFIG["holo"]["emit"]


def test_out_of_frustum_enum_validated_per_layout():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "config.json")
        with open(p, "w") as fh:
            json.dump({"layouts": {"table": {"out_of_frustum": "bogus"},
                                   "floating": {"out_of_frustum": "remove"}}}, fh)
        cfg = config.load_config(p)
        assert cfg["layouts"]["table"]["out_of_frustum"] == "keep"      # bad -> default
        assert cfg["layouts"]["floating"]["out_of_frustum"] == "remove"  # valid kept


def test_missing_and_malformed_fall_back():
    assert config.load_config("/no/such/file.json") == config.DEFAULT_CONFIG
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.json")
        with open(p, "w") as fh:
            fh.write("{ not valid json ]")
        assert config.load_config(p) == config.DEFAULT_CONFIG


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
