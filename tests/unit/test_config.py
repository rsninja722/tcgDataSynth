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
    assert set(cfg["postfx"]) == set(config.DEFAULT_CONFIG["postfx"])
    assert cfg["camera"]["aperture_fstop_range"] == [8.0, 20.0]
    assert cfg["lighting"]["shadow_plane_opacity"] == 0.9
    # per-layout sections
    assert "table" in cfg["layouts"] and "floating" in cfg["layouts"]
    assert config.load_layout_params("hand")["max_cards"] == 1
    assert config.load_layout_params("display_case")["camera_max_offaxis_deg"] == 30.0
    assert config.load_layout_params("stack")["max_cards"] == 10
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


def test_layout_camera_offaxis_maximum_is_validated():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "config.json")
        with open(p, "w") as fh:
            json.dump({"layouts": {
                "table": {"camera_max_offaxis_deg": 12.5},
                "display_case": {"camera_max_offaxis_deg": 90.0},
            }}, fh)
        cfg = config.load_config(p)
        assert cfg["layouts"]["table"]["camera_max_offaxis_deg"] == 12.5
        assert cfg["layouts"]["display_case"]["camera_max_offaxis_deg"] == 30.0

        with open(p, "w") as fh:
            json.dump({"layouts": {"stack": {"max_cards": 0}}}, fh)
        assert config.load_layout_params("stack", p)["max_cards"] == 10


def test_missing_and_malformed_fall_back():
    assert config.load_config("/no/such/file.json") == config.DEFAULT_CONFIG
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.json")
        with open(p, "w") as fh:
            fh.write("{ not valid json ]")
        assert config.load_config(p) == config.DEFAULT_CONFIG


def test_postfx_probabilities_and_ranges_are_normalized():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "config.json")
        with open(p, "w") as fh:
            json.dump({"postfx": {
                "sensor_noise": {"probability": 2.0, "luma_sigma_range": [0.02, 0.01]},
                "compression": {"jpeg_quality_range": [90, 20], "cycles_range": [0, 0]},
                "contrast": {"reduction_range": [-0.1, 0.7]},
            }}, fh)
        fx = config.load_postfx_tuning(p)
        assert fx["sensor_noise"]["probability"] == 1.0
        assert fx["sensor_noise"]["luma_sigma_range"] == [0.01, 0.02]
        assert fx["compression"]["jpeg_quality_range"] == [20, 90]
        assert fx["compression"]["cycles_range"] == [1, 1]
        assert fx["contrast"]["reduction_range"] \
            == config.DEFAULT_CONFIG["postfx"]["contrast"]["reduction_range"]


def test_camera_and_lighting_tuning_are_validated():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "config.json")
        with open(p, "w") as fh:
            json.dump({"camera": {"aperture_fstop_range": [11.0, 2.4]},
                       "lighting": {"shadow_plane_opacity": 1.5}}, fh)
        cfg = config.load_config(p)
        assert cfg["camera"]["aperture_fstop_range"] == [2.4, 11.0]
        assert cfg["lighting"]["shadow_plane_opacity"] == 1.0

        with open(p, "w") as fh:
            json.dump({"camera": {"aperture_fstop_range": [-1.0, 100.0]},
                       "lighting": {"shadow_plane_opacity": "bad"}}, fh)
        cfg = config.load_config(p)
        assert cfg["camera"] == config.DEFAULT_CONFIG["camera"]
        assert cfg["lighting"] == config.DEFAULT_CONFIG["lighting"]


def test_generation_settings_are_validated_and_persisted():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "config.json")
        with open(p, "w") as fh:
            json.dump({"keep": "this", "generation": {
                "count": 3,
                "base_seed": 99,
                "enabled_options": {
                    "layouts": ["hand", "bogus"],
                    "protections": [],
                    "lighting": {"spotlight": 0, "point_lights": 1},
                    "back_to_camera_prob": 2.0,
                    "cardless_scene_prob": -1.0,
                },
                "export_yolo_segmentation": 1,
            }}, fh)
        generation = config.load_generation_settings(p)
        assert generation["count"] == 3 and generation["base_seed"] == 99
        assert generation["enabled_options"]["layouts"] == ["hand"]
        assert generation["enabled_options"]["protections"] == []
        assert generation["enabled_options"]["lighting"] == {
            "spotlight": False, "point_lights": True, "occluders": True}
        assert generation["enabled_options"]["back_to_camera_prob"] == 1.0
        assert generation["enabled_options"]["cardless_scene_prob"] == 0.0
        assert generation["export_yolo_segmentation"] is True

        config.save_generation_settings(generation, p)
        with open(p) as fh:
            saved = json.load(fh)
        assert saved["keep"] == "this"
        assert config.load_generation_settings(p) == generation

        with open(p, "w") as fh:
            json.dump({"generation": {"export_yolo_segmentation": "false"}}, fh)
        assert config.load_generation_settings(p)["export_yolo_segmentation"] is False


def test_blender_executable_is_persisted_separately_from_generation_settings():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "config.json")
        config.save_generation_settings({"count": 2, "base_seed": 50,
                                         "enabled_options": {}}, p)
        executable = r"D:\Apps\Blender\blender.exe"
        config.save_blender_executable(executable, p)
        assert config.load_blender_executable(p) == executable
        assert config.load_generation_settings(p)["count"] == 2


def test_table_texture_directory_is_persisted_and_relative_to_config():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "config.json")
        config.save_generation_settings({"count": 1, "base_seed": 1}, p)
        config.save_table_texture_dir("textures", p)
        assert config.load_table_texture_dir(p) == os.path.join(d, "textures")
        with open(p, encoding="utf-8") as handle:
            assert json.load(handle)["table_texture_dir"] == "textures"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
