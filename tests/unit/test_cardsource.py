"""
Docker unit tests for texturegen/cardsource.py (pure Python, no bpy, no Blender).

Run:  python3 tests/unit/test_cardsource.py
(also pytest-compatible: pytest tests/unit/test_cardsource.py)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

# Make the project root importable regardless of cwd.
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import config  # noqa: E402
from texturegen import cardsource  # noqa: E402


def _make_tree(base: str) -> None:
    """Create a nested folder tree with some images and some non-images."""
    os.makedirs(os.path.join(base, "set1"), exist_ok=True)
    os.makedirs(os.path.join(base, "set2", "holos"), exist_ok=True)
    for rel in [
        "set1/charizard.png",
        "set1/pikachu.jpg",
        "set1/notes.txt",          # ignored
        "set2/blastoise.PNG",      # case-insensitive ext
        "set2/holos/mewtwo.webp",
        "set2/holos/thumbs.db",    # ignored
    ]:
        p = os.path.join(base, rel)
        with open(p, "wb") as fh:
            fh.write(b"\x89PNG\r\n")  # content irrelevant; discovery is by extension


def test_discovery_finds_only_images_recursively():
    with tempfile.TemporaryDirectory() as base:
        _make_tree(base)
        paths = cardsource.discover_card_paths(base)
        ids = sorted(cardsource.card_id_from_path(p) for p in paths)
        assert ids == ["blastoise", "charizard", "mewtwo", "pikachu"], ids
        # sorted() determinism
        assert paths == sorted(paths)


def test_back_image_is_excluded_from_faces():
    with tempfile.TemporaryDirectory() as base:
        _make_tree(base)
        # Drop the generic back texture into the root; it must NOT be selectable.
        with open(os.path.join(base, config.BACK_IMAGE_FILENAME), "wb") as fh:
            fh.write(b"\x89PNG\r\n")
        ids = sorted(cardsource.card_id_from_path(p)
                     for p in cardsource.discover_card_paths(base))
        assert "back" not in ids, ids
        assert ids == ["blastoise", "charizard", "mewtwo", "pikachu"], ids


def test_card_id_is_filename_stem():
    assert cardsource.card_id_from_path("/a/b/Base_Set_004.png") == "Base_Set_004"
    assert cardsource.card_id_from_path("x.JPEG") == "x"


def test_default_region_when_no_override():
    with tempfile.TemporaryDirectory() as base:
        _make_tree(base)
        lib = cardsource.CardLibrary(root=base)
        card = lib.card_for_path(os.path.join(base, "set1", "charizard.png"))
        assert card.card_id == "charizard"
        assert card.picture_region == config.DEFAULT_PICTURE_REGION


def test_region_override_file_applies():
    with tempfile.TemporaryDirectory() as base:
        _make_tree(base)
        override = {"charizard": [0.1, 0.2, 0.8, 0.5], "bad": "nope"}
        with open(os.path.join(base, config.PICTURE_REGIONS_FILENAME), "w") as fh:
            json.dump(override, fh)
        lib = cardsource.CardLibrary(root=base)
        chariz = lib.card_for_path(os.path.join(base, "set1", "charizard.png"))
        pika = lib.card_for_path(os.path.join(base, "set1", "pikachu.jpg"))
        assert chariz.picture_region == (0.1, 0.2, 0.8, 0.5)     # overridden
        assert pika.picture_region == config.DEFAULT_PICTURE_REGION  # default
        assert "bad" not in lib.region_overrides                  # malformed skipped


def test_selection_is_seed_deterministic():
    with tempfile.TemporaryDirectory() as base:
        _make_tree(base)
        lib = cardsource.CardLibrary(root=base)
        a = lib.select(np.random.default_rng(1234))
        b = lib.select(np.random.default_rng(1234))
        c = lib.select(np.random.default_rng(9999))
        assert a == b, "same seed must pick same card"
        # Different seed *may* pick a different card; over a few draws it should differ.
        draws_seed1 = [lib.select(np.random.default_rng(s)).card_id for s in range(20)]
        assert len(set(draws_seed1)) > 1, "selection should span multiple cards"


def test_empty_library_raises():
    with tempfile.TemporaryDirectory() as base:
        lib = cardsource.CardLibrary(root=base)
        assert lib.is_empty()
        try:
            lib.select(np.random.default_rng(0))
        except RuntimeError as exc:
            assert "No card images" in str(exc)
        else:
            raise AssertionError("empty library should raise RuntimeError")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
