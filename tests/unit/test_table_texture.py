"""Bpy-free tests for table texture source discovery."""
from __future__ import annotations

import os
import sys
import tempfile

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from texturegen.table_texture import discover_table_textures  # noqa: E402


def test_discovery_is_recursive_filtered_and_sorted():
    with tempfile.TemporaryDirectory() as directory:
        nested = os.path.join(directory, "nested")
        os.makedirs(nested)
        for path in (os.path.join(directory, "z.JPG"), os.path.join(directory, "a.png"),
                     os.path.join(nested, "b.webp"), os.path.join(nested, "ignore.txt")):
            with open(path, "wb") as handle:
                handle.write(b"x")
        assert discover_table_textures(directory) == [
            os.path.join(directory, "a.png"),
            os.path.join(directory, "z.JPG"),
            os.path.join(nested, "b.webp"),
        ]


def test_empty_setting_disables_photographic_textures_and_bad_directory_fails():
    assert discover_table_textures("") == []
    try:
        discover_table_textures("/definitely/not/a/table/texture/directory")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid configured texture directory must fail explicitly")


def _run_all():
    functions = [value for key, value in sorted(globals().items())
                 if key.startswith("test_") and callable(value)]
    for function in functions:
        function()
        print(f"  PASS {function.__name__}")
    print(f"\n{len(functions)}/{len(functions)} tests passed.")


if __name__ == "__main__":
    _run_all()
