"""Deterministic discovery of photographic table texture sources."""
from __future__ import annotations

import os
from typing import Sequence

import config


def discover_table_textures(directory: str,
                            extensions: Sequence[str] = config.CARD_IMAGE_EXTS) -> list[str]:
    """Return recursively discovered image paths in stable lexical order."""
    if not directory:
        return []
    root = os.path.abspath(directory)
    if not os.path.isdir(root):
        raise ValueError(f"Table texture directory does not exist: {directory!r}")
    allowed = {extension.lower() for extension in extensions}
    paths = []
    for current, dirs, files in os.walk(root):
        dirs.sort()
        for filename in sorted(files):
            if os.path.splitext(filename)[1].lower() in allowed:
                paths.append(os.path.join(current, filename))
    if not paths:
        raise ValueError(f"Table texture directory contains no supported images: {directory!r}")
    return paths
