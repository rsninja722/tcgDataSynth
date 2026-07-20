"""
asset_lib.py - link/share the pre-built protection meshes at generation time
(bpy REQUIRED). Loads a mesh datablock ONCE from assets/protection_lib.blend and
reuses it across instances. If the library hasn't been built (assets_build.py not
run yet), falls back to building the mesh in-process so generation still works.

Materials are applied per instance by the caller (protection.make_* -> slots); the
mesh here carries only geometry + empty material slots.
"""
from __future__ import annotations

import os

import bpy

from blender import protection as prot

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
LIB_PATH = os.path.join(_ROOT, "assets", "protection_lib.blend")

# In-process builders (fallback + cache source).
_BUILDERS = {
    "asset_sleeve_1mm": lambda: prot.build_sleeve_mesh("1mm", name="asset_sleeve_1mm"),
    "asset_sleeve_2p5mm": lambda: prot.build_sleeve_mesh("2.5mm", name="asset_sleeve_2p5mm"),
    "asset_toploader": lambda: prot._flat_pocket_mesh(
        "asset_toploader", prot.TOPLOADER_W, prot.TOPLOADER_H, prot.TOPLOADER_GAP,
        0.0, spine_width=prot._SPINE_W),
    "asset_semirigid": lambda: prot._flat_pocket_mesh(
        "asset_semirigid", prot.SEMIRIGID_W, prot.SEMIRIGID_H, prot.SEMIRIGID_GAP,
        prot.SEMIRIGID_LIP, spine_width=0.0),
    "asset_slab": lambda: prot._slab_mesh("asset_slab"),
}

# Map card protection kind -> (asset name, sleeve size or None) helpers for layouts.
SLEEVE_ASSET = {"1mm": "asset_sleeve_1mm", "2.5mm": "asset_sleeve_2p5mm"}


def get_protection_mesh(asset_name: str, use_library: bool = True):
    """Return a shared mesh datablock for `asset_name`. Reuses an already-loaded
    datablock if present (so a 12-card scene shares one mesh); otherwise links it
    from the library .blend, or builds it in-process if the library is missing."""
    if asset_name not in _BUILDERS:
        raise KeyError(f"unknown protection asset {asset_name!r}")
    existing = bpy.data.meshes.get(asset_name)
    if existing is not None:
        return existing
    if use_library and os.path.isfile(LIB_PATH):
        try:
            with bpy.data.libraries.load(LIB_PATH, link=False) as (src, dst):
                if asset_name in src.meshes:
                    dst.meshes = [asset_name]
            loaded = bpy.data.meshes.get(asset_name)
            if loaded is not None:
                return loaded
        except Exception as exc:  # noqa: BLE001
            print(f"[asset_lib] library load failed for {asset_name}: {exc}; building in-process")
    return _BUILDERS[asset_name]()
