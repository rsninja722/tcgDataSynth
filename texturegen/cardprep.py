"""
Per-card texture prep (bpy-FREE, Docker-testable; runs in Blender too since cv2 is
now installed there). Produces the image files the Blender front material loads:

  - physical_normal_path(): the §3.4 etched-foil normal map from the card art
    (deterministic per card -> cached per card ID).
  - damaged_card_path(): the card face with random dirt/scratches/surface-damage
    composited (varies per INSTANCE via seed -> cached per (card, seed) so the same
    card shows different wear each appearance; see texturegen/damage.py note).

Card images live on the user's machine; these run at generation time. In Docker we
test them against a synthetic card.
"""
from __future__ import annotations

import hashlib
import os

import cv2

from texturegen import damage as dmg
from texturegen import physical_texture as pt
from texturegen.cardsource import card_id_from_path

_CACHE_VERSION = "v2"


def _ensure(d: str):
    os.makedirs(d, exist_ok=True)


def _cache_token(card_path: str, *settings: object) -> str:
    """Digest source bytes and settings so stale generated textures are not reused."""
    digest = hashlib.sha256()
    digest.update(_CACHE_VERSION.encode("ascii"))
    with open(card_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    for setting in settings:
        digest.update(repr(setting).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _write_image(path: str, image) -> None:
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Could not write generated texture: {path}")


def physical_normal_path(card_path: str, cache_dir: str, seed: int = 0,
                          strength: float = 1.4) -> str:
    """Etched-foil normal map for this card (cached per card ID; deterministic)."""
    cid = card_id_from_path(card_path)
    token = _cache_token(card_path, "physical_normal", int(seed), float(strength))
    out = os.path.join(cache_dir, f"{cid}_physnormal_{token}.png")
    if os.path.isfile(out):
        return out
    card = cv2.imread(card_path)
    if card is None:
        raise RuntimeError(f"Could not read card image: {card_path}")
    _pattern, normal = pt.generate_physical_texture(card, seed=seed, normal_strength=strength)
    _ensure(cache_dir)
    _write_image(out, normal[:, :, ::-1])   # RGB -> BGR for cv2
    return out


def damaged_card_path(card_path: str, cache_dir: str, seed: int,
                      dirt: bool = False, scratches: bool = False,
                      surface: bool = False) -> str:
    """Card face with the enabled damage overlays composited (per-instance via seed).
    Returns the original path when no damage is enabled."""
    if not (dirt or scratches or surface):
        return card_path
    cid = card_id_from_path(card_path)
    flags = f"{int(dirt)}{int(scratches)}{int(surface)}"
    token = _cache_token(card_path, "damage", flags, int(seed))
    out = os.path.join(cache_dir, f"{cid}_dmg_{flags}_s{seed}_{token}.png")
    if os.path.isfile(out):
        return out
    card = cv2.imread(card_path)
    if card is None:
        raise RuntimeError(f"Could not read card image: {card_path}")
    h, w = card.shape[:2]
    overlays = []
    if dirt:
        overlays.append(dmg.dirt(w, h, seed + 11))
    if scratches:
        overlays.append(dmg.scratches(w, h, seed + 23))
    if surface:
        overlays.append(dmg.surface_damage(w, h, seed + 37))
    comp = dmg.composite_overlays(card, overlays)
    _ensure(cache_dir)
    _write_image(out, comp)
    return out
