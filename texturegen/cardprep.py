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

import os
from typing import Optional

import cv2

from texturegen import damage as dmg
from texturegen import physical_texture as pt
from texturegen.cardsource import card_id_from_path


def _ensure(d: str):
    os.makedirs(d, exist_ok=True)


def physical_normal_path(card_path: str, cache_dir: str, seed: int = 0,
                         strength: float = 1.4) -> Optional[str]:
    """Etched-foil normal map for this card (cached per card ID; deterministic)."""
    cid = card_id_from_path(card_path)
    out = os.path.join(cache_dir, f"{cid}_physnormal.png")
    if os.path.isfile(out):
        return out
    card = cv2.imread(card_path)
    if card is None:
        return None
    _pattern, normal = pt.generate_physical_texture(card, seed=seed, normal_strength=strength)
    _ensure(cache_dir)
    cv2.imwrite(out, normal[:, :, ::-1])   # RGB -> BGR for cv2
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
    out = os.path.join(cache_dir, f"{cid}_dmg_{flags}_s{seed}.png")
    if os.path.isfile(out):
        return out
    card = cv2.imread(card_path)
    if card is None:
        return card_path
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
    cv2.imwrite(out, comp)
    return out
