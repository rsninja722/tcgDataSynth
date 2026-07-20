"""
Scene builder (bpy REQUIRED) - the Phase 4 integration point. Turns a validated
SceneConfig (from rules/combinations.py) into actual Blender objects: full card
instances (base unit + finish + damage + physical texture + protection), assembled
per CardConfig. Layout modules position the returned instances.

Requires cv2 in Blender's Python for damage/physical/holo-pattern generation; falls
back to a plain textured card if cv2 is missing.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional

import bpy

import config
from blender import card_factory as cf
from blender import finishes
from blender import protection as prot

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ASSETS = os.path.join(_ROOT, "assets")
MM = 0.001

try:
    import cv2  # noqa: F401
    from texturegen import holo, cardprep
    _HAVE_CV2 = True
except Exception as exc:  # noqa: BLE001
    print(f"[scene_builder] cv2/texturegen unavailable ({exc}); cards use plain textures.")
    _HAVE_CV2 = False


@dataclass
class CardInstance:
    """One assembled card. `root` is what a layout positions (moving it moves the
    whole protected unit); `card` carries the finish + the labeled corners."""
    root: object
    card: object
    card_id: str
    objects: List[object]
    holo_tag: str = "none"   # none | full | holo | reverse (written into the label)


# Region mode -> holo label tag.
_HOLO_TAG_BY_REGION = {"entire": "full", "picture": "holo", "reverse": "reverse"}


def holo_tag_for_finish(finish) -> str:
    """Label tag for a FinishConfig: 'none' unless holo, then by region mode."""
    if finish.kind != "holo":
        return "none"
    return _HOLO_TAG_BY_REGION.get(finish.holo_region, "none")


def _asset(rng, prefix: str, n: int = 6) -> str:
    return os.path.join(ASSETS, f"{prefix}_{int(rng.integers(0, n))}.png")


def protection_footprint(pcfg) -> (float, float):
    """Outer (width, height) in meters of a card + its protection (for spacing)."""
    k = pcfg.kind
    if k == "slab":
        return prot.SLAB_W, prot.SLAB_H
    if k == "toploader":
        return prot.TOPLOADER_W, prot.TOPLOADER_H
    if k == "semi_rigid":
        return prot.SEMIRIGID_W, prot.SEMIRIGID_H + prot.SEMIRIGID_LIP
    if k == "sleeve" and pcfg.sleeve is not None:
        m = prot.SLEEVE_MARGINS.get(pcfg.sleeve.size, 0.001)
        return config.CARD_W_M + 2 * m, config.CARD_H_M + 2 * m
    return config.CARD_W_M, config.CARD_H_M


def protection_half_thickness(pcfg) -> float:
    """Half the z-extent, so an object can be lifted to rest ON a surface."""
    k = pcfg.kind
    if k == "slab":
        return prot.SLAB_T / 2.0
    if k in ("toploader", "semi_rigid"):
        return prot.TOPLOADER_GAP / 2.0 + 0.0003
    if k == "sleeve":
        return config.CARD_T_M / 2.0 + 0.0002
    return config.CARD_T_M / 2.0


def _gen_holo_pattern(pattern: str, cache_dir: str, seed: int):
    if not _HAVE_CV2:
        return None, None
    w, h = 504, 704
    g = holo.holo_pattern(w, h, pattern, seed)
    p = os.path.join(cache_dir, f"holo_pat_{pattern}_{seed}.png")
    n = os.path.join(cache_dir, f"holo_nrm_{pattern}_{seed}.png")
    cv2.imwrite(p, g)
    cv2.imwrite(n, holo.pattern_normal(g)[:, :, ::-1])
    return p, n


def _front_and_physical(card_img, damage, physical: bool, cache_dir: str, seed: int):
    """Return (front_image_path, physical_normal_path)."""
    front = card_img.path
    phys = None
    if _HAVE_CV2:
        front = cardprep.damaged_card_path(card_img.path, cache_dir, seed,
                                           dirt=damage.dirt, scratches=damage.scratches,
                                           surface=damage.surface)
        if physical:
            phys = cardprep.physical_normal_path(card_img.path, cache_dir)
    return front, phys


def _finish_material(name, finish, front, phys, card_img, cache_dir, seed):
    if finish.kind == "normal":
        return finishes.make_normal_material(name + "_fin", front, physical_normal_path=phys)
    pat, nrm = _gen_holo_pattern(finish.holo_pattern, cache_dir, seed)
    if pat is None:
        return finishes.make_normal_material(name + "_fin", front, physical_normal_path=phys)
    return finishes.make_holo(name + "_fin", "spectral", front, pat, nrm,
                              card_img.picture_region, finish.holo_region,
                              finish.holo_pattern, physical_normal_path=phys)


def _add_protection(name, card, pcfg, rng) -> (object, List[object]):
    """Build protection around `card`. Returns (root_object, all_protection_objects).
    root = the outermost object a layout should position (holder/slab, else the card)."""
    kind = pcfg.kind
    warp = _asset(rng, "plastic_warp")
    wear = _asset(rng, "toploader_wear")
    objs: List[object] = []
    if kind == "none":
        return card, objs

    # Sleeve (present for 'sleeve', 'toploader', 'semi_rigid').
    if pcfg.sleeve is not None:
        sv = prot.build_sleeve(name + "_slv", card, pcfg.sleeve.sleeve_type, pcfg.sleeve.size,
                               warp, uv_xform=prot.random_uv_xform(rng))
        objs.append(sv)
    if kind == "sleeve":
        return card, objs

    # Holder / slab: build at origin, then nest the card inside with its offset.
    tint = (0.9, 0.92, 0.95)
    uvx, wuvx = prot.random_uv_xform(rng), prot.random_uv_xform(rng)
    if kind == "toploader":
        holder = prot.build_toploader(name + "_tl", warp, wear_map_path=wear, tint=tint,
                                      uv_xform=uvx, wear_uv_xform=wuvx)
    elif kind == "semi_rigid":
        holder = prot.build_semirigid(name + "_sr", warp, wear_map_path=wear, tint=tint,
                                      uv_xform=uvx, wear_uv_xform=wuvx)
    elif kind == "slab":
        holder = prot.build_slab(name + "_slab", warp, wear_map_path=wear,
                                 label_path=_asset(rng, "slab_label"), tint=tint,
                                 uv_xform=uvx, wear_uv_xform=wuvx)
    else:
        return card, objs
    objs.append(holder)

    # Nest the card (and its sleeve) inside the holder at the configured offset.
    if kind == "slab":
        card.location = prot.SLAB_CARD_POS
    else:
        off = pcfg.inner_offset_mm or [0.0, 0.0]
        card.location = (off[0] * MM, off[1] * MM, 0.0)
        card.rotation_euler = (0.0, 0.0, math.radians(pcfg.inner_rot_deg or 0.0))
    card.parent = holder
    return holder, objs


def build_card_instance(name: str, card_cfg, card_img, cache_dir: str, rng) -> CardInstance:
    """Assemble one full card instance from a CardConfig + selected card image."""
    seed = int(rng.integers(0, 2 ** 30))
    front, phys = _front_and_physical(card_img, card_cfg.damage,
                                      card_cfg.finish.physical_texture, cache_dir, seed)
    card = cf.build_card_unit(name, card_img.card_id, front_image_path=front,
                              back_image_path=config.back_image_path())
    mat = _finish_material(name, card_cfg.finish, front, phys, card_img, cache_dir, seed)
    card.material_slots[0].link = "OBJECT"
    card.material_slots[0].material = mat

    root, prot_objs = _add_protection(name, card, card_cfg.protection, rng)
    return CardInstance(root=root, card=card, card_id=card_img.card_id,
                        objects=[card] + prot_objs,
                        holo_tag=holo_tag_for_finish(card_cfg.finish))
