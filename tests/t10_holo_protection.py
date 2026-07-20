"""
Diagnostic - a FULL holo card under identical camera + lighting in 4 protection
configs, to isolate which layer (sleeve vs toploader) changes the holo look:

    t10_bare.png              - holo card, no protection
    t10_sleeve.png            - holo card in a clear sleeve
    t10_toploader.png         - holo card in a toploader (NO sleeve; rule-violating,
                                but fine for this temporary diagnostic)
    t10_sleeve_toploader.png  - holo card in a sleeve AND a toploader

Same card / camera / lights across all four so differences are purely the plastic.
Materials are the reverted (transmission-glass) ones.

HOW TO RUN (headless; cv2 in Blender):
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t10_holo_protection.py

REPORT BACK: attach the 4 PNGs. Compare bare vs each protected version to see exactly
what the sleeve / toploader do to the holo.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import bpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from blender import card_factory as cf  # noqa: E402
from blender import protection as prot  # noqa: E402
from blender import finishes  # noqa: E402
from blender.render_setup import setup_render  # noqa: E402
from blender import scene_common as sc  # noqa: E402

try:
    import cv2
    from texturegen import holo
    from texturegen.cardsource import CardLibrary
    _HAVE_CV2 = True
except Exception as exc:  # noqa: BLE001
    print(f"[t10] cv2/texturegen unavailable: {exc}")
    _HAVE_CV2 = False

ASSETS = os.path.join(_ROOT, "assets")
CONFIGS = ["bare", "sleeve", "toploader", "sleeve_toploader"]
LIGHT_SEED = 777
PATTERN = "cosmos"      # full-card holo pattern for the comparison


def add_protection(cfg_name, card, warp, wear):
    objs = []
    if "sleeve" in cfg_name:
        objs.append(prot.build_sleeve("Sleeve", card, "clear", "1mm", warp))
    if "toploader" in cfg_name:
        tl = prot.build_toploader("TL", warp, wear_map_path=wear, tint=(0.9, 0.92, 0.95))
        card.parent = tl          # card centered inside the toploader (both at origin)
        objs.append(tl)
    return objs


def run_config(i, cfg_name, card_img, pat_path, nrm_path):
    import numpy as np
    sc.reset_scene()
    sc.setup_world(gray=0.12)
    scene = bpy.context.scene
    setup_render(scene, verbose=(i == 0))

    warp = os.path.join(ASSETS, "plastic_warp_0.png")
    wear = os.path.join(ASSETS, "toploader_wear_0.png")

    card = cf.build_card_unit("Card", card_img.card_id, front_image_path=card_img.path,
                              back_image_path=config.back_image_path())
    mat = finishes.make_holo("Holo", "spectral", card_img.path, pat_path, nrm_path,
                             card_img.picture_region, "entire", PATTERN)
    card.material_slots[0].link = "OBJECT"
    card.material_slots[0].material = mat
    add_protection(cfg_name, card, warp, wear)

    dist = sc.frame_distance(50, subject_h=config.CARD_H_M, target_frac=0.8)
    cam = sc.setup_camera(50, 22, 15, dist)
    sc.add_lights(cam.location, rng=np.random.default_rng(LIGHT_SEED))  # identical every config

    bpy.context.view_layer.update()
    scene.render.filepath = os.path.join(_ROOT, config.OUTPUT.root, f"t10_{cfg_name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"  rendered t10_{cfg_name}")


def main():
    if not _HAVE_CV2:
        print("[t10] cv2 required; aborting.")
        return
    lib = CardLibrary()
    if lib.is_empty():
        print("[t10] no card images; aborting")
        return
    import numpy as np
    card_img = lib.select(np.random.default_rng(3))
    print(f"[t10] card: {card_img.path}  pattern={PATTERN}")

    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    os.makedirs(out_dir, exist_ok=True)
    w, h = 504, 704
    g = holo.holo_pattern(w, h, PATTERN, seed=0)
    pat_path = os.path.join(out_dir, f"t10_holo_pat_{PATTERN}.png")
    nrm_path = os.path.join(out_dir, f"t10_holo_nrm_{PATTERN}.png")
    cv2.imwrite(pat_path, g)
    cv2.imwrite(nrm_path, holo.pattern_normal(g)[:, :, ::-1])

    for i, cfg_name in enumerate(CONFIGS):
        run_config(i, cfg_name, card_img, pat_path, nrm_path)
    print("\n[t10] done. Compare t10_bare vs sleeve/toploader/sleeve_toploader.")


if __name__ == "__main__":
    main()
