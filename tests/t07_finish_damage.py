"""
Phase 3 (Blender) - physical-texture (etched) normal + damage compositing on the card.

WHY THIS EXISTS
    Reviews the two remaining Phase-3 pieces wired end-to-end (cv2 in Blender):
      - §3.4 physical-texture etched-foil NORMAL from the card art (raised layer),
      - damage compositing (dirt / scratches / surface whitening + rips) onto the face,
    on both a normal finish and a holo finish.

HOW TO RUN (headless), requires cv2 in Blender's python:
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t07_finish_damage.py

OUTPUT (out/): t07_{case}.png (+ labels). Cases:
    normal_plain, normal_physical, normal_damage, holo_cosmos_phys_damage, holo_lines

REPORT BACK: attach the PNGs. PASS if: physical texture shows fine etched lines following
the art (subtle raised look), damage shows grime/scratches/edge whitening+rips, and both
combine sensibly with the holo flash.
"""
import os
import sys
import math
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import bpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from blender import card_factory as cf  # noqa: E402
from blender import finishes  # noqa: E402
from blender.labeling import label_card  # noqa: E402
from blender.render_setup import setup_render  # noqa: E402
from blender import scene_common as sc  # noqa: E402
from labeltools.yolo_pose import write_label_file  # noqa: E402

try:
    import cv2
    from texturegen import holo, cardprep
    from texturegen.cardsource import CardLibrary
    _HAVE_CV2 = True
except Exception as exc:  # noqa: BLE001
    print(f"[t07] cv2/texturegen unavailable: {exc}\n      Install opencv into Blender's python.")
    _HAVE_CV2 = False

CACHE = None
CASES = ["normal_plain", "normal_physical", "normal_damage",
         "holo_none_phys_damage", "holo_lines"]


def gen_pattern(pattern):
    w, h = 504, 704
    g = holo.holo_pattern(w, h, pattern, seed=0)
    p = os.path.join(CACHE, f"holo_pat_{pattern}.png")
    n = os.path.join(CACHE, f"holo_nrm_{pattern}.png")
    cv2.imwrite(p, g)
    cv2.imwrite(n, holo.pattern_normal(g)[:, :, ::-1])
    return p, n


def build_material(case, card):
    path = card.path if card else None
    region = card.picture_region if card else config.DEFAULT_PICTURE_REGION
    if case == "normal_plain":
        return finishes.make_normal_material("M", path)
    if case == "normal_physical":
        phys = cardprep.physical_normal_path(path, CACHE) if path else None
        return finishes.make_normal_material("M", path, physical_normal_path=phys)
    if case == "normal_damage":
        dmg = cardprep.damaged_card_path(path, CACHE, seed=1, dirt=True, scratches=True,
                                         surface=True) if path else path
        return finishes.make_normal_material("M", dmg)
    if case == "holo_none_phys_damage":
        # Rule: physical texture pairs with the 'none' holo pattern.
        phys = cardprep.physical_normal_path(path, CACHE) if path else None
        dmg = cardprep.damaged_card_path(path, CACHE, seed=2, surface=True) if path else path
        pat, nrm = gen_pattern("none")
        return finishes.make_holo("M", "spectral", dmg, pat, nrm, region, "entire",
                                  "none", physical_normal_path=phys)
    if case == "holo_lines":
        pat, nrm = gen_pattern("horizontal_lines")
        return finishes.make_holo("M", "spectral", path, pat, nrm, region, "entire",
                                  "horizontal_lines")
    raise ValueError(case)


def run_case(i, case, card):
    sc.reset_scene()
    sc.setup_world(gray=0.15)
    scene = bpy.context.scene
    setup_render(scene, verbose=(i == 0))

    card_id = card.card_id if card else "testcard"
    obj = cf.build_card_unit("Card", card_id, front_image_path=(card.path if card else None),
                             back_image_path=config.back_image_path())
    obj.material_slots[0].link = "OBJECT"
    obj.material_slots[0].material = build_material(case, card)

    import numpy as np
    dist = sc.frame_distance(50, subject_h=config.CARD_H_M, target_frac=0.75)
    cam = sc.setup_camera(50, 22, 14, dist)
    sc.add_lights(cam.location, rng=np.random.default_rng(700 + i))

    bpy.context.view_layer.update()
    lbl, reason, _ = label_card(scene, cam, obj, card_id)
    scene.render.filepath = os.path.join(_ROOT, config.OUTPUT.root, f"t07_{case}.png")
    bpy.ops.render.render(write_still=True)
    write_label_file(os.path.join(_ROOT, config.OUTPUT.root, f"t07_{case}.txt"),
                     [lbl] if lbl else [])
    print(f"  rendered t07_{case} ({reason})")


def main():
    global CACHE
    if not _HAVE_CV2:
        return
    CACHE = os.path.join(_ROOT, config.OUTPUT.root, "card_cache")
    os.makedirs(CACHE, exist_ok=True)
    card = None
    try:
        lib = CardLibrary()
        if not lib.is_empty():
            import numpy as np
            card = lib.select(np.random.default_rng(6))
            print(f"[t07] card: {card.path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[t07] card lib: {exc}")

    for i, case in enumerate(CASES):
        run_case(i, case, card)
    print("\n[t07] done. Eyeball physical texture + damage on normal and holo finishes.")


if __name__ == "__main__":
    main()
