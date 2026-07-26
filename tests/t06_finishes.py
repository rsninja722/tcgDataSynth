"""
Phase 3 (Blender) - HOLO finish comparison: two approaches x patterns x 3 view angles.

WHY THIS EXISTS
    Compare (1) thin-film iridescence vs (2) faked-spectral goniochromism, and check
    that COLOUR (not just brightness) shifts with view angle. The SAME card is rendered
    at three camera azimuths for each version so the hue shift is directly comparable.
    Patterns: none / cosmos / horizontal_lines / water_web (region = entire).

REQUIRES cv2 in Blender's Python (pip install opencv-python-headless into Blender's
python) to generate the pattern textures.

HOW TO RUN (headless):
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t06_finishes.py

OUTPUT (out/): t06_{thinfilm,spectral}_{pattern}_az{0,25,45}.png (+ .txt labels).

REPORT BACK: attach the PNGs. Compare the two versions: does the hue shift with
camera angle (good holo) or only the brightness (bad)? Which looks better per pattern?
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
    from texturegen import holo
    from texturegen.cardsource import CardLibrary
    _HAVE_CV2 = True
except Exception as exc:  # noqa: BLE001
    print(f"[t06] cv2/texturegen unavailable: {exc}\n"
          f"      Install into Blender's python:\n"
          f'      "...\\Blender 5.0\\5.0\\python\\bin\\python.exe" -m pip install opencv-python-headless')
    _HAVE_CV2 = False

PATTERNS = ["none", "cosmos", "horizontal_lines", "water_web"]
VERSIONS = ["spectral"]        # thin-film dropped (reflected white, not spectral)
ANGLES = [0, 25, 45]           # camera azimuths (degrees); same card, moving camera
CACHE = None                   # set in main()


def gen_pattern_assets(pattern, seed, out_dir):
    w, h = 504, 704            # card aspect (63:88)
    g = holo.holo_pattern(w, h, pattern, seed)
    p_path = os.path.join(out_dir, f"holo_pat_{pattern}.png")
    n_path = os.path.join(out_dir, f"holo_nrm_{pattern}.png")
    cv2.imwrite(p_path, g)
    cv2.imwrite(n_path, holo.pattern_normal(g)[:, :, ::-1])  # RGB->BGR
    return p_path, n_path


def render_one(version, pattern, az, pat_path, nrm_path, card):
    sc.reset_scene()
    sc.setup_world(gray=0.12)
    scene = bpy.context.scene
    setup_render(scene, verbose=False)

    card_id = card.card_id if card else "testcard"
    front_path = card.path if card else None
    region = card.picture_region if card else config.DEFAULT_PICTURE_REGION
    back_path = config.back_image_path()

    obj = cf.build_card_unit("Card", card_id, front_image_path=front_path,
                             back_image_path=back_path)
    mat = finishes.make_holo(f"Holo_{version}", version, front_path, pat_path, nrm_path,
                             region, "entire", pattern, seed=0)
    obj.material_slots[0].link = "OBJECT"
    obj.material_slots[0].material = mat

    dist = sc.frame_distance(50, subject_h=config.CARD_H_M, target_frac=0.72)
    cam = sc.setup_camera(50, az, 14, dist)
    # Fixed lights per (version,pattern) so only the CAMERA angle changes across az.
    import numpy as np
    light_seed = 6000 + VERSIONS.index(version) * 100 + PATTERNS.index(pattern)
    sc.add_lights(cam.location, rng=np.random.default_rng(light_seed))

    bpy.context.view_layer.update()
    lbl, reason, _ = label_card(scene, cam, obj, card_id)
    name = f"t06_{version}_{pattern}_az{az}"
    scene.render.filepath = os.path.join(_ROOT, config.OUTPUT.root, f"{name}.png")
    bpy.ops.render.render(write_still=True)
    write_label_file(os.path.join(_ROOT, config.OUTPUT.root, f"{name}.txt"),
                     [lbl] if lbl else [])
    print(f"  rendered {name} ({reason})")


def main():
    if not _HAVE_CV2:
        print("[t06] aborting: cv2 required. See message above.")
        return
    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    os.makedirs(out_dir, exist_ok=True)

    card = None
    try:
        lib = CardLibrary()
        if not lib.is_empty():
            import numpy as np
            card = lib.select(np.random.default_rng(4))
            print(f"[t06] card: {card.path} region={card.picture_region}")
    except Exception as exc:  # noqa: BLE001
        print(f"[t06] card lib: {exc}")

    # Generate pattern textures once.
    assets = {p: gen_pattern_assets(p, seed=0, out_dir=out_dir) for p in PATTERNS}

    for version in VERSIONS:
        for pattern in PATTERNS:
            pat_path, nrm_path = assets[pattern]
            for az in ANGLES:
                render_one(version, pattern, az, pat_path, nrm_path, card)
    print("\n[t06] done. Compare thinfilm vs spectral; check hue shift across az angles.")


if __name__ == "__main__":
    main()
