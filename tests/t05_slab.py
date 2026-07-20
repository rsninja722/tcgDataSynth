"""
Phase 2 (step 4) - GRADED SLAB (spec §3.2), the last protection type.

WHY THIS EXISTS
    Validates the slab: 80x135mm, 6.7mm thick; label area (20x68mm, 4mm from top);
    a horizontal line 4mm below it; a card-recess outline; card sealed inside with
    NO sleeve (slab rule). Reuses the toploader clear surface for the surface and
    the spine material for the edges + internal rectangle outlines (ridges). Card
    stays labeled through the clear surface.

HOW TO RUN (headless):
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t05_slab.py

WHAT IT PRODUCES (out/)
    t05_slab_front.png/.txt, t05_slab_tilt.png/.txt

WHAT TO REPORT BACK
    Attach the 2 PNGs. PASS if: slab shows a clear body with the card in the lower
    recess, a procedural label at the top, ridge outlines for the label/line/recess,
    opaque edges, surface wear, and the card is labeled correctly.
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
from blender import protection as prot  # noqa: E402
from blender.labeling import label_card  # noqa: E402
from blender.render_setup import setup_render  # noqa: E402
from blender import scene_common as sc  # noqa: E402
from labeltools.yolo_pose import write_label_file  # noqa: E402

try:
    from texturegen.cardsource import CardLibrary
except Exception:  # noqa: BLE001
    CardLibrary = None

ASSETS = os.path.join(_ROOT, "assets")

# name, cam(az, el)
CASES = [
    ("slab_front", (0, 0)),
    ("slab_tilt", (26, 16)),
]


def run_case(i, name, cam_az, cam_el, front_path, back_path):
    sc.reset_scene()
    sc.setup_world()
    scene = bpy.context.scene
    setup_render(scene, verbose=(i == 0))

    import numpy as np
    rng = np.random.default_rng(300 + i)
    warp = os.path.join(ASSETS, f"plastic_warp_{int(rng.integers(0, 6))}.png")
    wear = os.path.join(ASSETS, f"toploader_wear_{int(rng.integers(0, 6))}.png")
    label = os.path.join(ASSETS, f"slab_label_{int(rng.integers(0, 6))}.png")
    t = float(rng.random())
    tint = (0.90 - 0.12 * t, 0.90 - 0.05 * t, min(0.99, 0.90 + 0.05 * t))
    wear_rough = 0.05 + float(rng.random()) * 0.30

    # Card sits in the recess (no sleeve - slab rule).
    card_id = os.path.splitext(os.path.basename(front_path))[0] if front_path else "testcard"
    card = cf.build_card_unit("Card", card_id, front_image_path=front_path,
                              back_image_path=back_path)
    card.location = prot.SLAB_CARD_POS

    prot.build_slab("Slab", warp, wear_map_path=wear, label_path=label, tint=tint,
                    wear_rough=wear_rough, uv_xform=prot.random_uv_xform(rng),
                    wear_uv_xform=prot.random_uv_xform(rng))

    dist = sc.frame_distance(45, subject_h=prot.SLAB_H, target_frac=0.82)
    cam = sc.setup_camera(45, cam_az, cam_el, dist, target=(0.0, 0.0, 0.0))
    sc.add_lights(cam.location, target=(0.0, 0.0, 0.0))

    bpy.context.view_layer.update()
    lbl, reason, dbg = label_card(scene, cam, card, card_id)

    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    os.makedirs(out_dir, exist_ok=True)
    scene.render.filepath = os.path.join(out_dir, f"t05_{name}.png")
    bpy.ops.render.render(write_still=True)
    write_label_file(os.path.join(out_dir, f"t05_{name}.txt"), [lbl] if lbl else [])
    print(f"=== {name} -> {reason} ===")
    print(f"   {'LABEL: ' + lbl.to_line() if lbl else 'NO LABEL (' + reason + ')'}")


def main():
    front_path = None
    back_path = config.back_image_path()
    if CardLibrary is not None:
        try:
            lib = CardLibrary()
            if not lib.is_empty():
                import numpy as np
                front_path = lib.select(np.random.default_rng(9)).path
                print(f"[t05] front card: {front_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[t05] card lib: {exc}")

    for i, (name, cam) in enumerate(CASES):
        run_case(i, name, cam[0], cam[1], front_path, back_path)
    print("\n[t05] done. Eyeball slab geometry/label/ridges; confirm card labeled.")


if __name__ == "__main__":
    main()
