"""
Phase 2 (step 3) - SEMI-RIGID holder + TOPLOADER (spec §3.2).

WHY THIS EXISTS
    Validates the two rigid holders around a (sleeved) card:
      - Toploader: 70x98mm, 1mm interior, sealed L/R/bottom, open top.
      - Semi-rigid: 81x108mm + a 12mm tab (lip) above on the back sheet only.
      - Card inside is sleeved (rule) and offset +/-2mm & rotated +/-2deg (spec §3.2).
      - Card stays labeled through the clear rigid plastic.
    Rigid sheets are flat & parallel, so the glass should be clean (no curved-seam
    TIR like the sleeves had).

HOW TO RUN (headless):
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t04_holders.py

WHAT IT PRODUCES (out/)
    t04_toploader_front.png/.txt, t04_toploader_tilt.png/.txt,
    t04_semirigid_front.png/.txt, t04_semirigid_tilt.png/.txt

WHAT TO REPORT BACK
    Attach the 4 PNGs + say if the card is labeled correctly through the plastic.
    PASS if: card sits inside a clear rigid holder (slightly offset/rotated), the
    semi-rigid shows its 12mm top tab, plastic is clean, and corners are labeled.
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
MM = 0.001

# name, holder, subject_height(m), inner_offset_mm(x,y), inner_rot_deg (<=1), cam(az,el)
CASES = [
    ("toploader_front", "toploader", prot.TOPLOADER_H, (2.0, 0.0), 1.0, (0, 0)),
    ("toploader_tilt", "toploader", prot.TOPLOADER_H, (-1.5, 2.0), -0.8, (28, 18)),
    ("semirigid_front", "semirigid", prot.SEMIRIGID_H + prot.SEMIRIGID_LIP, (-2.0, 1.0), -0.6, (0, 0)),
    ("semirigid_tilt", "semirigid", prot.SEMIRIGID_H + prot.SEMIRIGID_LIP, (2.0, -1.5), 0.9, (26, 16)),
]


def _random_tint_and_wear(rng, base_rough):
    """Slight bright-grey -> bluish tint, and a random scratch amount (base_rough
    == no scratches, up to 0.35 == full)."""
    t = float(rng.random())
    tint = (0.90 - 0.15 * t, 0.90 - 0.06 * t, min(0.99, 0.90 + 0.06 * t))
    wear_rough = base_rough + float(rng.random()) * (0.35 - base_rough)
    return tint, wear_rough


def run_case(i, name, holder, subject_h, inner_off_mm, inner_rot_deg, cam_az, cam_el,
             front_path, back_path):
    sc.reset_scene()
    sc.setup_world()
    scene = bpy.context.scene
    setup_render(scene, verbose=(i == 0))

    # Card (always sleeved inside a holder) offset & rotated within the holder.
    card_id = os.path.splitext(os.path.basename(front_path))[0] if front_path else "testcard"
    card = cf.build_card_unit("Card", card_id, front_image_path=front_path,
                              back_image_path=back_path)
    card.location = (inner_off_mm[0] * MM, inner_off_mm[1] * MM, 0.0)
    card.rotation_euler = (0.0, 0.0, math.radians(inner_rot_deg))
    # Per-instance randomness via a seeded rng: which base map, tint, scratch
    # amount, and (crucially) a random UV crop/zoom so the reflection & scratch
    # PATTERNS differ every instance instead of repeating across the dataset.
    import numpy as np
    rng = np.random.default_rng(200 + i)
    warp = os.path.join(ASSETS, f"plastic_warp_{int(rng.integers(0, 6))}.png")
    wear = os.path.join(ASSETS, f"toploader_wear_{int(rng.integers(0, 6))}.png")
    base_rough = 0.05 if holder == "toploader" else 0.12
    tint, wear_rough = _random_tint_and_wear(rng, base_rough)
    sleeve_uv = prot.random_uv_xform(rng)
    warp_uv = prot.random_uv_xform(rng)
    wear_uv = prot.random_uv_xform(rng)

    prot.build_sleeve("Sleeve", card, "clear", "1mm", warp, uv_xform=sleeve_uv)

    # Holder centered at origin (the card sits offset inside it).
    if holder == "toploader":
        prot.build_toploader("Toploader", warp, wear_map_path=wear, tint=tint,
                             wear_rough=wear_rough, uv_xform=warp_uv, wear_uv_xform=wear_uv)
    else:
        prot.build_semirigid("SemiRigid", warp, wear_map_path=wear, tint=tint,
                             wear_rough=wear_rough, uv_xform=warp_uv, wear_uv_xform=wear_uv)

    # Frame the holder; its center for a semi-rigid sits a bit low because of the lip.
    center_y = (prot.SEMIRIGID_LIP / 2.0) if holder == "semirigid" else 0.0
    target = (0.0, center_y, 0.0)
    dist = sc.frame_distance(45, subject_h=subject_h, target_frac=0.8)
    cam = sc.setup_camera(45, cam_az, cam_el, dist, target=target)
    sc.add_lights(cam.location, target=target)

    bpy.context.view_layer.update()
    label, reason, dbg = label_card(scene, cam, card, card_id)

    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    os.makedirs(out_dir, exist_ok=True)
    scene.render.filepath = os.path.join(out_dir, f"t04_{name}.png")
    bpy.ops.render.render(write_still=True)
    write_label_file(os.path.join(out_dir, f"t04_{name}.txt"), [label] if label else [])
    print(f"=== {name} ({holder}) offset={inner_off_mm}mm rot={inner_rot_deg} -> {reason} ===")
    print(f"   {'LABEL: ' + label.to_line() if label else 'NO LABEL (' + reason + ')'}")


def main():
    front_path = None
    back_path = config.back_image_path()
    if CardLibrary is not None:
        try:
            lib = CardLibrary()
            if not lib.is_empty():
                import numpy as np
                front_path = lib.select(np.random.default_rng(5)).path
                print(f"[t04] front card: {front_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[t04] card lib: {exc}")

    for i, (name, holder, sh, off, rot, cam) in enumerate(CASES):
        run_case(i, name, holder, sh, off, rot, cam[0], cam[1], front_path, back_path)
    print("\n[t04] done. Expect all 4 labeled; eyeball holder geometry + clean plastic.")


if __name__ == "__main__":
    main()
