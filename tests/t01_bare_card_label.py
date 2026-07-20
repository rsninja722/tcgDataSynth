"""
Phase 1 - One bare card + labels (end-to-end labeling skeleton).

WHY THIS EXISTS
    This validates the ENTIRE labeling math before anything else is built:
    rounded-corner card mesh -> real-scale placement -> corner projection with
    world_to_camera_view -> Y-flip to top-left origin -> frustum containment ->
    front-face-visible rule -> YOLO-pose label line. If the drawn corners sit on
    the card's true (un-rounded) rectangle corners across camera angles/focals,
    the math is correct and every later phase can trust it.

HOW TO RUN (headless, no GUI needed), from inside the tcgDataSynth folder:
    blender -b -P tests/t01_bare_card_label.py

    Windows example:
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t01_bare_card_label.py

WHAT IT PRODUCES  (all under out/)
    t01_<case>.png   - 1280x1280 render for each camera case
    t01_<case>.txt   - its YOLO-pose label (empty file when the card is not labeled)
    Console: per-case, the 4 projected corners (x,y,z), in/out-of-frustum, and
             whether the card was labeled + why.

CASES
    headon_35mm, off30_35mm, off50_35mm, wide_15mm, long_55mm  -> EXPECT 1 label
    back_facing   -> EXPECT 0 labels (card's back faces the camera)
    half_out_frame-> EXPECT 0 labels (a corner leaves the frustum)

WHAT TO REPORT BACK
    1) Paste the full console output.
    2) On your machine, run the visualizer on a few cases, e.g.:
         python3 labeltools/visualize.py out/t01_headon_35mm.png out/t01_headon_35mm.txt
         python3 labeltools/visualize.py out/t01_off50_35mm.png out/t01_off50_35mm.txt
         python3 labeltools/visualize.py out/t01_half_out_frame.png out/t01_half_out_frame.txt
       Attach the resulting *_viz.png files.
    PASS if the numbered corner dots land exactly on the card's sharp rectangle
    corners (NOT on the rounded edge) in every labeled case, and the two negative
    cases show "labels: 0".
"""
import os
import sys
import math

import bpy
import bmesh
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# --- make the Docker-tested pure-Python modules importable ------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from labeltools.yolo_pose import write_label_file, write_dataset_yaml  # noqa: E402
from labeltools.frustum import classify, is_front_visible  # noqa: E402

try:
    from texturegen.cardsource import CardLibrary  # optional real card face
except Exception:  # noqa: BLE001
    CardLibrary = None

# Card geometry (meters). Front face normal = +Z, +Y = up (card's own frame).
W, H, R = config.CARD_W_M, config.CARD_H_M, config.CARD_CORNER_RADIUS_M

# The FOUR ideal (un-rounded) corner points, in KEYPOINT_ORDER = TL,TR,BR,BL,
# as local coordinates on the front face (z=0 for this flat Phase-1 card).
IDEAL_CORNERS_LOCAL = [
    Vector((-W / 2, +H / 2, 0.0)),  # 1 TL
    Vector((+W / 2, +H / 2, 0.0)),  # 2 TR
    Vector((+W / 2, -H / 2, 0.0)),  # 3 BR
    Vector((-W / 2, -H / 2, 0.0)),  # 4 BL
]


# --------------------------------------------------------------------------- #
# Scene building
# --------------------------------------------------------------------------- #
def reset_scene():
    """Remove everything so the script is idempotent across the case loop."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def rounded_rect_outline(w, h, r, seg=10):
    """CCW outline (viewed from +Z) of a w x h rectangle with radius-r corners."""
    r = min(r, w / 2, h / 2)
    cx, cy = w / 2 - r, h / 2 - r
    arcs = [
        (cx, cy, 0.0, 90.0),       # top-right
        (-cx, cy, 90.0, 180.0),    # top-left
        (-cx, -cy, 180.0, 270.0),  # bottom-left
        (cx, -cy, 270.0, 360.0),   # bottom-right
    ]
    pts = []
    for (ox, oy, a0, a1) in arcs:
        for i in range(seg + 1):
            a = math.radians(a0 + (a1 - a0) * i / seg)
            pts.append((ox + r * math.cos(a), oy + r * math.sin(a)))
    # Drop duplicate seam vertices between consecutive arcs.
    dedup = [pts[0]]
    for p in pts[1:]:
        if (abs(p[0] - dedup[-1][0]) > 1e-9) or (abs(p[1] - dedup[-1][1]) > 1e-9):
            dedup.append(p)
    return dedup


def build_card(name="Card"):
    """Build a flat rounded card in its own frame; front (+Z) carries the texture.

    Thickness is omitted in Phase 1 (0.45mm is negligible for label geometry);
    real thickness arrives with the card factory in Phase 2. The ideal corners
    are tracked analytically (IDEAL_CORNERS_LOCAL), independent of the rounding.
    """
    outline = rounded_rect_outline(W, H, R)
    bm = bmesh.new()
    verts = [bm.verts.new((x, y, 0.0)) for (x, y) in outline]
    face = bm.faces.new(verts)
    bm.normal_update()  # make face.normal valid before we test its orientation
    if face.normal.z < 0:
        face.normal_flip()
    uv = bm.loops.layers.uv.new("UVMap")
    for loop in face.loops:
        vx, vy, _ = loop.vert.co
        loop[uv].uv = ((vx + W / 2) / W, (vy + H / 2) / H)
    mesh = bpy.data.meshes.new(name + "_mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def make_card_material(image_path=None):
    """Glossy 'Normal' finish (clearcoat-style). Base color from a real card image
    if available, else a checker so corners are visually unambiguous."""
    mat = bpy.data.materials.new("CardMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Coat Weight"].default_value = 0.3        # clearcoat-style gloss
    bsdf.inputs["Coat Roughness"].default_value = 0.05
    if image_path and os.path.isfile(image_path):
        tex = nt.nodes.new("ShaderNodeTexImage")
        try:
            tex.image = bpy.data.images.load(image_path, check_existing=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[t01] could not load card image {image_path}: {exc}; using checker")
            image_path = None
    if not image_path:
        tex = nt.nodes.new("ShaderNodeTexChecker")
        tex.inputs["Scale"].default_value = 12.0
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def setup_world(gray=0.18):
    """Flat ambient world so the whole card face is evenly lit for corner checks."""
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (gray, gray, gray, 1.0)
    bg.inputs["Strength"].default_value = 1.0


def add_lights(cam_loc):
    """A low sun for fill + a point light near the camera (phone-flash analog)."""
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 2.0
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (math.radians(45), 0.0, math.radians(30))
    bpy.context.collection.objects.link(sun)

    pt_data = bpy.data.lights.new("Flash", type="POINT")
    pt_data.energy = 6.0  # watts; subtle highlight (card sits ~0.14m from cam)
    pt = bpy.data.objects.new("Flash", pt_data)
    pt.location = cam_loc
    bpy.context.collection.objects.link(pt)


def setup_camera(focal_mm, azimuth_deg, distance):
    """Place a perspective camera at `azimuth_deg` off head-on, aimed at origin."""
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = "PERSP"
    cam_data.lens = focal_mm
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    # Default near clip is 0.1m, but at macro scale cards sit as close as ~0.06m
    # (esp. wide lenses) and would be clipped away. Pull the near plane in.
    cam_data.clip_start = 0.001
    cam_data.clip_end = 100.0
    cam = bpy.data.objects.new("Cam", cam_data)
    az = math.radians(azimuth_deg)
    cam.location = Vector((math.sin(az) * distance, 0.0, math.cos(az) * distance))
    direction = (Vector((0, 0, 0)) - cam.location).normalized()
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def frame_distance(focal_mm, target_frac=0.62):
    """Distance so the card's 88mm dimension fills ~target_frac of the frame."""
    fov = 2.0 * math.atan((36.0 / 2.0) / focal_mm)  # square render, horizontal fit
    return (H / target_frac) / (2.0 * math.tan(fov / 2.0))


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = config.RENDER_ENGINE
    scene.render.resolution_x = config.RENDER_W
    scene.render.resolution_y = config.RENDER_H
    scene.render.resolution_percentage = 100
    try:
        scene.eevee.taa_render_samples = config.EEVEE_RENDER_SAMPLES
    except Exception as exc:  # noqa: BLE001
        print(f"[t01] could not set eevee samples: {exc}")
    try:
        scene.view_settings.view_transform = config.VIEW_TRANSFORM
    except Exception as exc:  # noqa: BLE001
        print(f"[t01] view_transform {config.VIEW_TRANSFORM!r} not set: {exc}")
    scene.render.image_settings.file_format = "PNG"


# --------------------------------------------------------------------------- #
# Labeling
# --------------------------------------------------------------------------- #
def project_and_label(scene, cam, card):
    """Project the 4 ideal corners with Blender, then defer the label decision to
    the Docker-tested labeltools.frustum.classify. Returns (label, debug_rows, reason)."""
    mw = card.matrix_world

    # Front-face-visible rule (user requirement): skip cards whose back faces cam.
    world_front_normal = (mw.to_3x3() @ Vector((0, 0, 1))).normalized()
    front_visible = is_front_visible(
        world_front_normal, cam.matrix_world.translation, mw.translation
    )

    ndc_corners = []
    debug_rows = []
    for i, local in enumerate(IDEAL_CORNERS_LOCAL):
        world_co = mw @ local
        ndc = world_to_camera_view(scene, cam, world_co)  # (x,y,z); (0,0)=bottom-left
        ndc_corners.append((ndc.x, ndc.y, ndc.z))
        in_f = (0.0 <= ndc.x <= 1.0) and (0.0 <= ndc.y <= 1.0) and (ndc.z > 1e-6)
        debug_rows.append((i + 1, ndc.x, ndc.y, ndc.z, in_f))

    card_id = card.get("card_id", card.name)
    label, reason = classify(card_id, ndc_corners, front_visible)
    return label, debug_rows, reason


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
CASES = [
    # name,            focal, azimuth, card_rot_euler,          card_loc
    ("headon_35mm",     35, 0,  (0, 0, 0),          (0, 0, 0)),
    ("off30_35mm",      35, 30, (0, 0, 0),          (0, 0, 0)),
    ("off50_35mm",      35, 50, (0, 0, 0),          (0, 0, 0)),
    ("wide_15mm",       15, 0,  (0, 0, 0),          (0, 0, 0)),
    ("long_55mm",       55, 0,  (0, 0, 0),          (0, 0, 0)),
    ("back_facing",     35, 0,  (0, math.pi, 0),    (0, 0, 0)),   # EXPECT 0 labels
    ("half_out_frame",  35, 0,  (0, 0, 0),          (0.055, 0, 0)),  # EXPECT 0 labels
]


def run_case(name, focal, azimuth, rot, loc, image_path):
    reset_scene()
    setup_world()
    setup_render()

    card = build_card("Card")
    card["card_id"] = os.path.splitext(os.path.basename(image_path))[0] if image_path else "testcard"
    card.data.materials.append(make_card_material(image_path))
    card.rotation_euler = rot
    card.location = loc

    dist = frame_distance(focal)
    cam = setup_camera(focal, azimuth, dist)
    add_lights(cam.location)

    scene = bpy.context.scene
    bpy.context.view_layer.update()  # ensure matrices current before projection

    label, debug_rows, reason = project_and_label(scene, cam, card)

    out_dir = os.path.join(_ROOT, config.OUTPUT.root)
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"t01_{name}.png")
    lbl_path = os.path.join(out_dir, f"t01_{name}.txt")

    scene.render.filepath = img_path
    bpy.ops.render.render(write_still=True)
    write_label_file(lbl_path, [label] if label else [])

    print(f"\n=== case {name}  focal={focal}mm azimuth={azimuth} dist={dist:.3f}m -> {reason} ===")
    for (idx, x, y, z, inf) in debug_rows:
        print(f"   corner{idx}  ndc=({x:+.4f}, {y:+.4f}, z={z:+.4f})  in_frustum={inf}")
    if label:
        print(f"   LABEL: {label.to_line()}")
    else:
        print(f"   NO LABEL ({reason}) -> empty label file written")


def main():
    lib = None
    if CardLibrary is not None:
        try:
            lib = CardLibrary()
            if lib.is_empty():
                lib = None
        except Exception as exc:  # noqa: BLE001
            print(f"[t01] card library unavailable ({exc}); using checker texture")
            lib = None

    image_path = None
    if lib is not None:
        card = lib.select(__import__("numpy").random.default_rng(7))
        image_path = card.path
        print(f"[t01] using real card face: {image_path} (id={card.card_id})")
    else:
        print("[t01] no card images found; using procedural checker texture")

    for (name, focal, azimuth, rot, loc) in CASES:
        run_case(name, focal, azimuth, rot, loc, image_path)

    # dataset.yaml alongside the labels for downstream sanity.
    write_dataset_yaml(os.path.join(_ROOT, config.OUTPUT.root, "dataset.yaml"))
    print("\n[t01] done. Expected labels: 5 cases labeled, back_facing & half_out_frame = 0.")


if __name__ == "__main__":
    main()
