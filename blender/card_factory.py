"""
Card factory (bpy REQUIRED) — assembles a physical "card unit" per instance.

Phase 2 scope: the BASE card unit only — a real-scale rounded box
(63 x 88 x 0.45mm, 3mm corner radius) with three material faces:
  slot 0 = front (+Z)  -> card image
  slot 1 = back  (-Z)  -> generic back.png
  slot 2 = edges       -> mid-grey solid
Protection layers (sleeve / semi-rigid / toploader / slab) attach in later steps.

DATABLOCK SHARING STRATEGY (justification, spec §2):
  build_card_mesh() creates ONE mesh with 3 empty material slots; it can be shared
  by many card objects (linked duplicates) since geometry is identical for every
  card. Per-instance variation (the card image, later damage/holo) rides on
  OBJECT-linked material slots set in build_card_unit(), so a 12-card binder shares
  a single mesh datablock and only pays for per-object materials. Back and edge
  materials can be shared across instances too (pass shared_back_mat/shared_edge_mat).
"""
from __future__ import annotations

import math
import os
from typing import Optional, Sequence, Tuple

import bpy
import bmesh

import config

MID_GREY: Tuple[float, float, float, float] = (0.18, 0.18, 0.18, 1.0)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _rounded_rect_outline(w: float, h: float, r: float, seg: int = 10):
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
    dedup = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - dedup[-1][0]) > 1e-9 or abs(p[1] - dedup[-1][1]) > 1e-9:
            dedup.append(p)
    return dedup


def build_card_mesh(name: str = "CardUnit", seg: int = 10):
    """Build the shared rounded-box card mesh with 3 empty material slots.

    Face material_index: front(+Z)=0, back(-Z)=1, sides=2. UVs map front/back to
    the full [0,1] card rectangle (back's U flipped so its texture isn't mirrored
    when viewed from behind). Returns a bpy mesh datablock (no materials assigned).
    """
    W, H, T, R = config.CARD_W_M, config.CARD_H_M, config.CARD_T_M, config.CARD_CORNER_RADIUS_M
    outline = _rounded_rect_outline(W, H, R, seg)
    n = len(outline)

    bm = bmesh.new()
    top = [bm.verts.new((x, y, +T / 2)) for (x, y) in outline]
    bot = [bm.verts.new((x, y, -T / 2)) for (x, y) in outline]
    uv = bm.loops.layers.uv.new("UVMap")

    front = bm.faces.new(top)
    back = bm.faces.new(list(reversed(bot)))
    sides = []
    for i in range(n):
        j = (i + 1) % n
        sides.append(bm.faces.new((top[i], top[j], bot[j], bot[i])))

    # Make all normals point outward (front->+Z, back->-Z, sides->outward).
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))

    front.material_index = 0
    back.material_index = 1
    for f in sides:
        f.material_index = 2

    for loop in front.loops:
        x, y, _ = loop.vert.co
        loop[uv].uv = ((x + W / 2) / W, (y + H / 2) / H)
    for loop in back.loops:
        x, y, _ = loop.vert.co
        loop[uv].uv = (1.0 - (x + W / 2) / W, (y + H / 2) / H)

    mesh = bpy.data.meshes.new(name + "_mesh")
    bm.to_mesh(mesh)
    bm.free()
    # Establish 3 material slots so face material_index 0/1/2 are valid.
    for _ in range(3):
        mesh.materials.append(None)
    return mesh


# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #
def make_image_material(name: str, image_path: Optional[str],
                        roughness: float = 0.35, coat: float = 0.3,
                        fallback_color=(0.5, 0.5, 0.5, 1.0)):
    """Glossy clearcoat-style 'Normal' finish; Base Color from an image if given."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = 0.05
    if image_path and os.path.isfile(image_path):
        tex = nt.nodes.new("ShaderNodeTexImage")
        try:
            tex.image = bpy.data.images.load(image_path, check_existing=True)
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        except Exception as exc:  # noqa: BLE001
            print(f"[card_factory] failed to load {image_path}: {exc}")
            bsdf.inputs["Base Color"].default_value = fallback_color
    else:
        bsdf.inputs["Base Color"].default_value = fallback_color
    return mat


def make_solid_material(name: str, color=MID_GREY, roughness: float = 0.6):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_card_unit(
    name: str,
    card_id: str,
    front_image_path: Optional[str] = None,
    back_image_path: Optional[str] = None,
    edge_color=MID_GREY,
    shared_mesh=None,
    shared_back_mat=None,
    shared_edge_mat=None,
    link: bool = True,
):
    """Create one card-unit object. If shared_mesh is given, reuse it (linked
    duplicate) and set OBJECT-linked material slots so this instance can carry its
    own front image while sharing geometry.
    """
    mesh = shared_mesh if shared_mesh is not None else build_card_mesh(name)
    obj = bpy.data.objects.new(name, mesh)
    obj["card_id"] = card_id
    if link:
        bpy.context.collection.objects.link(obj)

    front_mat = make_image_material(name + "_front", front_image_path)
    back_mat = shared_back_mat or make_image_material(name + "_back", back_image_path,
                                                      roughness=0.5, coat=0.1)
    edge_mat = shared_edge_mat or make_solid_material("CardEdge", edge_color)

    # Per-object slots (so a shared mesh can host per-instance materials).
    for i, m in enumerate((front_mat, back_mat, edge_mat)):
        slot = obj.material_slots[i]
        slot.link = "OBJECT"
        slot.material = m

    return obj
