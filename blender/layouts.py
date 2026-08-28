"""
Layout builders (bpy REQUIRED). Phase 4. Each takes a validated SceneConfig +
card library + rng and populates the scene with assembled card instances plus the
layout's background/props. Returns the list of CardInstance for labeling.

Started with TABLE (spec §3.5.2): cards on a surface, random flat clutter rectangles,
a background plane with a random noisy material behind everything.
"""
from __future__ import annotations

import math
import os
from typing import List

import bpy
import bmesh

import config
from blender import scene_builder as sb
from blender import protection as prot
from blender import hand as hand_asset
from labeltools.refraction import (BOUNDS_MAX_PROPERTY, BOUNDS_MIN_PROPERTY,
                                   IOR_PROPERTY)
from rules import combinations as C

_ASSETS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets"))
_GRID_JITTER_M = 0.002
_GRID_JITTER_ROT_RAD = math.radians(1.0)


def _plane(name: str, sx: float, sy: float, z: float = 0.0):
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    verts = [bm.verts.new((x, y, z)) for x, y in
             [(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)]]
    f = bm.faces.new(verts)
    for loop in f.loops:
        x, y, _ = loop.vert.co
        loop[uv].uv = ((x + sx / 2) / sx, (y + sy / 2) / sy)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _noisy_material(name: str, rng):
    """A random colored-noise material (spec §3.5: background/clutter)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = float(rng.uniform(0.5, 1.0))
    c1 = tuple(float(v) for v in rng.uniform(0.05, 0.7, 3))
    c2 = tuple(float(v) for v in rng.uniform(0.05, 0.7, 3))
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = float(rng.uniform(3.0, 18.0))
    tex.inputs["Detail"].default_value = 4.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*c1, 1.0)
    ramp.color_ramp.elements[1].color = (*c2, 1.0)
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _table_material(name: str, rng, texture_paths=()):
    """Photographic table material, or the legacy procedural fallback."""
    paths = tuple(texture_paths)
    if not paths:
        return _noisy_material(name, rng)

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = float(rng.uniform(0.45, 0.9))
    uv = nodes.new("ShaderNodeTexCoord")

    if float(rng.random()) < 0.5:
        selected = [paths[int(rng.integers(0, len(paths)))]]
        image = nodes.new("ShaderNodeTexImage")
        image.image = bpy.data.images.load(selected[0], check_existing=True)
        links.new(uv.outputs["UV"], image.inputs["Vector"])
        links.new(image.outputs["Color"], bsdf.inputs["Base Color"])
        mode = "single"
    else:
        indices = rng.choice(len(paths), size=4, replace=len(paths) < 4)
        selected = [paths[int(index)] for index in indices]
        colors = []
        tile_span = 0.525
        tile_scale = 1.0 / tile_span
        for index, path in enumerate(selected):
            col, row = index % 2, index // 2
            scale = nodes.new("ShaderNodeVectorMath")
            scale.operation = "MULTIPLY"
            scale.inputs[1].default_value = (tile_scale, tile_scale, 1.0)
            shift = nodes.new("ShaderNodeVectorMath")
            shift.operation = "ADD"
            shift.inputs[1].default_value = (
                -0.475 * tile_scale * float(col),
                -0.475 * tile_scale * float(row), 0.0)
            image = nodes.new("ShaderNodeTexImage")
            image.image = bpy.data.images.load(path, check_existing=True)
            image.extension = "CLIP"
            links.new(uv.outputs["UV"], scale.inputs[0])
            links.new(scale.outputs["Vector"], shift.inputs[0])
            links.new(shift.outputs["Vector"], image.inputs["Vector"])
            colors.append(image.outputs["Color"])

        separate = nodes.new("ShaderNodeSeparateXYZ")
        links.new(uv.outputs["UV"], separate.inputs["Vector"])

        def seam_mask(source):
            mask = nodes.new("ShaderNodeMapRange")
            mask.interpolation_type = "SMOOTHERSTEP"
            mask.inputs["From Min"].default_value = 0.475
            mask.inputs["From Max"].default_value = 0.525
            links.new(source, mask.inputs["Value"])
            return mask.outputs["Result"]

        x_mask = seam_mask(separate.outputs["X"])
        y_mask = seam_mask(separate.outputs["Y"])
        lower = nodes.new("ShaderNodeMixRGB")
        upper = nodes.new("ShaderNodeMixRGB")
        combined = nodes.new("ShaderNodeMixRGB")
        links.new(x_mask, lower.inputs[0])
        links.new(colors[0], lower.inputs[1])
        links.new(colors[1], lower.inputs[2])
        links.new(x_mask, upper.inputs[0])
        links.new(colors[2], upper.inputs[1])
        links.new(colors[3], upper.inputs[2])
        links.new(y_mask, combined.inputs[0])
        links.new(lower.outputs[0], combined.inputs[1])
        links.new(upper.outputs[0], combined.inputs[2])
        links.new(combined.outputs[0], bsdf.inputs["Base Color"])
        mode = "quad"

    mat["tcg_table_texture_mode"] = mode
    mat["tcg_table_texture_paths"] = "|".join(selected)
    mat["tcg_table_texture_seam_overlap"] = 0.05 if mode == "quad" else 0.0
    return mat


def build_background(rng, texture_paths=(), name="Background", material_name="BgMat",
                     z: float = -0.0005):
    p = _plane(name, 3.0, 3.0, z=z)
    p.data.materials.append(_table_material(material_name, rng, texture_paths))
    return p


def _build_scene_unit(scene_cfg, name, card_cfg, card_lib, cache_dir, rng):
    if scene_cfg.cardless:
        # Match the normal path's card selection and per-instance texture seed draws
        # before constructing the same protection geometry without a card mesh.
        card_lib.select(rng)
        rng.integers(0, 2 ** 30)
        return sb.build_empty_card_unit(name, card_cfg, rng), False
    return sb.build_card_instance(name, card_cfg, card_lib.select(rng), cache_dir, rng), True


def build_clutter(rng, n: int):
    objs = []
    for i in range(int(n)):
        c = _plane(f"Clutter{i}", float(rng.uniform(0.03, 0.09)),
                   float(rng.uniform(0.03, 0.09)), z=0.0)
        c.location = (float(rng.uniform(-0.14, 0.14)), float(rng.uniform(-0.14, 0.14)),
                      0.0002 + 0.0001 * i)
        c.rotation_euler = (0.0, 0.0, float(rng.uniform(0.0, 6.283)))
        c.data.materials.append(_noisy_material(f"ClutterMat{i}", rng))
        objs.append(c)
    return objs


def build_table(scene_cfg, card_lib, cache_dir: str, rng,
                allow_overlap: bool = False, table_texture_paths=()) -> List["sb.CardInstance"]:
    """Table layout: cards laid flat (face up) on a cluttered surface. If
    `allow_overlap`, cards may overlap up to ~15% (tighter); else fully spaced."""
    build_background(rng, table_texture_paths)
    build_clutter(rng, scene_cfg.layout.params.get("clutter_rects", 3))

    instances = []
    n = len(scene_cfg.cards)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    # Grid spacing = largest protection footprint (diagonal, so rotation never clips).
    # allow_overlap tightens it to ~15% overlap; otherwise a small gap (no overlap).
    diags = [math.hypot(*sb.protection_footprint(c.protection)) for c in scene_cfg.cards]
    base = max(diags) if diags else 0.1
    spacing = base * 0.85 + 0.004 if allow_overlap else base + 0.006
    jit = 0.012 if allow_overlap else 0.006
    for i, ccfg in enumerate(scene_cfg.cards):
        inst, has_card = _build_scene_unit(
            scene_cfg, f"Card{i}", ccfg, card_lib, cache_dir, rng)
        gx, gy = i % cols, i // cols
        x = (gx - (cols - 1) / 2.0) * spacing + float(rng.uniform(-jit, jit))
        y = (gy - (rows - 1) / 2.0) * spacing + float(rng.uniform(-jit, jit))
        # Rest ON the table: lift by half thickness (a slab is 6.7mm thick); tiny
        # per-index step only to avoid z-fighting where footprints just touch.
        z = sb.protection_half_thickness(ccfg.protection) + 0.0003 * i
        rx = math.pi if ccfg.back_to_camera else 0.0   # flipped -> back up, not labeled
        inst.root.location = (x, y, z)
        inst.root.rotation_euler = (rx, 0.0, float(rng.uniform(0.0, 6.283)))
        if has_card:
            instances.append(inst)
    return instances


# --------------------------------------------------------------------------- #
# Floating (spec §3.5.5): cards floating in space; random textured rectangular
# prisms and cylinders scattered in the background.
# --------------------------------------------------------------------------- #
def _bg_prop(rng, i: int):
    """A random textured prism (cube) or cylinder placed behind the cards."""
    if rng.random() < 0.5:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        o = bpy.context.active_object
        o.scale = (float(rng.uniform(0.02, 0.09)), float(rng.uniform(0.02, 0.09)),
                   float(rng.uniform(0.02, 0.14)))
    else:
        bpy.ops.mesh.primitive_cylinder_add(radius=float(rng.uniform(0.015, 0.05)),
                                            depth=float(rng.uniform(0.05, 0.16)), vertices=24)
        o = bpy.context.active_object
    o.name = f"Prop{i}"
    o.location = (float(rng.uniform(-0.26, 0.26)), float(rng.uniform(-0.26, 0.26)),
                  float(rng.uniform(-0.36, -0.12)))
    o.rotation_euler = tuple(float(rng.uniform(0.0, 6.283)) for _ in range(3))
    o.data.materials.append(_noisy_material(f"PropM{i}", rng))
    return o


def _solid_material(name: str, rng, color=None, roughness: float = 0.6):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    c = color if color is not None else tuple(float(v) for v in rng.uniform(0.08, 0.65, 3))
    bsdf.inputs["Base Color"].default_value = (c[0], c[1], c[2], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def _weld_material(name: str):
    """Frosted semi-transparent seam where the two page layers are welded together."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.85, 0.85, 0.88, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Alpha"].default_value = 0.6
    return mat


def _empty(name: str, loc):
    e = bpy.data.objects.new(name, None)
    e.location = loc
    bpy.context.collection.objects.link(e)
    return e


def _box(name: str, sx: float, sy: float, sz: float):
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    o = bpy.context.active_object
    o.name = name
    o.scale = (sx, sy, sz)
    return o


def _apply_grid_jitter(instance, rng) -> None:
    """Break up rigid grids by moving and rotating the complete protected unit."""
    dx = float(rng.uniform(-_GRID_JITTER_M, _GRID_JITTER_M))
    dy = float(rng.uniform(-_GRID_JITTER_M, _GRID_JITTER_M))
    rz = float(rng.uniform(-_GRID_JITTER_ROT_RAD, _GRID_JITTER_ROT_RAD))
    instance.root.location.x += dx
    instance.root.location.y += dy
    instance.root.rotation_euler.z += rz
    instance.root["tcg_grid_jitter_x_mm"] = dx / 0.001
    instance.root["tcg_grid_jitter_y_mm"] = dy / 0.001
    instance.root["tcg_grid_jitter_rotation_deg"] = math.degrees(rz)


# --------------------------------------------------------------------------- #
# Binder (spec §3.5.1): two-layer pages (clear front over cards, colored/clear back)
# in a grid, on a hard-cover board with a spine; one or two offset pages.
# --------------------------------------------------------------------------- #
# Per content type: slot (w,h), content thickness, and page front-sheet gap (m).
_BINDER_CONTENT = {
    "sleeved":   {"size": (0.067, 0.092), "thick": 0.0018, "gap": 0.003},
    "toploader": {"size": (0.074, 0.098), "thick": 0.0022, "gap": 0.004},
    "slab":      {"size": (0.080, 0.135), "thick": 0.0067, "gap": 0.009},
}


def _build_binder_page(pivot, pivot_x, pcx, page_w, page_h, pad, cw, ch, gap, rows, cols,
                        cg, ct, page_color, warp, rng, cards, filled, cache_dir, card_lib,
                        instances, tag, scene_cfg):
    """Build one page (back + welded slot grid + cards + clear front) on a half,
    parented to that half's `pivot`. World x = pivot_x + local x."""
    def local(wx):
        return wx - pivot_x

    back = _plane(f"PageBack_{tag}", page_w, page_h)
    back.location = (local(pcx), 0.0, -0.0006)
    back.parent = pivot
    if page_color == "solid":
        back.data.materials.append(_solid_material(f"PageBack_{tag}", rng, roughness=0.5))
    else:
        back.data.materials.append(prot.make_clear_plastic(f"PageBackClr_{tag}", warp,
                                                           warp_strength=0.5))

    # Welded seams at the midpoints between slots (+ page edges) => separate slots.
    # Anti-overfitting measures (per user):
    #  0. 50% of pages have NO dividers at all, so their presence isn't a constant.
    #  1. Bars are inset by `pad` from the page rim so they stop at the card-grid
    #     boundary instead of running to the very edge. This breaks the continuous
    #     border rectangle and the +-shaped crossings at the page corners/edges,
    #     which would otherwise be a reliable landmark for the model to latch onto.
    #  2. ~25% of the divider lines are dropped at random so the grid of seams is
    #     not a dependable cue (Bernoulli(0.25) per line, from the scene rng).
    if float(rng.random()) < 0.5:         # this page keeps its slot dividers
        wm = _weld_material(f"Weld_{tag}")
        lw = 0.0015
        z0, z1 = -0.0006, cg
        wz, wh = (z0 + z1) / 2.0, (z1 - z0)
        edge_inset = pad                      # keep bar ends off the page rim
        v_len = max(lw, page_h - 2 * edge_inset)   # vertical bars: shortened height
        h_len = max(lw, page_w - 2 * edge_inset)   # horizontal bars: shortened width
        drop_p = 0.25                         # fraction of divider lines to omit
        vxs = ([-page_w / 2 + pad / 2]
               + [-page_w / 2 + pad + c * (cw + gap) + cw + gap / 2 for c in range(cols - 1)]
               + [page_w / 2 - pad / 2])
        for j, vx in enumerate(vxs):
            if float(rng.random()) < drop_p:
                continue
            b = _box(f"WeldV_{tag}_{j}", lw, v_len, wh)
            b.location = (local(pcx + vx), 0.0, wz)
            b.parent = pivot
            b.data.materials.append(wm)
        hys = ([page_h / 2 - pad / 2]
               + [page_h / 2 - pad - r * (ch + gap) - ch - gap / 2 for r in range(rows - 1)]
               + [-page_h / 2 + pad / 2])
        for j, hy in enumerate(hys):
            if float(rng.random()) < drop_p:
                continue
            b = _box(f"WeldH_{tag}_{j}", h_len, lw, wh)
            b.location = (local(pcx), hy, wz)
            b.parent = pivot
            b.data.materials.append(wm)

    for card_cfg, slot in zip(cards, filled):
        r, c = slot // cols, slot % cols
        sx = -page_w / 2 + pad + c * (cw + gap) + cw / 2
        sy = page_h / 2 - pad - r * (ch + gap) - ch / 2
        inst, has_card = _build_scene_unit(
            scene_cfg, f"Card{slot}", card_cfg, card_lib, cache_dir, rng)
        inst.root.location = (local(pcx + sx), sy, ct / 2 + 0.0004)
        inst.root.rotation_euler = (math.pi if card_cfg.back_to_camera else 0.0, 0.0, 0.0)
        _apply_grid_jitter(inst, rng)
        inst.root.parent = pivot
        if has_card:
            instances.append(inst)

    front = _plane(f"PageFront_{tag}", page_w, page_h)
    front.location = (local(pcx), 0.0, cg)
    front.parent = pivot
    front.data.materials.append(prot.make_clear_plastic(f"PageFront_{tag}", warp,
                                                        warp_strength=0.5))


def build_binder(scene_cfg, card_lib, cache_dir: str, rng, table_texture_paths=(), **_ignored):
    """Binder layout: centered 30mm spine with two cover halves that tilt inward up to
    10deg; the content page (welded slot grid) sits on the configured side; a reused
    table backdrop fills the background. Returns (instances, frame_extent_m)."""
    p = scene_cfg.layout.params
    # Grid string is WIDTH x HEIGHT: "4x3" => 4 cols wide, 3 rows tall.
    cols, rows = (int(x) for x in str(p.get("grid", "3x3")).split("x"))
    content = _BINDER_CONTENT.get(p.get("content_type", "sleeved"), _BINDER_CONTENT["sleeved"])
    cw, ch = content["size"]
    ct, cg = content["thick"], content["gap"]
    gap = float(p.get("slot_gap_mm", 12)) / 1000.0
    pad = float(rng.uniform(0.007, 0.018))
    page_w = cols * cw + (cols - 1) * gap + 2 * pad
    page_h = rows * ch + (rows - 1) * gap + 2 * pad
    warp = os.path.join(_ASSETS, "plastic_warp_0.png")
    page_color = p.get("page_color", "clear")
    margin = float(rng.uniform(0.010, 0.020))
    spine_w = 0.030
    half_w = page_w + 2 * margin
    board_h = page_h + 2 * margin
    side = p.get("side", "right")
    tilt = math.radians(float(rng.uniform(0.0, 10.0)))   # inward tilt of the halves

    # Reuse the table (background plane, no clutter) so the bg isn't a grey void.
    build_background(
        rng, table_texture_paths, name="Table", material_name="TableMat", z=-0.11)

    # Centered spine.
    spine = _box("Spine", spine_w, board_h, 0.012)
    spine.location = (0.0, 0.0, -0.006)
    spine.data.materials.append(_solid_material("SpineMat", rng, roughness=0.5))

    board_color = tuple(float(v) for v in rng.uniform(0.08, 0.5, 3))
    filled = list(p.get("filled_slots", list(range(len(scene_cfg.cards)))))
    instances = []
    for sign, this_side in ((-1, "left"), (1, "right")):
        pivot_x = sign * spine_w / 2.0
        pivot = _empty(f"Half_{this_side}", (pivot_x, 0.0, 0.0))
        board_cx = sign * (spine_w / 2.0 + half_w / 2.0)
        board = _plane(f"Board_{this_side}", half_w, board_h)
        board.location = (board_cx - pivot_x, 0.0, -0.005)
        board.parent = pivot
        board.data.materials.append(_solid_material(f"Board_{this_side}", rng,
                                                    color=board_color, roughness=0.5))
        if this_side == side:
            _build_binder_page(pivot, pivot_x, board_cx, page_w, page_h, pad, cw, ch, gap,
                               rows, cols, cg, ct, page_color, warp, rng, scene_cfg.cards,
                               filled, cache_dir, card_lib, instances, this_side, scene_cfg)
        pivot.rotation_euler = (0.0, -sign * tilt, 0.0)   # tilt both halves inward

    extent = max(2.0 * half_w + spine_w, board_h)
    return instances, extent


def scatter_reflectors(rng, center, spread: float = 0.45, n: int = 9):
    """Scatter colorful prisms/cylinders around `center` (e.g. BEHIND the camera) so
    they show up as reflections in glossy plastic (binder pages / sleeves)."""
    cx, cy, cz = center
    for i in range(int(n)):
        if rng.random() < 0.5:
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            o = bpy.context.active_object
            o.scale = tuple(float(rng.uniform(0.03, 0.14)) for _ in range(3))
        else:
            bpy.ops.mesh.primitive_cylinder_add(radius=float(rng.uniform(0.02, 0.07)),
                                                depth=float(rng.uniform(0.06, 0.2)), vertices=24)
            o = bpy.context.active_object
        o.name = f"Reflector{i}"
        o.location = (cx + float(rng.uniform(-spread, spread)),
                      cy + float(rng.uniform(-spread, spread)),
                      cz + float(rng.uniform(-spread * 0.6, spread * 0.6)))
        o.rotation_euler = tuple(float(rng.uniform(0.0, 6.283)) for _ in range(3))
        # Bright saturated colors so the reflections are obvious.
        col = tuple(float(v) for v in rng.uniform(0.2, 1.0, 3))
        o.data.materials.append(_solid_material(f"ReflMat{i}", rng, color=col, roughness=0.35))



def build_floating(scene_cfg, card_lib, cache_dir: str, rng,
                   allow_overlap: bool = False, max_shapes: int = 10) -> List["sb.CardInstance"]:
    """Floating layout: cards at varied depths/orientations, scattered props behind.
    `max_shapes` caps the number of background prisms/cylinders."""
    bg = _plane("Background", 3.0, 3.0, z=-0.45)
    bg.data.materials.append(_noisy_material("BgMat", rng))
    n_shapes = int(rng.integers(max(1, max_shapes // 2), max(2, max_shapes) + 1))
    for i in range(n_shapes):
        _bg_prop(rng, i)

    instances = []
    n = len(scene_cfg.cards)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    diags = [math.hypot(*sb.protection_footprint(c.protection)) for c in scene_cfg.cards]
    base = max(diags) if diags else 0.1
    spacing = base * 0.8 + 0.004 if allow_overlap else base + 0.004
    jit = 0.02
    for i, ccfg in enumerate(scene_cfg.cards):
        inst, has_card = _build_scene_unit(
            scene_cfg, f"Card{i}", ccfg, card_lib, cache_dir, rng)
        gx, gy = i % cols, i // cols
        x = (gx - (cols - 1) / 2.0) * spacing + float(rng.uniform(-jit, jit))
        y = (gy - (rows - 1) / 2.0) * spacing + float(rng.uniform(-jit, jit))
        z = float(rng.uniform(-0.10, 0.04))                       # depth variation
        # Mostly face the camera (+Z) with tilt variety; flip if back_to_camera.
        rx, ry = float(rng.uniform(-0.8, 0.8)), float(rng.uniform(-0.8, 0.8))
        rz = float(rng.uniform(0.0, 6.283))
        if ccfg.back_to_camera:
            ry += math.pi
        inst.root.location = (x, y, z)
        inst.root.rotation_euler = (rx, ry, rz)
        if has_card:
            instances.append(inst)
    return instances


# --------------------------------------------------------------------------- #
# Display case (spec §3.5.3): random-material base; a TIGHT grid of toploadered
# or slabbed cards, all flat or all tilted forward 25deg; a scratched/smudged
# clear cover a fixed headroom above the tallest item.
# --------------------------------------------------------------------------- #
_CASE_HEADROOM = 0.040   # cover sits 40mm above the top of the tallest (rotated) item


def build_display_case(scene_cfg, card_lib, cache_dir: str, rng, enabled_options=None,
                       table_texture_paths=(), **_ignored):
    """Display case: cards in a tight aligned grid on a random-material base with
    four side walls (same material) rising to a scratched/smudged 6mm acrylic cover
    that clears the tallest item by _CASE_HEADROOM (40mm); cards all flat or all
    tilted forward 25deg. Content is toploaders or slabs (enforced by rules). Grid
    capped at 24 cards upstream. Returns (instances, frame_extent_m)."""
    p = scene_cfg.layout.params
    cards = scene_cfg.cards
    n = len(cards)
    cols = max(1, int(p.get("cols", 4)))
    rows = max(1, (n + cols - 1) // cols)   # effective rows (robust if cards trimmed)
    tilt = math.radians(float(p.get("tilt_deg", 0.0))) if p.get("tilt_forward") else 0.0

    # Tight aligned grid: every card shares orientation (no random spin), so pack by
    # the actual footprint. Forward tilt shrinks the projected y-footprint by cos(tilt).
    foot = [sb.protection_footprint(c.protection) for c in cards]
    fw = max((w for w, _ in foot), default=0.075)
    fh = max((h for _, h in foot), default=0.10)
    gap = 0.004
    sx = fw + gap
    sy = fh * math.cos(tilt) + gap
    grid_w, grid_h = cols * sx, rows * sy

    case_w, case_h = grid_w + 0.05, grid_h + 0.05

    # Random-material base under the cards (spec: random color/material base). The
    # four side walls reuse the SAME material and connect the base up to the cover.
    if rng.random() < 0.5:
        case_mat = _solid_material("CaseBaseMat", rng, roughness=0.5)
    else:
        case_mat = _noisy_material("CaseBaseMat", rng)
    base = _plane("CaseBase", case_w, case_h, z=0.0)
    base.data.materials.append(case_mat)

    eps = 0.0003
    instances = []
    top_z = 0.0                                    # tallest item's top after rotation
    for i, ccfg in enumerate(cards):
        gx, gy = i % cols, i // cols
        cx = (gx - (cols - 1) / 2.0) * sx
        cy = ((rows - 1) / 2.0 - gy) * sy         # fill the top row first
        fh_i = sb.protection_footprint(ccfg.protection)[1]
        ht_i = sb.protection_half_thickness(ccfg.protection)
        inst, has_card = _build_scene_unit(
            scene_cfg, f"Card{i}", ccfg, card_lib, cache_dir, rng)
        # Stand the card UP by hinging about its BOTTOM (-Y, art-bottom) edge, which
        # rests on the base: the bottom edge stays planted while the top (+Y, art-top)
        # edge lifts toward the camera. This keeps the art right-side-up and stops the
        # top edge from rotating DOWN through the base (the old center-pivot -tilt bug).
        # Hinge rotation = center rotation (rx=+tilt) + compensating translation so the
        # hinge point maps to itself. Center is at +fh/2 (Y) and +ht (Z) from the hinge.
        rx = tilt
        half = (fh_i / 2.0) * math.sin(tilt) + ht_i * math.cos(tilt)
        loc_y = cy - (fh_i / 2.0) * (1.0 - math.cos(tilt)) - ht_i * math.sin(tilt)
        loc_z = half + eps
        if ccfg.back_to_camera:                   # display_case never sets this
            rx += math.pi
        inst.root.location = (cx, loc_y, loc_z)
        inst.root.rotation_euler = (rx, 0.0, 0.0)
        _apply_grid_jitter(inst, rng)
        if has_card:
            instances.append(inst)
        top_z = max(top_z, loc_z + half)          # top-front corner world z

    # Case interior height = tallest item's top + headroom. Side walls (same material
    # as the base) rise from z=0 to that height; the cover rests on them. Walls are
    # inset so their outer faces sit flush with the base edge.
    case_height = top_z + _CASE_HEADROOM
    wall_t = 0.004
    for nm, sxw, syw, wx, wy in (
        ("CaseWallL", wall_t, case_h, -(case_w / 2 - wall_t / 2), 0.0),
        ("CaseWallR", wall_t, case_h,  (case_w / 2 - wall_t / 2), 0.0),
        ("CaseWallF", case_w, wall_t, 0.0, -(case_h / 2 - wall_t / 2)),
        ("CaseWallB", case_w, wall_t, 0.0,  (case_h / 2 - wall_t / 2)),
    ):
        wall = _box(nm, sxw, syw, case_height)
        wall.location = (wx, wy, case_height / 2.0)
        wall.data.materials.append(case_mat)

    # Scratched/smudged cover: a REAL 6mm-thick acrylic box (not a plane) resting on
    # the walls. Uses make_slab_surface (real transmission + front/back faces, so a flat
    # parallel lid refracts cleanly), tinted ever so slightly. It reads as CLEAR glass:
    # low base roughness (small wear span) so you see through it, with only the fine
    # scratch map catching light. UV maps the near-FULL 2048 wear texture (win~1.0) so
    # the hairline scratches stay small and numerous on the large lid, not magnified.
    cover_t = 0.006
    warp = os.path.join(_ASSETS, f"plastic_warp_{int(rng.integers(0, 6))}.png")
    wear = os.path.join(_ASSETS, f"case_cover_wear_{int(rng.integers(0, 6))}.png")
    tint = (float(rng.uniform(0.86, 0.93)), float(rng.uniform(0.90, 0.96)),
            float(rng.uniform(0.94, 0.99)))          # faint cool tint, not perfectly clear
    cover = _box("CaseCover", case_w, case_h, cover_t)
    # _box uses a unit cube with object scale carrying the requested dimensions.
    cover[IOR_PROPERTY] = 1.5
    cover[BOUNDS_MIN_PROPERTY] = (-0.5, -0.5, -0.5)
    cover[BOUNDS_MAX_PROPERTY] = (+0.5, +0.5, +0.5)
    cover_top = case_height + cover_t
    cover.location = (0.0, 0.0, case_height + cover_t / 2.0)
    cover.data.materials.append(prot.make_slab_surface(
        "CaseCoverMat", warp, wear, base_rough=0.02, wear_rough=0.18, tint=tint,
        uv_xform=prot.random_uv_xform(rng, win_range=(0.9, 1.0)),
        wear_uv_xform=prot.random_uv_xform(rng, win_range=(0.9, 1.0))))

    # By default, a 20% chance adds a stray card lying flat ON TOP of the case lid
    # (any of bare/sleeved/toploadered/slabbed, per rules.sample_top_card). Acceptance
    # scenes can override the probability to exercise this geometry every time. Its
    # CENTER can be anywhere within the top face of the case, so it may hang partly off
    # an edge but is always on the display.
    if float(rng.random()) < float(p.get("top_card_probability", 0.20)):
        top_cfg = C.sample_top_card(rng, enabled_options)
        ht_t = sb.protection_half_thickness(top_cfg.protection)
        inst, has_card = _build_scene_unit(
            scene_cfg, "TopCard", top_cfg, card_lib, cache_dir, rng)
        inst.root.location = (float(rng.uniform(-case_w / 2.0, case_w / 2.0)),
                              float(rng.uniform(-case_h / 2.0, case_h / 2.0)),
                              cover_top + ht_t + eps)
        inst.root.rotation_euler = (0.0, 0.0, float(rng.uniform(0.0, 2.0 * math.pi)))
        if has_card:
            instances.append(inst)

    # Table the whole case sits on (reused from the binder scene): a large noisy plane
    # just BELOW the case base so the background isn't a void. Built LAST so its rng
    # draw doesn't shift the earlier base/grid/cover/top-card randomness.
    build_background(
        rng, table_texture_paths, name="CaseTable", material_name="CaseTableMat", z=-0.003)

    extent = max(grid_w, grid_h) + 0.05
    return instances, extent


# --------------------------------------------------------------------------- #
# Hand (spec section 3.5.4): one front-facing bare/sleeved/toploadered card held
# in a side or pinch grip above the same noisy table used by other Phase 4 scenes.
# --------------------------------------------------------------------------- #
def build_hand(scene_cfg, card_lib, cache_dir: str, rng, table_texture_paths=(), **_ignored):
    """Build one card and a seeded left/right hand grip.

    Returns ``(instances, frame_extent_m)``. Hands are deliberately not label
    occluders: hand-held cards retain their original four-corner labels.
    """
    params = scene_cfg.layout.params
    card_cfg = scene_cfg.cards[0]

    table = build_background(rng, table_texture_paths)
    table.name = "HandTable"
    table.location.z = -0.12
    table.data.materials[0].name = "HandTableMat"

    instance, has_card = _build_scene_unit(
        scene_cfg, "Card0", card_cfg, card_lib, cache_dir, rng)
    instance.root.location = (0.0, 0.0, 0.0)
    instance.root.rotation_euler = (0.0, 0.0, 0.0)

    hand_asset.build_hand(
        "GripHand",
        params["handedness"],
        params["grip"],
        sb.protection_footprint(card_cfg.protection),
        float(params["approach_deg"]),
        float(params["depth"]),
        protection_half_thickness=sb.protection_half_thickness(card_cfg.protection),
        card_z=0.0,
        rng=rng,
    )
    return ([instance] if has_card else []), 0.30


# --------------------------------------------------------------------------- #
# Stack: 1-10 uniformly protected cards, almost touching vertically over a table.
# Only the top card is eligible for labels and label occlusion; lower cards remain
# render-only stack geometry.
# --------------------------------------------------------------------------- #
def build_stack(scene_cfg, card_lib, cache_dir: str, rng, table_texture_paths=(), **_ignored):
    build_background(
        rng, table_texture_paths, name="StackTable", material_name="StackTableMat", z=-0.00001)
    configs = scene_cfg.cards
    footprint = sb.protection_footprint(configs[-1].protection)
    thickness = sb.stack_thickness(configs[-1].protection)
    half_thickness = thickness / 2.0
    step_z = thickness + 0.0001
    clearance = float(scene_cfg.layout.params["table_clearance_m"])
    offsets = scene_cfg.layout.params["offsets"]
    instances = []
    top_root = None
    for index, (card_cfg, offset) in enumerate(zip(configs, offsets)):
        unit, has_card = _build_scene_unit(
            scene_cfg, f"Card{index}", card_cfg, card_lib, cache_dir, rng)
        unit.root.location = (
            float(offset["x_frac"]) * config.CARD_W_M,
            float(offset["y_frac"]) * config.CARD_H_M,
            clearance + half_thickness + index * step_z,
        )
        unit.root.rotation_euler = (
            0.0, 0.0, math.radians(float(offset["rotation_deg"])))
        top_root = unit.root
        if has_card:
            unit.label_enabled = index == len(configs) - 1
            unit.root["tcg_label_enabled"] = unit.label_enabled
            instances.append(unit)

    if scene_cfg.layout.params.get("with_hand"):
        params = scene_cfg.layout.params["hand"]
        hand_asset.build_hand(
            "StackHand", params["handedness"], params["grip"], footprint,
            float(params["approach_deg"]), float(params["depth"]),
            protection_half_thickness=half_thickness,
            card_z=float(top_root.location.z), rng=rng)

    extent = max(footprint) * 1.3
    if scene_cfg.layout.params.get("with_hand"):
        extent = max(extent, 0.30)
    return instances, extent
