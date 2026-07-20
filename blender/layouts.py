"""
Layout builders (bpy REQUIRED). Phase 4. Each takes a validated SceneConfig +
card library + rng and populates the scene with assembled card instances plus the
layout's background/props. Returns the list of CardInstance for labeling.

Started with TABLE (spec §3.5.2): cards on a surface, random flat clutter rectangles,
a background plane with a random noisy material behind everything.
"""
from __future__ import annotations

import math
from typing import List

import bpy
import bmesh

from blender import scene_builder as sb


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


def build_background(rng):
    p = _plane("Background", 3.0, 3.0, z=-0.0005)
    p.data.materials.append(_noisy_material("BgMat", rng))
    return p


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
                allow_overlap: bool = False) -> List["sb.CardInstance"]:
    """Table layout: cards laid flat (face up) on a cluttered surface. If
    `allow_overlap`, cards may overlap up to ~15% (tighter); else fully spaced."""
    build_background(rng)
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
        card_img = card_lib.select(rng)
        inst = sb.build_card_instance(f"Card{i}", ccfg, card_img, cache_dir, rng)
        gx, gy = i % cols, i // cols
        x = (gx - (cols - 1) / 2.0) * spacing + float(rng.uniform(-jit, jit))
        y = (gy - (rows - 1) / 2.0) * spacing + float(rng.uniform(-jit, jit))
        # Rest ON the table: lift by half thickness (a slab is 6.7mm thick); tiny
        # per-index step only to avoid z-fighting where footprints just touch.
        z = sb.protection_half_thickness(ccfg.protection) + 0.0003 * i
        rx = math.pi if ccfg.back_to_camera else 0.0   # flipped -> back up, not labeled
        inst.root.location = (x, y, z)
        inst.root.rotation_euler = (rx, 0.0, float(rng.uniform(0.0, 6.283)))
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
        card_img = card_lib.select(rng)
        inst = sb.build_card_instance(f"Card{i}", ccfg, card_img, cache_dir, rng)
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
        instances.append(inst)
    return instances
