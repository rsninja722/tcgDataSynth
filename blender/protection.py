"""
Protection layers (bpy REQUIRED). Phase 2 step 2: SLEEVES.

Builds trading-card sleeves as two thin plastic layers that sit slightly off the
card faces and curve inward to meet along all four edges (fully sealed envelope),
extended by a per-side margin (spec §3.2). Plastic uses a pre-baked warp normal
map (from texturegen, loaded as an image) so reflections are uneven.

(The spec's open-top variant was dropped at the user's request for simplicity.)

Types:  'clear' (both layers clear) | 'opaque_back' (opaque colored back layer,
         slightly-matte clear front layer).
Sizes:  '1mm' (+1mm all sides) | '2.5mm' (+2.5mm all sides).

Semi-rigid / toploader / slab arrive in later steps (t04/t05).
"""
from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import bpy
import bmesh

import config

_OFF = 0.00012          # layer standoff from the card surface (0.12mm; spec says 0.05mm
                        # but that razor gap z-fights the card at grazing angles).
SLEEVE_MARGINS = {"1mm": 0.001, "2.5mm": 0.0025}


# --------------------------------------------------------------------------- #
# Per-instance texture variation
# --------------------------------------------------------------------------- #
# The pre-baked warp/wear maps are FEW (shared datablocks), so without variation
# every instance would show the identical reflection/scratch pattern. Each instance
# instead samples a random cropped/zoomed sub-region (+ random flips) of the shared
# texture via a Mapping node, giving near-infinite variety from a handful of assets.
def random_uv_xform(rng):
    """Return a UV transform (loc_x, loc_y, win, flip_x, flip_y) that crops a random
    `win`-sized sub-window (30-60% of the texture) placed ENTIRELY within [0,1], so
    the sample never crosses a tile boundary -> no hard seam on non-tiling maps.
    Mapping applies result = win*uv + loc, so a `win`<1 zooms IN to a sub-region."""
    win = float(rng.uniform(0.3, 0.6))
    flip_x = bool(rng.random() < 0.5)
    flip_y = bool(rng.random() < 0.5)
    # For a flipped axis the window runs [loc-win, loc], so loc must be in [win, 1];
    # otherwise it runs [loc, loc+win], so loc in [0, 1-win]. Both stay inside [0,1].
    loc_x = float(rng.uniform(win, 1.0)) if flip_x else float(rng.uniform(0.0, 1.0 - win))
    loc_y = float(rng.uniform(win, 1.0)) if flip_y else float(rng.uniform(0.0, 1.0 - win))
    return (loc_x, loc_y, win, flip_x, flip_y)


def _apply_uv_mapping(nt, tex_node, uv_xform):
    """Insert TexCoord -> Mapping -> tex_node.Vector so this instance samples a
    random sub-window of the shared texture. The window stays within [0,1] and the
    texture is set to EXTEND, so there is never a wrapped-tile seam. Offset/scale/
    flip only (no rotation) keeps normal-map vectors valid."""
    if uv_xform is None:
        return
    loc_x, loc_y, win, flip_x, flip_y = uv_xform
    tex_node.extension = "EXTEND"   # clamp at edges as a belt-and-suspenders vs seams
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Location"].default_value = (loc_x, loc_y, 0.0)
    mapping.inputs["Scale"].default_value = (-win if flip_x else win,
                                             -win if flip_y else win, 1.0)
    nt.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], tex_node.inputs["Vector"])


# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #
def _warp_normal_socket(nt, warp_map_path: Optional[str], strength: float = 0.18,
                        uv_xform=None):
    """Build a warp-normal-map -> Normal Map node chain; return its Normal output
    socket (or None if no map). `uv_xform` randomizes which region is sampled."""
    if not warp_map_path or not os.path.isfile(warp_map_path):
        return None
    tex = nt.nodes.new("ShaderNodeTexImage")
    try:
        img = bpy.data.images.load(warp_map_path, check_existing=True)
        img.colorspace_settings.name = "Non-Color"  # normal data, not color
        tex.image = img
    except Exception as exc:  # noqa: BLE001
        print(f"[protection] warp map load failed {warp_map_path}: {exc}")
        return None
    _apply_uv_mapping(nt, tex, uv_xform)
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.inputs["Strength"].default_value = strength
    nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
    return nmap.outputs["Normal"]


def _add_warp_normal(nt, bsdf, warp_map_path: Optional[str], strength: float = 0.18,
                     uv_xform=None):
    """Wire a pre-baked plastic-warp normal map into a BSDF's Normal input."""
    sock = _warp_normal_socket(nt, warp_map_path, strength, uv_xform)
    if sock is not None:
        nt.links.new(sock, bsdf.inputs["Normal"])


def _glossy(nt, roughness_val=0.06, roughness_socket=None):
    """Glossy reflector node (fallback to Principled mirror if the id differs)."""
    try:
        g = nt.nodes.new("ShaderNodeBsdfGlossy")
    except (RuntimeError, KeyError):
        g = nt.nodes.new("ShaderNodeBsdfPrincipled")
        g.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        g.inputs["Metallic"].default_value = 1.0
    if roughness_socket is not None:
        nt.links.new(roughness_socket, g.inputs["Roughness"])
    else:
        g.inputs["Roughness"].default_value = roughness_val
    return g, g.inputs.get("Normal")


def _thin_walled_graph(nt, roughness_val=0.06, roughness_socket=None, tint=(1.0, 1.0, 1.0),
                       warp_sock=None, ior=1.5):
    """Thin-walled clear plastic (single zero-thickness sheet): NO transmission/
    refraction (which would permanently bend the camera ray and corrupt the holo
    card's dot(N,Incoming) underneath). Instead Fresnel(IOR, warp normal) mixes a
    Transparent BSDF body with a Glossy reflection (warp normal into Glossy only), and
    a Light-Path 'Is Shadow Ray' branch outputs pure Transparent so lamp light passes
    through unattenuated (no darkening). Caller clears the tree + builds warp first."""
    nodes, links = nt.nodes, nt.links
    out = nodes.new("ShaderNodeOutputMaterial")
    transp_body = nodes.new("ShaderNodeBsdfTransparent")
    transp_body.inputs["Color"].default_value = (tint[0], tint[1], tint[2], 1.0)
    glossy, gnorm = _glossy(nt, roughness_val, roughness_socket)
    fres = nodes.new("ShaderNodeFresnel")
    fres.inputs["IOR"].default_value = ior
    if warp_sock is not None:
        if gnorm is not None:
            links.new(warp_sock, gnorm)          # warp -> Glossy only
        links.new(warp_sock, fres.inputs["Normal"])
    inner = nodes.new("ShaderNodeMixShader")
    links.new(fres.outputs["Fac"], inner.inputs[0])
    links.new(transp_body.outputs["BSDF"], inner.inputs[1])
    links.new(glossy.outputs["BSDF"], inner.inputs[2])
    # Shadow-ray branch: pure Transparent so lamp light isn't attenuated.
    lp = nodes.new("ShaderNodeLightPath")
    transp_shadow = nodes.new("ShaderNodeBsdfTransparent")   # white
    outer = nodes.new("ShaderNodeMixShader")
    links.new(lp.outputs["Is Shadow Ray"], outer.inputs[0])
    links.new(inner.outputs[0], outer.inputs[1])
    links.new(transp_shadow.outputs["BSDF"], outer.inputs[2])
    links.new(outer.outputs[0], (out.inputs.get("Surface") or out.inputs[0]))


def make_clear_plastic(name: str, warp_map_path: Optional[str],
                       roughness: float = 0.06, uv_xform=None, warp_strength: float = 0.18):
    """Clear thin plastic (sleeves): thin-walled shader (no refraction -> holo card
    behind it stays correct). `roughness` 0.06 clear / 0.12 matte. `warp_strength`
    controls how warped/loose the surface looks (binder pages use a high value)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    warp = _warp_normal_socket(nt, warp_map_path, strength=warp_strength, uv_xform=uv_xform)
    _thin_walled_graph(nt, roughness_val=roughness, warp_sock=warp)
    return mat


def make_opaque_plastic(name: str, color: Sequence[float], warp_map_path: Optional[str],
                        roughness: float = 0.5, uv_xform=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    r, g, b = color[0], color[1], color[2]
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    _add_warp_normal(nt, bsdf, warp_map_path, uv_xform=uv_xform)
    return mat


# --------------------------------------------------------------------------- #
# Geometry: one plastic pocket layer (front or back)
# --------------------------------------------------------------------------- #
def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _build_layer(bm, uv, Wo, Ho, off, sign, seam_band, nx, ny, mat_index):
    """Grid layer at z=sign*off in the interior, curving to z=0 (meeting the other
    layer) within `seam_band` of ALL four edges (fully sealed envelope)."""
    V = {}
    for r in range(ny + 1):
        for c in range(nx + 1):
            x = -Wo / 2 + Wo * c / nx
            y = -Ho / 2 + Ho * r / ny
            d_edge = min(x + Wo / 2, Wo / 2 - x, y + Ho / 2, Ho / 2 - y)  # all 4 edges
            z = sign * off * _smoothstep(d_edge / seam_band)
            V[(r, c)] = bm.verts.new((x, y, z))
    for r in range(ny):
        for c in range(nx):
            a, b, cc, d = V[(r, c)], V[(r, c + 1)], V[(r + 1, c + 1)], V[(r + 1, c)]
            # winding so front layer normal -> +Z, back layer -> -Z
            f = bm.faces.new((a, b, cc, d) if sign > 0 else (a, d, cc, b))
            f.material_index = mat_index
            for loop in f.loops:
                vx, vy, _ = loop.vert.co
                loop[uv].uv = ((vx + Wo / 2) / Wo, (vy + Ho / 2) / Ho)
    return V


# --------------------------------------------------------------------------- #
# Public: build a sleeve wrapping a card
# --------------------------------------------------------------------------- #
def build_sleeve_mesh(size: str, name: str = "Sleeve"):
    """Build the card-independent sleeve pocket mesh (2 material slots: front, back)
    for a given size. Shareable across instances / pre-buildable into the asset lib."""
    if size not in SLEEVE_MARGINS:
        raise ValueError(f"bad sleeve size {size!r}; expected one of {list(SLEEVE_MARGINS)}")
    margin = SLEEVE_MARGINS[size]
    W, H, T = config.CARD_W_M, config.CARD_H_M, config.CARD_T_M
    Wo, Ho = W + 2 * margin, H + 2 * margin
    off = T / 2 + _OFF
    # Convergence band stays within the margin ring (else the layer dips below the
    # card face inside its edge -> poke-through). 0.5mm cells so a vertex lands near
    # the card boundary.
    seam_band = min(margin * 0.9, 0.0015)
    nx = max(8, int(Wo / 0.0005))
    ny = max(8, int(Ho / 0.0005))

    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    _build_layer(bm, uv, Wo, Ho, off, +1.0, seam_band, nx, ny, 0)  # front
    _build_layer(bm, uv, Wo, Ho, off, -1.0, seam_band, nx, ny, 1)  # back
    # Weld the two layers where they meet (z=0 along the sealed edges).
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-6)
    bm.normal_update()
    mesh = bpy.data.meshes.new(name + "_mesh")
    bm.to_mesh(mesh)
    bm.free()
    for _ in range(2):
        mesh.materials.append(None)
    return mesh


def build_sleeve(
    name: str,
    card_obj,
    sleeve_type: str,
    size: str,
    warp_map_path: Optional[str],
    back_color: Sequence[float] = (0.05, 0.1, 0.4),
    front_finish: str = "clear",   # 'clear' | 'matte' (opaque_back only; both stay see-through)
    uv_xform=None,                 # per-instance warp crop (see random_uv_xform)
    link: bool = True,
):
    """Build a sleeve around `card_obj` and parent it to the card. Returns the obj.

    slot 0 = front layer, slot 1 = back layer. 'clear' => both clear; 'opaque_back'
    => opaque colored back + a fully-transmissive front (clear, or slightly rougher
    'matte' — never milky).
    """
    if size not in SLEEVE_MARGINS:
        raise ValueError(f"bad sleeve size {size!r}; expected one of {list(SLEEVE_MARGINS)}")
    mesh = build_sleeve_mesh(size, name=name)
    obj = bpy.data.objects.new(name, mesh)
    if link:
        bpy.context.collection.objects.link(obj)

    if sleeve_type == "clear":
        front_mat = make_clear_plastic(name + "_front", warp_map_path, roughness=0.06,
                                       uv_xform=uv_xform)
        back_mat = make_clear_plastic(name + "_back", warp_map_path, roughness=0.06,
                                      uv_xform=uv_xform)
    elif sleeve_type == "opaque_back":
        # Front is a clear alpha layer; "slightly matte" = a small roughness bump only.
        front_rough = 0.06 if front_finish == "clear" else 0.12
        front_mat = make_clear_plastic(name + "_front", warp_map_path, roughness=front_rough,
                                       uv_xform=uv_xform)
        back_mat = make_opaque_plastic(name + "_back", back_color, warp_map_path,
                                       uv_xform=uv_xform)
    else:
        raise ValueError(f"bad sleeve_type {sleeve_type!r}")

    for i, m in enumerate((front_mat, back_mat)):
        slot = obj.material_slots[i]
        slot.link = "OBJECT"
        slot.material = m

    # Keep the sleeve locked to the card as one unit. The sleeve mesh is centered at
    # local origin, so parenting with the DEFAULT (identity) parent-inverse makes it
    # inherit the card's full transform and wrap the card wherever the card is placed
    # (e.g. offset inside a holder). Do NOT set matrix_parent_inverse to the card's
    # inverse -- that pins the sleeve at world origin and misaligns it for offset cards.
    obj.parent = card_obj
    return obj


# --------------------------------------------------------------------------- #
# Rigid holders: toploader + semi-rigid (spec §3.2)
# --------------------------------------------------------------------------- #
# Both are two flat rigid sheets held `gap` apart, connected on left/right/bottom,
# open at top. Unlike sleeves the sheets stay FLAT (rigid) and parallel, so the
# glass is clean (no curved-seam TIR). Semi-rigid adds a lip (tab) above on the
# back sheet only. Dimensions in meters.
TOPLOADER_W, TOPLOADER_H = 0.074, 0.098   # 4mm wider than the classic 70mm (user)
TOPLOADER_GAP = 0.001                      # 1mm interior (holds a sleeved card)
SEMIRIGID_W, SEMIRIGID_H = 0.081, 0.108
SEMIRIGID_GAP = 0.001
SEMIRIGID_LIP = 0.012                       # 12mm tab above, back sheet only


_SPINE_W = 0.002   # 2mm physical width of the opaque connecting spine (spec ~1.5mm)


def _add_box(bm, x0, x1, y0, y1, z0, z1, mat_index):
    """Add a closed box; return its 6 faces (normals fixed later via recalc)."""
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    v = [bm.verts.new(p) for p in c]
    quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    faces = []
    for q in quads:
        f = bm.faces.new([v[i] for i in q])
        f.material_index = mat_index
        faces.append(f)
    return faces


def _flat_pocket_mesh(name: str, Wo: float, Ho: float, gap: float, lip: float,
                      spine_width: float = 0.0):
    """Two full flat clear sheets (front z=+gap/2, back z=-gap/2), open top, back
    sheet extended by `lip`. If spine_width>0 (toploader), a semi-opaque U-spine of
    that width sits between the sheets on L/R/bottom (slot 1). If 0 (semi-rigid),
    the sheets are joined by a thin CLEAR rim wall instead (all slot 0). Returns a
    bpy mesh; slot count is 2 when spined, else 1."""
    g = gap / 2.0
    hx, yb, yt, ytl = Wo / 2.0, -Ho / 2.0, Ho / 2.0, Ho / 2.0 + lip
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")

    def V(x, y, z):
        return bm.verts.new((x, y, z))

    # Full clear sheets (slot 0). Manual winding: front +Z, back/lip -Z.
    FBL, FBR, FTR, FTL = V(-hx, yb, g), V(hx, yb, g), V(hx, yt, g), V(-hx, yt, g)
    BBL, BBR = V(-hx, yb, -g), V(hx, yb, -g)
    BML, BMR = V(-hx, yt, -g), V(hx, yt, -g)
    sheet_faces = [bm.faces.new((FBL, FBR, FTR, FTL)),
                   bm.faces.new((BBR, BBL, BML, BMR))]
    if lip > 0:
        BTL, BTR = V(-hx, ytl, -g), V(hx, ytl, -g)
        sheet_faces.append(bm.faces.new((BMR, BML, BTL, BTR)))
    for f in sheet_faces:
        f.material_index = 0

    n_slots = 1
    if spine_width > 0:
        # Semi-opaque U-spine (slot 1): bottom bar full width, side bars above it.
        w = spine_width
        zc = g - 0.00005  # sit just inside the sheets (avoid coincident faces)
        box_faces = []
        box_faces += _add_box(bm, -hx, hx, yb, yb + w, -zc, zc, 1)
        box_faces += _add_box(bm, -hx, -hx + w, yb + w, yt, -zc, zc, 1)
        box_faces += _add_box(bm, hx - w, hx, yb + w, yt, -zc, zc, 1)
        bmesh.ops.recalc_face_normals(bm, faces=box_faces)
        n_slots = 2
    else:
        # Thin CLEAR rim wall joining the two sheets on L/R/bottom (slot 0).
        rim = [bm.faces.new((FBL, FBR, BBR, BBL)),
               bm.faces.new((FTL, FBL, BBL, BML)),
               bm.faces.new((FBR, FTR, BMR, BBR))]
        for f in rim:
            f.material_index = 0
        bmesh.ops.recalc_face_normals(bm, faces=list(sheet_faces) + rim)

    span_y = Ho + lip
    for f in sheet_faces:
        for loop in f.loops:
            x, y, _ = loop.vert.co
            loop[uv].uv = ((x + hx) / Wo, (y - yb) / span_y)

    mesh = bpy.data.meshes.new(name + "_mesh")
    bm.to_mesh(mesh)
    bm.free()
    for _ in range(n_slots):
        mesh.materials.append(None)
    return mesh


# Semi-opaque, darker blue-tinted plastic for the toploader connecting spine
# (~35% see-through, per user). Toploader only (semi-rigid has a clear rim).
_SPINE_W = 0.002                          # 2mm spine width
_SPINE_COLOR = (0.28, 0.30, 0.40)         # darker, slightly blue
_SPINE_ALPHA = 0.65                       # ~35% see-through


def make_spine_material(name: str, color=_SPINE_COLOR, alpha: float = _SPINE_ALPHA,
                        roughness: float = 0.4):
    """Darker blue-tinted, partially transparent spine (alpha handled by Cycles)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Alpha"].default_value = alpha
    return mat


def make_toploader_plastic(name: str, warp_map_path: Optional[str],
                           wear_map_path: Optional[str], base_rough: float = 0.05,
                           wear_rough: float = 0.35, tint=(1.0, 1.0, 1.0),
                           uv_xform=None, wear_uv_xform=None):
    """Rigid clear plastic (TOPLOADER / semi-rigid surface): thin-walled shader (no
    refraction, so a holo card inside stays correct). Glossy roughness modulated by the
    wear map; `tint` lightly colors the see-through body."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    warp = _warp_normal_socket(nt, warp_map_path, uv_xform=uv_xform)
    rough_sock = _wear_roughness_socket(nt, wear_map_path, base_rough, wear_rough, wear_uv_xform)
    _thin_walled_graph(nt, roughness_val=base_rough, roughness_socket=rough_sock,
                       tint=tint, warp_sock=warp)
    return mat


def _wear_roughness_socket(nt, wear_map_path, base_rough, wear_rough, wear_uv_xform):
    """MapRange(wear -> [base,worn]) -> Result socket driving roughness, or None."""
    if not (wear_map_path and os.path.isfile(wear_map_path)):
        return None
    try:
        tex = nt.nodes.new("ShaderNodeTexImage")
        img = bpy.data.images.load(wear_map_path, check_existing=True)
        img.colorspace_settings.name = "Non-Color"
        tex.image = img
        _apply_uv_mapping(nt, tex, wear_uv_xform)
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = base_rough
        mr.inputs["To Max"].default_value = wear_rough
        nt.links.new(tex.outputs["Color"], mr.inputs["Value"])
        return mr.outputs["Result"]
    except Exception as exc:  # noqa: BLE001
        print(f"[protection] wear map load failed {wear_map_path}: {exc}")
        return None


def make_slab_surface(name: str, warp_map_path: Optional[str], wear_map_path: Optional[str],
                      base_rough: float = 0.05, wear_rough: float = 0.35,
                      tint=(1.0, 1.0, 1.0), uv_xform=None, wear_uv_xform=None):
    """SLAB surface: KEEP real Principled transmission (it's solid acrylic with front
    AND back faces, so refraction cancels). Reduced front-face warp normal strength +
    a Light-Path shadow-ray branch so lamp light passes through unattenuated."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    out = nt.nodes.get("Material Output")
    bsdf.inputs["Base Color"].default_value = (tint[0], tint[1], tint[2], 1.0)
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    bsdf.inputs["IOR"].default_value = 1.5
    rough_sock = _wear_roughness_socket(nt, wear_map_path, base_rough, wear_rough, wear_uv_xform)
    if rough_sock is not None:
        nt.links.new(rough_sock, bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = base_rough
    _add_warp_normal(nt, bsdf, warp_map_path, strength=0.08, uv_xform=uv_xform)  # reduced
    lp = nt.nodes.new("ShaderNodeLightPath")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Shadow Ray"], mix.inputs[0])
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def _assign_materials(obj, mats):
    for i, m in enumerate(mats):
        slot = obj.material_slots[i]
        slot.link = "OBJECT"
        slot.material = m


def build_toploader(name: str, warp_map_path: Optional[str],
                    wear_map_path: Optional[str] = None, tint=(1.0, 1.0, 1.0),
                    wear_rough: float = 0.35, uv_xform=None, wear_uv_xform=None,
                    link: bool = True):
    """Rigid toploader (74x98mm, 1mm interior) with a 2mm semi-opaque blue spine
    and micro-scratch/dust wear. Slightly tinted (not fully clear). Card always
    sleeved. `wear_rough`==base_rough => scratches effectively absent. `uv_xform`/
    `wear_uv_xform` randomize the warp/wear regions per instance."""
    mesh = _flat_pocket_mesh(name, TOPLOADER_W, TOPLOADER_H, TOPLOADER_GAP, 0.0,
                             spine_width=_SPINE_W)
    obj = bpy.data.objects.new(name, mesh)
    if link:
        bpy.context.collection.objects.link(obj)
    _assign_materials(obj, [
        make_toploader_plastic(name + "_pvc", warp_map_path, wear_map_path,
                               base_rough=0.05, wear_rough=wear_rough, tint=tint,
                               uv_xform=uv_xform, wear_uv_xform=wear_uv_xform),
        make_spine_material(name + "_spine"),
    ])
    return obj


def build_semirigid(name: str, warp_map_path: Optional[str],
                    wear_map_path: Optional[str] = None, tint=(1.0, 1.0, 1.0),
                    wear_rough: float = 0.35, uv_xform=None, wear_uv_xform=None,
                    link: bool = True):
    """Semi-rigid holder (81x108mm main + 12mm back lip, 1mm interior). Clear rim
    (NO opaque spine, per user). Slight tint + surface wear. Card always sleeved."""
    mesh = _flat_pocket_mesh(name, SEMIRIGID_W, SEMIRIGID_H, SEMIRIGID_GAP,
                             SEMIRIGID_LIP, spine_width=0.0)
    obj = bpy.data.objects.new(name, mesh)
    if link:
        bpy.context.collection.objects.link(obj)
    _assign_materials(obj, [
        make_toploader_plastic(name + "_pvc", warp_map_path, wear_map_path,
                               base_rough=0.12, wear_rough=wear_rough, tint=tint,
                               uv_xform=uv_xform, wear_uv_xform=wear_uv_xform),
    ])
    return obj


# --------------------------------------------------------------------------- #
# Graded slab (spec §3.2). Reuses the toploader clear surface for the surface and
# the spine material for the edges + internal rectangle outlines (ridges).
# --------------------------------------------------------------------------- #
SLAB_W, SLAB_H, SLAB_T = 0.080, 0.135, 0.0067        # outer size + overall thickness
# Label area 20x68mm, centered horizontally, 4mm from the top.
_LBL_X0, _LBL_X1 = -0.034, 0.034
_LBL_Y1 = SLAB_H / 2 - 0.004                          # 4mm from top
_LBL_Y0 = _LBL_Y1 - 0.020
_LINE_Y = _LBL_Y0 - 0.004                             # horizontal line 4mm below label
# Card recess 64.2 x 89.70mm, centered in the area below the line.
_RCS_W, _RCS_H = 0.0642, 0.08970
_RCS_CY = (_LINE_Y + (-SLAB_H / 2)) / 2.0
_RCS_X0, _RCS_X1 = -_RCS_W / 2, _RCS_W / 2
_RCS_Y0, _RCS_Y1 = _RCS_CY - _RCS_H / 2, _RCS_CY + _RCS_H / 2
SLAB_CARD_POS = (0.0, _RCS_CY, 0.0)                  # card centred in the recess (z=0)
_RIDGE_LW = 0.0012                                    # outline line width


def _add_bar(bm, x0, y0, x1, y1, lw, z0, z1, mat_index):
    """Add a thin raised bar (box) of width `lw` along segment (x0,y0)->(x1,y1) at
    any orientation (used for the notched/diagonal recess outline). Returns faces."""
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return []
    px, py = -dy / length * lw / 2.0, dx / length * lw / 2.0  # perpendicular * half-w
    corners = [(x0 + px, y0 + py), (x0 - px, y0 - py),
               (x1 - px, y1 - py), (x1 + px, y1 + py)]
    vb = [bm.verts.new((cx, cy, z0)) for (cx, cy) in corners]
    vt = [bm.verts.new((cx, cy, z1)) for (cx, cy) in corners]
    faces = [bm.faces.new(vb), bm.faces.new(vt)]
    for i in range(4):
        j = (i + 1) % 4
        faces.append(bm.faces.new((vb[i], vb[j], vt[j], vt[i])))
    for f in faces:
        f.material_index = mat_index
    return faces


def _octagon_segments(x0, x1, y0, y1, cut):
    """Segments of a rectangle with `cut`-sized corners removed (notched), so a
    card's sharp corners sit free (spec §3.2)."""
    return [
        ((x0 + cut, y0), (x1 - cut, y0)),   # bottom
        ((x1 - cut, y0), (x1, y0 + cut)),   # bottom-right diagonal
        ((x1, y0 + cut), (x1, y1 - cut)),   # right
        ((x1, y1 - cut), (x1 - cut, y1)),   # top-right diagonal
        ((x1 - cut, y1), (x0 + cut, y1)),   # top
        ((x0 + cut, y1), (x0, y1 - cut)),   # top-left diagonal
        ((x0, y1 - cut), (x0, y0 + cut)),   # left
        ((x0, y0 + cut), (x0 + cut, y0)),   # bottom-left diagonal
    ]


def make_connector_material(name: str):
    """Slab connector material for the outer edge + internal outlines: physical,
    slightly LESS see-through than the clear front (transmission 0.7 + a little
    roughness), so the connectors read as solid frosted structure rather than the
    black thin-transmission slivers the raised ridges produced."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.92, 0.93, 0.95, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.2
    bsdf.inputs["Transmission Weight"].default_value = 0.7
    bsdf.inputs["IOR"].default_value = 1.5
    return mat


def make_frosted_back(name: str, warp_map_path: Optional[str], uv_xform=None):
    """Bumpy, rough, translucent back surface that diffuses light (spec/user). High
    roughness + partial transmission + a stronger warp normal for the bumpiness."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.85, 0.85, 0.87, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.85
    bsdf.inputs["Transmission Weight"].default_value = 0.5   # translucent, diffusing
    bsdf.inputs["IOR"].default_value = 1.5
    _add_warp_normal(nt, bsdf, warp_map_path, strength=0.6, uv_xform=uv_xform)  # bumpy
    return mat


def make_label_material(name: str, label_path: Optional[str]):
    """Diffuse procedural grading-label placeholder (color image, sRGB)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.6
    if label_path and os.path.isfile(label_path):
        tex = nt.nodes.new("ShaderNodeTexImage")
        try:
            tex.image = bpy.data.images.load(label_path, check_existing=True)
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        except Exception as exc:  # noqa: BLE001
            print(f"[protection] slab label load failed {label_path}: {exc}")
            bsdf.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
    else:
        bsdf.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
    return mat


def _ridge_rect(bm, x0, x1, y0, y1, lw, z0, z1, mat_index, gap=0.0):
    """A rectangle-outline of 4 bar boxes; returns faces. If gap>0 the bars stop
    `gap` short of each corner so the CORNERS ARE EMPTY (spec: card recess corners
    left free). gap<=0 gives a continuous outline."""
    inset_tb = gap if gap > 0 else 0.0        # top/bottom bar x-inset
    inset_lr = gap if gap > 0 else lw          # left/right bar y-inset
    faces = []
    faces += _add_box(bm, x0 + inset_tb, x1 - inset_tb, y0, y0 + lw, z0, z1, mat_index)  # bottom
    faces += _add_box(bm, x0 + inset_tb, x1 - inset_tb, y1 - lw, y1, z0, z1, mat_index)  # top
    faces += _add_box(bm, x0, x0 + lw, y0 + inset_lr, y1 - inset_lr, z0, z1, mat_index)  # left
    faces += _add_box(bm, x1 - lw, x1, y0 + inset_lr, y1 - inset_lr, z0, z1, mat_index)  # right
    return faces


def _slab_mesh(name: str):
    """Slab mesh. slot 0 = clear front; slot 1 = frosted bumpy back; slot 2 = label;
    slot 3 = CONNECTOR (outer edge side walls + internal outlines + separating line),
    all one physical, slightly-less-see-through material spanning the FULL thickness.
    The card-recess outline is 4 straight bars with EMPTY corners."""
    hw, hh, hz = SLAB_W / 2, SLAB_H / 2, SLAB_T / 2
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")

    def V(x, y, z):
        return bm.verts.new((x, y, z))

    # Front (clear, slot 0), back (frosted, slot 1), outer side walls (connector, slot 3).
    FBL, FBR, FTR, FTL = V(-hw, -hh, hz), V(hw, -hh, hz), V(hw, hh, hz), V(-hw, hh, hz)
    BBL, BBR, BTR, BTL = V(-hw, -hh, -hz), V(hw, -hh, -hz), V(hw, hh, -hz), V(-hw, hh, -hz)
    front = bm.faces.new((FBL, FBR, FTR, FTL))
    front.material_index = 0
    back = bm.faces.new((BBR, BBL, BTL, BTR))
    back.material_index = 1
    edges = [bm.faces.new((FBL, FBR, BBR, BBL)),   # bottom side
             bm.faces.new((FTR, FTL, BTL, BTR)),   # top side
             bm.faces.new((FTL, FBL, BBL, BTL)),   # left side
             bm.faces.new((FBR, FTR, BTR, BBR))]   # right side
    for f in edges:
        f.material_index = 3
    bmesh.ops.recalc_face_normals(bm, faces=[front, back] + edges)
    for f in (front, back):
        for loop in f.loops:
            x, y, _ = loop.vert.co
            loop[uv].uv = ((x + hw) / SLAB_W, (y + hh) / SLAB_H)

    # Full-thickness CONNECTORS (slot 3): label outline, separating line, recess.
    # Span nearly the full depth (inset 0.05mm so they don't coincide with the sheets).
    zc = hz - 0.00005
    conn = []
    conn += _ridge_rect(bm, _LBL_X0, _LBL_X1, _LBL_Y0, _LBL_Y1, _RIDGE_LW, -zc, zc, 3)
    conn += _add_box(bm, -0.037, 0.037, _LINE_Y - _RIDGE_LW / 2, _LINE_Y + _RIDGE_LW / 2,
                     -zc, zc, 3)
    # Recess: 4 straight bars, 4mm gap at each corner (corners left empty).
    conn += _ridge_rect(bm, _RCS_X0, _RCS_X1, _RCS_Y0, _RCS_Y1, _RIDGE_LW, -zc, zc, 3,
                        gap=0.004)
    bmesh.ops.recalc_face_normals(bm, faces=conn)

    # Label plane (slot 2), just in front of the front sheet.
    lz = hz + 0.0002
    P0, P1 = V(_LBL_X0, _LBL_Y0, lz), V(_LBL_X1, _LBL_Y0, lz)
    P2, P3 = V(_LBL_X1, _LBL_Y1, lz), V(_LBL_X0, _LBL_Y1, lz)
    label = bm.faces.new((P0, P1, P2, P3))
    label.material_index = 2
    for loop in label.loops:
        x, y, _ = loop.vert.co
        loop[uv].uv = ((x - _LBL_X0) / (_LBL_X1 - _LBL_X0),
                       (y - _LBL_Y0) / (_LBL_Y1 - _LBL_Y0))

    mesh = bpy.data.meshes.new(name + "_mesh")
    bm.to_mesh(mesh)
    bm.free()
    for _ in range(4):
        mesh.materials.append(None)   # 0 clear front, 1 frosted back, 2 label, 3 connector
    return mesh


def build_slab(name: str, warp_map_path: Optional[str], wear_map_path: Optional[str] = None,
               label_path: Optional[str] = None, tint=(1.0, 1.0, 1.0),
               wear_rough: float = 0.35, uv_xform=None, wear_uv_xform=None, link: bool = True):
    """Graded slab (80x135mm, 6.7mm thick). Clear surface + physical clear edges &
    ridge outlines (same material), a frosted bumpy light-diffusing back, and a
    procedural label. Card sits in the recess (SLAB_CARD_POS) with NO sleeve (slab
    rule). Returns obj at origin."""
    mesh = _slab_mesh(name)
    obj = bpy.data.objects.new(name, mesh)
    if link:
        bpy.context.collection.objects.link(obj)
    _assign_materials(obj, [
        make_slab_surface(name + "_surface", warp_map_path, wear_map_path,
                          base_rough=0.05, wear_rough=wear_rough, tint=tint,
                          uv_xform=uv_xform, wear_uv_xform=wear_uv_xform),
        make_frosted_back(name + "_back", warp_map_path, uv_xform=wear_uv_xform),
        make_label_material(name + "_label", label_path),
        make_connector_material(name + "_connector"),
    ])
    return obj
