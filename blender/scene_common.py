"""
Shared scene-setup helpers for the numbered test scripts (bpy REQUIRED).

Consolidates the reset / world / camera / lights / framing boilerplate that t02-t03
each re-implemented, so later phase scripts stay short and consistent. Production
layout/lighting/camera code (Phase 4-5) will supersede these with randomized
versions; these are the plain deterministic defaults used for close-up reviews.
"""
from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Vector

import config


def reset_scene():
    """Delete all objects and purge orphaned datablocks (idempotent per case)."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials,
                 bpy.data.lights, bpy.data.cameras, bpy.data.curves,
                 bpy.data.images):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def setup_world(gray: float = 0.2):
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (gray, gray, gray, 1.0)
    bg.inputs["Strength"].default_value = 1.0


def frame_distance(focal_mm: float, subject_h: float = config.CARD_H_M,
                   target_frac: float = 0.72) -> float:
    """Distance so `subject_h` (meters) fills ~target_frac of the square frame."""
    fov = 2.0 * math.atan((36.0 / 2.0) / focal_mm)
    return (subject_h / target_frac) / (2.0 * math.tan(fov / 2.0))


def setup_camera(focal_mm: float, azimuth_deg: float, elevation_deg: float,
                 distance: float, target=(0.0, 0.0, 0.0)):
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = "PERSP"
    cam_data.lens = focal_mm
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.shift_x = 0.0
    cam_data.shift_y = 0.0
    cam_data.clip_start = 0.001
    cam_data.clip_end = 100.0
    tgt = Vector(target)
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    offset = Vector((math.sin(az) * math.cos(el), math.sin(el),
                     math.cos(az) * math.cos(el))) * distance
    loc = tgt + offset
    cam = bpy.data.objects.new("Cam", cam_data)
    cam.location = loc
    cam.rotation_euler = (tgt - loc).normalized().to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def _link(obj):
    bpy.context.collection.objects.link(obj)


def add_lights(cam_loc, target=(0.0, 0.0, 0.0), rng=None, sun_energy: float = 0.2,
               spot_energy: float = 3.0, point_energy_range=(1.0, 2.5), n_points: int = 2):
    """§3.6-style rig for close-up reviews (tuned down to avoid overexposure):
      - a LOW sun (90% dimmer than before) for ambient fill,
      - a cold SPOT beside the camera aimed along the view (phone-flash analog),
      - `n_points` randomly-positioned point lights (warm<->cold) in the front hemisphere.
    Pass an rng (numpy Generator) to vary the point lights per scene."""
    if rng is None:
        rng = np.random.default_rng(0)
    cam = Vector(cam_loc)
    tgt = Vector(target)
    view_dir = (tgt - cam).normalized()

    # Low sun (front hemisphere).
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = sun_energy
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(25))
    _link(sun)

    # Cold spot beside the camera, aimed at the subject.
    right = view_dir.cross(Vector((0.0, 0.0, 1.0)))
    if right.length < 1e-6:
        right = Vector((1.0, 0.0, 0.0))
    right.normalize()
    spot_data = bpy.data.lights.new("Spot", type="SPOT")
    spot_data.energy = spot_energy
    spot_data.spot_size = math.radians(70.0)
    spot_data.spot_blend = 0.3
    spot_data.color = (0.85, 0.9, 1.0)  # cold
    spot = bpy.data.objects.new("Spot", spot_data)
    spot.location = cam + right * 0.03 + Vector((0.0, 0.0, 0.02))
    spot.rotation_euler = (tgt - spot.location).normalized().to_track_quat("-Z", "Y").to_euler()
    _link(spot)

    # Random point lights, warm<->cold, front hemisphere.
    for i in range(n_points):
        pos = tgt + Vector((float(rng.uniform(-0.25, 0.25)),
                            float(rng.uniform(-0.25, 0.25)),
                            float(rng.uniform(0.08, 0.35))))
        t = float(rng.random())            # 0 warm -> 1 cold
        color = (1.0, 0.82 + 0.18 * t, 0.66 + 0.34 * t)
        pt_data = bpy.data.lights.new(f"Point{i}", type="POINT")
        pt_data.energy = float(rng.uniform(*point_energy_range))
        pt_data.color = color
        pt = bpy.data.objects.new(f"Point{i}", pt_data)
        pt.location = pos
        _link(pt)
