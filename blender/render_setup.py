"""
Unified render setup (bpy REQUIRED). One place to configure engine + output so
every test script and the production loop stay consistent.

Supports Cycles (default; correct transmission for plastic, GPU/CUDA) and EEVEE
(kept for the Phase 8 throughput comparison). Color management is locked to
config.VIEW_TRANSFORM regardless of engine (postfx assumes constant input).
"""
from __future__ import annotations

from typing import Optional

import bpy

import config


def _setup_cycles(scene, samples: int, device: str) -> str:
    scene.render.engine = "CYCLES"
    note = "CYCLES"
    # Respect the user's configured compute device type (e.g. CUDA); just make
    # sure the individual devices are enabled and the scene renders on GPU.
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        enabled = []
        for dev in prefs.devices:
            if dev.type != "CPU":
                dev.use = True
            if dev.use:
                enabled.append(f"{dev.name}({dev.type})")
        note += f" compute_type={getattr(prefs, 'compute_device_type', '?')} devices={enabled}"
    except Exception as exc:  # noqa: BLE001
        note += f" [prefs unavailable: {exc}]"
    try:
        scene.cycles.device = device
    except Exception as exc:  # noqa: BLE001
        note += f" [device set failed: {exc}]"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    # Stacked thin-walled plastic (card + sleeve[2] + toploader[2]) needs many
    # transparent bounces or the layers turn opaque/black.
    try:
        scene.cycles.transparent_max_bounces = 32
        scene.cycles.max_bounces = max(scene.cycles.max_bounces, 16)
    except Exception as exc:  # noqa: BLE001
        note += f" [bounce set failed: {exc}]"
    return note


def _setup_eevee(scene, samples: int) -> str:
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = samples
    try:
        scene.eevee.use_raytracing = True  # needed for reflections/transmission
    except Exception as exc:  # noqa: BLE001
        print(f"[render_setup] eevee use_raytracing: {exc}")
    return "BLENDER_EEVEE (use_raytracing=True)"


def setup_render(scene, engine: Optional[str] = None, verbose: bool = True) -> str:
    """Configure resolution, output, color management, and the render engine.
    Returns a short human-readable note about the engine/devices selected."""
    engine = engine or config.RENDER_ENGINE
    r = scene.render
    r.resolution_x = config.RENDER_W
    r.resolution_y = config.RENDER_H
    r.resolution_percentage = 100
    r.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = config.VIEW_TRANSFORM

    if engine == "CYCLES":
        note = _setup_cycles(scene, config.CYCLES_SAMPLES, config.CYCLES_DEVICE)
    elif engine == "BLENDER_EEVEE":
        note = _setup_eevee(scene, config.EEVEE_RENDER_SAMPLES)
    else:
        raise ValueError(f"Unsupported render engine: {engine!r}")

    if verbose:
        print(f"[render_setup] engine -> {note}")
    return note
