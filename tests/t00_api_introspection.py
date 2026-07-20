"""
Phase 0 - API ground truth & environment check.

WHY THIS EXISTS
    Everything else in this project is written against the *actual* Blender 5.0
    Python API, not against assumptions. Socket names on the Principled BSDF,
    DoF property paths, render-engine identifiers, and view-transform names have
    all churned across Blender 4.x -> 5.0. This script dumps the real surface so
    all later code is based on YOUR reported output, not the author's memory.

HOW TO RUN (headless, no GUI needed):
    blender -b -P tests/t00_api_introspection.py

    `-b` = background/headless. `-P` = run this python file. On Windows the
    `blender` executable is typically:
      "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe"
    so the full command would be, from inside the tcgDataSynth folder:
      "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tests/t00_api_introspection.py

WHAT IT PRODUCES
    - Console output (printed).
    - A copy written to  out/phase0_api_report.txt

WHAT TO REPORT BACK
    Paste the FULL console output, OR attach out/phase0_api_report.txt.
    That single file unblocks Phase 1. Nothing else needed for this step.

This script only reads the API and creates throwaway datablocks in memory.
It writes exactly one text file (out/phase0_api_report.txt) and nothing else.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Tiny buffered logger so we print to console AND capture to a file, and so a
# failure in any one probe never aborts the rest of the report.
# ---------------------------------------------------------------------------
_LINES = []


def log(msg=""):
    print(msg)
    _LINES.append(str(msg))


def section(title):
    log("")
    log("=" * 78)
    log(f"## {title}")
    log("=" * 78)


def probe(title, fn):
    """Run one introspection block, catching and reporting any error."""
    section(title)
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - we want everything, even failures
        import traceback
        log(f"!! PROBE FAILED: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def p_environment():
    import bpy
    log(f"bpy.app.version        = {bpy.app.version}")
    log(f"bpy.app.version_string = {bpy.app.version_string}")
    log(f"bpy.app.build_branch   = {bpy.app.build_branch}")
    log(f"python sys.version     = {sys.version}")
    log(f"python executable      = {sys.executable}")
    log(f"platform               = {sys.platform}")
    # Numpy ships inside Blender; confirm version for our bpy-free modules parity.
    try:
        import numpy
        log(f"bundled numpy          = {numpy.__version__}")
    except Exception as exc:  # noqa: BLE001
        log(f"bundled numpy          = <import failed: {exc}>")


def p_render_engines():
    import bpy
    prop = bpy.types.RenderSettings.bl_rna.properties["engine"]
    log("Available render engine enum identifiers (use the IDENTIFIER in code):")
    for item in prop.enum_items:
        log(f"  identifier={item.identifier!r:32}  name={item.name!r:24}  desc={item.description!r}")
    log("")
    log(f"Default scene engine currently = {bpy.context.scene.render.engine!r}")


def p_principled_sockets():
    import bpy
    mat = bpy.data.materials.new("t00_probe_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    node = nt.nodes.new("ShaderNodeBsdfPrincipled")
    log("Principled BSDF INPUT sockets (index | name | identifier | type | default):")
    for i, sock in enumerate(node.inputs):
        try:
            dv = sock.default_value
            if hasattr(dv, "__len__") and not isinstance(dv, str):
                dv = tuple(round(float(x), 4) for x in dv)
        except Exception:  # noqa: BLE001
            dv = "<no default_value>"
        log(f"  [{i:2}] name={sock.name!r:26} id={sock.identifier!r:26} "
            f"type={sock.type!r:8} default={dv}")
    log("")
    # Distribution / subsurface method enums often carry iridescence-relevant options.
    for enum_prop in ("distribution", "subsurface_method"):
        if hasattr(node, enum_prop):
            try:
                items = node.bl_rna.properties[enum_prop].enum_items
                opts = [it.identifier for it in items]
                log(f"Principled node.{enum_prop} options = {opts}")
            except Exception as exc:  # noqa: BLE001
                log(f"Principled node.{enum_prop}: <error {exc}>")
    bpy.data.materials.remove(mat)


def p_camera_and_dof():
    import bpy
    cam_data = bpy.data.cameras.new("t00_probe_cam")
    log("Camera data properties of interest:")
    for name in ("lens", "lens_unit", "sensor_width", "sensor_height",
                 "sensor_fit", "angle", "type", "clip_start", "clip_end",
                 "shift_x", "shift_y"):
        log(f"  cam.{name:14} = {getattr(cam_data, name, '<MISSING>')!r}")
    log("")
    log("DoF sub-struct (cam.dof.*):")
    dof = getattr(cam_data, "dof", None)
    if dof is None:
        log("  !! cam.dof does not exist -- report this")
    else:
        for prop in dof.bl_rna.properties:
            if prop.identifier == "rna_type":
                continue
            log(f"  cam.dof.{prop.identifier:22} type={prop.type:8} "
                f"(default={getattr(dof, prop.identifier, '<?>')!r})")
    bpy.data.cameras.remove(cam_data)


def p_world_to_camera_view():
    log("Projection helper availability (used for corner labeling in Phase 1):")
    try:
        import bpy_extras
        from bpy_extras.object_utils import world_to_camera_view
        log("  bpy_extras.object_utils.world_to_camera_view = AVAILABLE")
        log(f"  signature hint: {world_to_camera_view.__doc__!r}")
    except Exception as exc:  # noqa: BLE001
        log(f"  world_to_camera_view import FAILED: {exc}")
    # object_utils also holds camera_fit / helpers; list what's there.
    try:
        import bpy_extras.object_utils as ou
        names = [n for n in dir(ou) if not n.startswith("_")]
        log(f"  bpy_extras.object_utils members = {names}")
    except Exception as exc:  # noqa: BLE001
        log(f"  could not list object_utils members: {exc}")


def p_cycles_settings():
    import bpy
    scene = bpy.context.scene
    log("scene.cycles present? " + str(hasattr(scene, "cycles")))
    cyc = getattr(scene, "cycles", None)
    if cyc is None:
        log("  (Cycles addon may be disabled in this build; enable to compare in Phase 8.)")
        return
    for name in ("samples", "preview_samples", "use_denoising", "denoiser",
                 "use_adaptive_sampling", "device", "max_bounces",
                 "use_persistent_data", "film_transparent"):
        val = getattr(cyc, name, "<MISSING>")
        log(f"  scene.cycles.{name:22} = {val!r}")
    # persistent data may live on render, confirm both
    log(f"  scene.render.use_persistent_data = "
        f"{getattr(scene.render, 'use_persistent_data', '<MISSING>')!r}")


def p_eevee_settings():
    import bpy
    scene = bpy.context.scene
    log("scene.eevee present? " + str(hasattr(scene, "eevee")))
    ev = getattr(scene, "eevee", None)
    if ev is None:
        log("  !! no scene.eevee -- report this")
        return
    log("All scene.eevee properties (identifier = current default):")
    for prop in ev.bl_rna.properties:
        if prop.identifier == "rna_type":
            continue
        val = getattr(ev, prop.identifier, "<?>")
        if hasattr(val, "__len__") and not isinstance(val, str):
            try:
                val = tuple(val)
            except Exception:  # noqa: BLE001
                pass
        log(f"  scene.eevee.{prop.identifier:30} = {val!r}")


def p_color_management():
    import bpy
    scene = bpy.context.scene
    vs = scene.view_settings
    log(f"scene.view_settings.view_transform (current) = {vs.view_transform!r}")
    try:
        items = vs.bl_rna.properties["view_transform"].enum_items
        log(f"Available view_transform options = {[it.identifier for it in items]}")
    except Exception as exc:  # noqa: BLE001
        log(f"  could not enumerate view_transform: {exc}")
    log(f"scene.view_settings.look = {vs.look!r}")
    log(f"scene.display_settings.display_device = {scene.display_settings.display_device!r}")
    log(f"scene.sequencer_colorspace_settings present? "
        f"{hasattr(scene, 'sequencer_colorspace_settings')}")
    # Output image settings we will lock down for postfx consistency.
    img = scene.render.image_settings
    log(f"render.image_settings.file_format = {img.file_format!r}")
    log(f"render.image_settings.color_mode  = {img.color_mode!r}")
    log(f"render.image_settings.color_depth = {img.color_depth!r}")


def p_resolution_defaults():
    import bpy
    r = bpy.context.scene.render
    log(f"render.resolution_x = {r.resolution_x}, resolution_y = {r.resolution_y}, "
        f"resolution_percentage = {r.resolution_percentage}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("Blender API introspection report - Phase 0")
    log("Generated by tests/t00_api_introspection.py")

    probe("1. Environment / versions", p_environment)
    probe("2. Render engines available", p_render_engines)
    probe("3. Principled BSDF sockets (thin-film / coat / iridescence live here)",
          p_principled_sockets)
    probe("4. Camera + Depth of Field properties", p_camera_and_dof)
    probe("5. world_to_camera_view projection helper", p_world_to_camera_view)
    probe("6. Cycles scene settings", p_cycles_settings)
    probe("7. EEVEE scene settings (full dump)", p_eevee_settings)
    probe("8. Color management / view transform", p_color_management)
    probe("9. Render resolution defaults", p_resolution_defaults)

    # Write the report next to the project's out/ folder. We resolve out/ relative
    # to this script so it works regardless of Blender's current working dir.
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(here, "..", "out"))
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "phase0_api_report.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_LINES) + "\n")
        log("")
        log(f"[OK] Report written to: {report_path}")
    except Exception as exc:  # noqa: BLE001
        log(f"[WARN] Could not write report file: {exc}")

    log("")
    log(">>> Phase 0 complete. Paste this output or attach out/phase0_api_report.txt <<<")


if __name__ == "__main__":
    main()
