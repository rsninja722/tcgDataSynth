# Project Status

Last consolidated: 2026-07-27.

## Active Checkpoint

Phase 7, standalone GUI and orchestration. The active goal is a desktop Python GUI
that persists option toggles, launches one headless Blender worker per pair, writes a
contiguous seed manifest, and safely pauses/resumes only between completed pairs.

Acceptance focus:

- Verify `tests/t20_standalone_generation.py` completes one isolated worker pair under
  Blender 5.0.0.
- Interactively verify `gui.py` Start, Pause during rendering, Resume, and contiguous
  `out/images`, `out/labels`, and `out/manifest.jsonl` records with a small run.

## Phase Progress

- Phase 0: Blender 5.0 API introspection complete.
- Phase 1: bare card and fixed-corner projection validated.
- Phase 2: protection assets implemented; sleeves and holders reviewed, slab review was waived after final connector changes.
- Phase 3: finish, holo, physical-texture, and damage pipelines implemented and integrated.
- Phase 4: all five layouts, labels, and the bundled compact hand library accepted.
- Phase 5: deterministic lighting/camera and final non-sun simplex shadow masks accepted.
- Phase 6: complete. All eight configurable image-space effects passed review, and the
  `t19` Blender 5.0 output-transaction acceptance passed with aligned labels and no
  staged-output leftovers.
- Phase 7: standalone GUI/worker orchestration implementation and Docker validation
  complete; Blender worker plus interactive Start-Pause-Resume acceptance pending.
- Phase 8: throughput comparison and 50-image pilot not started.

## Locked Decisions

- Blender executable: `C:\Program Files\Blender Foundation\Blender 5.0\blender.exe`.
- Blender Python: `C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe`.
- Runtime dependencies in Blender: OpenCV headless and Shapely.
- Primary renderer: Cycles with GPU/CUDA, 128 samples, denoising. EEVEE is comparison-only.
- Color management: AgX.
- Orchestration: standalone Tkinter GUI launches one headless Blender worker per pair.
  Pause lets the active worker finish and publish its image/label pair before stopping.
- Card image root: configurable with `TCG_CARD_IMAGE_ROOT`; current user default remains in `config.py`.
- Card back: `back.png` in the image root.
- Picture region: project default unless `picture_regions.json` provides a per-card override.
- Slab label: procedural placeholder.
- Canonical card-corner order: TL, TR, BR, BL.
- A back-facing card is not labeled.
- Damage variation is per instance, not cached solely by card ID.
- Bulk acrylic label projection uses finite oriented boxes, nominal IOR 1.5, and the
  geometric-normal Snell ray as the apparent scattering centroid.
- Runtime uses the compact project-local `assets/hand_rig.blend`; `TCG_HAND_ASSET` is
  only an explicit diagnostic override.
- The compact hand library keeps both mesh/armature pairs and their dependencies, not
  the source file's unrelated scene datablocks or legacy materials.
- `assets/hand_rig.blend` passed t15 under Blender 5.0.0 at 2,680,719 bytes and the
  accepted five t14 scenes passed from the bundled default without an override.
- Hands are render-only geometry for labels; they never act as card occluders.
- Accepted Phase 5 energies are sun `0.028125-0.16875`, points `1.125-22.5`, and fixed
  phone flash `14.0625`. Two-point scenes cap each point at `14.5`; three-point scenes
  cap each at `7.5`; one- and four-point scenes use the baseline maximum.
- Accepted point-light color temperature range is 2000-9000 K.
- Shadow masks use a 50x50 face grid. Face-center samples combine 2x and 12x seeded
  simplex noise at 65/35 weights; faces over 0.50 brightness are removed.
- The sun is never masked. The phone flash and every point light are each independently
  sampled at 25% when lighting occluders are enabled.
- Shadow-plane faces are 95% opaque (user-tuned).
- Masked finite lights use a larger emitter radius than unmasked lights to soften the
  grid silhouette. The accepted setting midpoints both that radius and the prior/current
  plane placements to split the difference between sharp and soft shadows; blocker
  sizing includes the resulting radius so no-hole controls retain cover.

## Active Label Contract

The user selected an occlusion-aware custom polygon label during Phase 4. It supersedes fixed YOLO-pose labels for current layout scripts. Full details are in `LABEL_FORMAT.md`.

This is a project blocker before Phase 8: the custom format cannot be passed directly to Ultralytics. Choose a custom trainer/adapter or define a supported export representation before generating the pilot dataset.

## Known Risks

- The display-case and occlusion-aware/refraction labeling checkpoint was accepted by
  the user after Blender 5.0 visual testing.
- The Blender 2.79 hand meshes, armatures, Multires, constraints, skin materials, and
  control deformation were accepted after the t13 Blender 5.0 report/renders. The
  t13 control-grid meshes were hidden by a diagnostic copy bug, now fixed; numeric
  control response in the report confirmed all four finger chains. The generated
  Blender 5 compact library passed t15 inventory/rig validation and t14 equivalence.
- Refraction intentionally ignores surface roughness, scratches, smudges, and normal
  maps. Intersecting/touching refractive boxes fail explicitly rather than using an
  incorrect nested-medium approximation.
- Card-card occlusion still uses projected rectangles and mean depth. Hand geometry is
  intentionally excluded from occlusion calculations.
- The single-ring custom format keeps one connected polygon and bridges holes; disconnected visible regions are not represented exactly.
- The prebuilt protection-library loader exists but the integrated scene builder still constructs protection geometry per instance. Address sharing before throughput work or earlier if display-case memory is excessive.
- Camera orbit is now sampled explicitly around the 0-50 degree off-axis cone. Point
  positions and sun angles are interpreted in a camera-relative front-hemisphere basis.
- The final t17 shadow shape, 95% opacity, and midpoint softness/placement were visually
  accepted by the user under Blender 5.0.0.
- Container validation for the lighting revamp passes all 126 unit tests, `compileall`,
  and `git diff --check`.
- Some older numbered test headers describe historical implementations. Treat current source and this status as authoritative; update a script when it becomes the active acceptance test.

## Next Goals

1. Complete the Phase 7 Docker tests and Blender `t20` registration/interactive
   Start-Pause-Resume acceptance.
2. Mark Phase 7 complete after the small-run image/label/manifest continuity report.
