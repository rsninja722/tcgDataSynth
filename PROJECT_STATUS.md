# Project Status

Last consolidated: 2026-07-26.

## Active Checkpoint

Phase 4, display-case acceptance. Run `tests/t12_display_case.py` in Blender 5.0 and review its console output plus all visualized renders.

Acceptance focus:

- Coplanar grid cards do not occlude one another.
- A nearer top card or overlapping card carves the farther card's polygon.
- Cards with more than 80% of their in-frustum area occluded are not labeled.
- Created polygon vertices are magenta in the visualizer.
- Full and partial frustum bounds align with the rendered cards.
- Grid bounds follow the apparent card corners after lid/slab refraction; the card on
  top of the lid remains directly projected unless it is itself slabbed.
- Every t12 scene has a top card resting flat on the acrylic lid.
- The table backdrop fills the area below the case.

Do not start the hand layout until this checkpoint is reviewed with the user.

## Phase Progress

- Phase 0: Blender 5.0 API introspection complete.
- Phase 1: bare card and fixed-corner projection validated.
- Phase 2: protection assets implemented; sleeves and holders reviewed, slab review was waived after final connector changes.
- Phase 3: finish, holo, physical-texture, and damage pipelines implemented and integrated.
- Phase 4: table and floating reviewed; binder implemented; display case awaits current acceptance; hand not started.
- Phase 5: lighting and camera randomization not started.
- Phase 6: post effects not started.
- Phase 7: modal-timer GUI/orchestration not started.
- Phase 8: throughput comparison and 50-image pilot not started.

## Locked Decisions

- Blender executable: `C:\Program Files\Blender Foundation\Blender 5.0\blender.exe`.
- Blender Python: `C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe`.
- Runtime dependencies in Blender: OpenCV headless and Shapely.
- Primary renderer: Cycles with GPU/CUDA, 128 samples, denoising. EEVEE is comparison-only.
- Color management: AgX.
- Orchestration: modal timer, one completed scene per tick. Pause finishes the current image/label pair before stopping.
- Card image root: configurable with `TCG_CARD_IMAGE_ROOT`; current user default remains in `config.py`.
- Card back: `back.png` in the image root.
- Picture region: project default unless `picture_regions.json` provides a per-card override.
- Slab label: procedural placeholder.
- Canonical card-corner order: TL, TR, BR, BL.
- A back-facing card is not labeled.
- Damage variation is per instance, not cached solely by card ID.
- Bulk acrylic label projection uses finite oriented boxes, nominal IOR 1.5, and the
  geometric-normal Snell ray as the apparent scattering centroid.

## Active Label Contract

The user selected an occlusion-aware custom polygon label during Phase 4. It supersedes fixed YOLO-pose labels for current layout scripts. Full details are in `LABEL_FORMAT.md`.

This is a project blocker before Phase 8: the custom format cannot be passed directly to Ultralytics. Choose a custom trainer/adapter or define a supported export representation before generating the pilot dataset.

## Known Risks

- The bpy portion of occlusion-aware labeling has not yet been executed by the user.
- The refractive label projection has pure-Python coverage but has not yet been
  compared visually against Blender Cycles; t12 is the active acceptance path.
- Refraction intentionally ignores surface roughness, scratches, smudges, and normal
  maps. Intersecting/touching refractive boxes fail explicitly rather than using an
  incorrect nested-medium approximation.
- Occlusion currently approximates each embedded card with a projected rectangle and mean depth. Intersecting or strongly tilted geometry may need a depth-aware method before the hand layout.
- The single-ring custom format keeps one connected polygon and bridges holes; disconnected visible regions are not represented exactly.
- The prebuilt protection-library loader exists but the integrated scene builder still constructs protection geometry per instance. Address sharing before throughput work or earlier if display-case memory is excessive.
- Some older numbered test headers describe historical implementations. Treat current source and this status as authoritative; update a script when it becomes the active acceptance test.

## Next Goals

1. Receive and address the t12 Blender acceptance results.
2. Implement and review the hand layout in one focused acceptance script.
3. Revisit occlusion depth/shape limitations exposed by hand geometry.
4. Proceed to Phase 5 only after Phase 4 is accepted.
