# Project Status

Last consolidated: 2026-07-26.

## Active Checkpoint

Phase 4, integrated hand acceptance. Run `tests/t14_hand.py` in Blender 5.0, visualize
all five image/label pairs, and review grip contact plus original card polygons.

Acceptance focus:

- Pinch grips place the thumb in front and index behind bare/sleeved/toploadered cards.
- Side grips put thumb and fingers on opposite faces of sleeves/toploaders.
- Left/right hands approach from the requested cardinal or diagonal side and vary
  between shallow and normal contact. Depth is capped at 0.34 because deeper grips
  clipped into cards during t14 review.
- The reused noisy table fills the background and skin tones remain plausible.
- Every fully in-frame hand-held card retains flags 1,2,4,3. Hands and transparent
  protection do not carve hand-scene labels.

## Phase Progress

- Phase 0: Blender 5.0 API introspection complete.
- Phase 1: bare card and fixed-corner projection validated.
- Phase 2: protection assets implemented; sleeves and holders reviewed, slab review was waived after final connector changes.
- Phase 3: finish, holo, physical-texture, and damage pipelines implemented and integrated.
- Phase 4: table, floating, binder, and display case reviewed; integrated hand acceptance active.
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
- The supplied CC0 hand model was validated in Blender 5.0 and is resolved by
  `TCG_HAND_ASSET` (with the user's known Windows path as a default).
- Hands are render-only geometry for labels; they never act as card occluders.

## Active Label Contract

The user selected an occlusion-aware custom polygon label during Phase 4. It supersedes fixed YOLO-pose labels for current layout scripts. Full details are in `LABEL_FORMAT.md`.

This is a project blocker before Phase 8: the custom format cannot be passed directly to Ultralytics. Choose a custom trainer/adapter or define a supported export representation before generating the pilot dataset.

## Known Risks

- The display-case and occlusion-aware/refraction labeling checkpoint was accepted by
  the user after Blender 5.0 visual testing.
- The Blender 2.79 hand meshes, armatures, Multires, constraints, skin materials, and
  control deformation were accepted after the t13 Blender 5.0 report/renders. The
  t13 control-grid meshes were hidden by a diagnostic copy bug, now fixed; numeric
  control response in the report confirmed all four finger chains.
- Refraction intentionally ignores surface roughness, scratches, smudges, and normal
  maps. Intersecting/touching refractive boxes fail explicitly rather than using an
  incorrect nested-medium approximation.
- Card-card occlusion still uses projected rectangles and mean depth. Hand geometry is
  intentionally excluded from occlusion calculations.
- The single-ring custom format keeps one connected polygon and bridges holes; disconnected visible regions are not represented exactly.
- The prebuilt protection-library loader exists but the integrated scene builder still constructs protection geometry per instance. Address sharing before throughput work or earlier if display-case memory is excessive.
- Some older numbered test headers describe historical implementations. Treat current source and this status as authoritative; update a script when it becomes the active acceptance test.

## Next Goals

1. Receive and address the t14 integrated grip renders and visualized labels.
2. Tune source-rig pose axes/contact transforms if t14 exposes intersections.
3. Mark Phase 4 accepted, then proceed to Phase 5.
