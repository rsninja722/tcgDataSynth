# Project Status

Last consolidated: 2026-08-28.

## Active Checkpoint

Requested dataset-generation extensions implemented; Blender t22 acceptance pending.

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
- Shadow-plane opacity is read from `config.json`; the current default is 95%.
- Masked finite lights use a larger emitter radius than unmasked lights to soften the
  grid silhouette. The accepted setting midpoints both that radius and the prior/current
  plane placements to split the difference between sharp and soft shadows; blocker
  sizing includes the resulting radius so no-hole controls retain cover.
- Binder and display-case card units receive independent +/-2mm XY offsets and
  +/-1-degree center rotations. Existing card-within-holder transforms remain nested.
- Every image samples a center-weighted 7x7 motion-blur trail in one of 16 directions.
  The default effect probability is 1.0 and its light copy strength is configurable.
- Camera f-stop sampling uses `camera.aperture_fstop_range` in `config.json`.
- Camera off-axis maxima are configurable per layout. Display cases are limited to
  30 degrees from straight down; all other layouts currently retain 50 degrees.
- Production cameras aim and focus through one randomly selected front-facing card.
  After layout construction, fully contained scenes zoom in by 1mm focal-length steps
  to the first card/frustum crossing and randomly retain that step or roll back 1-2;
  scenes already containing a partial/out-of-view card keep their initial zoom.
- Sun, point, and phone-flash emitter sizes were increased modestly to soften shadow
  edges. Shadow masks retain their accepted placement and 50x50 breakup geometry.
- Optional standard YOLO segmentation collapses full/partial cards to class 0 and writes
  synchronized identity/holo metadata under `extra_label`.
- Table-bearing layouts use a configured image directory, choosing 50/50 between a
  single image and a four-image 2x2 material with a smooth 5% seam overlap.
- Stack scenes contain 1-10 uniformly protected cards; only the top card is labelable,
  and a hand is sampled around it with 25% probability. Lower stack cards remain
  render-only and are excluded from label occlusion so mean-depth ordering cannot hide
  the physically topmost card.
- A single GUI probability controls intentional cardless variants for every layout.
- Each standalone worker removes `out/card_cache` in a `finally` block.
- Failed GUI workers are persisted as skipped manifest attempts so resume advances to
  the next seed; fully published outputs are recovered instead of skipped.

## Active Label Contract

The occlusion-aware custom polygon remains the primary label. A selectable standard
YOLO segmentation export now writes class-0 polygons plus positional identity/holo
metadata in a sibling directory. Full details are in `LABEL_FORMAT.md`.

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
- A total-internal-reflection result from the initial direct apparent-ray trial causes
  the solver to search nearby finite-box transmission branches. If any card corner
  still has no converged branch, production uses the direct pre-refraction four-corner
  polygon for both labeling and occlusion and appends the fallback instance to
  `refraction_failures.txt`.
- Card-card occlusion still uses projected rectangles and mean depth. Hand geometry is
  intentionally excluded from occlusion calculations.
- The single-ring custom format keeps one connected polygon and bridges holes; disconnected visible regions are not represented exactly.
- The prebuilt protection-library loader exists but the integrated scene builder still constructs protection geometry per instance. Address sharing before throughput work or earlier if display-case memory is excessive.
- Camera orbit is now sampled explicitly around the 0-50 degree off-axis cone. Point
  positions and sun angles are interpreted in a camera-relative front-hemisphere basis.
- The final t17 shadow shape and midpoint placement were visually accepted by the user
  under Blender 5.0.0. The later configurable opacity/softness touch-up awaits t21 review.
- Container validation passes all 160 unit tests,
  `compileall`, and `git diff --check`.
- The t21 Blender 5.0 touch-up acceptance script has not yet been run in this container,
  which has no Blender.
- Some older numbered test headers describe historical implementations. Treat current source and this status as authoritative; update a script when it becomes the active acceptance test.
- The container has no Blender. The integrated stack, image-texture shader, cardless
  geometry, YOLO publication, and cache cleanup await `tests/t22_requested_features.py`
  under Blender 5.0.0.
- The seed-867779 stack/YOLO missing-label regression awaits
  `tests/t23_stack_yolo_seed.py` under Blender 5.0.0.

## Next Goals
