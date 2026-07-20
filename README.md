# tcgDataSynth — Synthetic dataset generator for trading-card corner detection

Blender 5.0 renders randomized trading-card scenes and auto-emits Ultralytics
YOLO-pose labels (4 corner keypoints per card) for training a corner-detection model.

## Working protocol (important)

Development happens in a **headless Docker container with no Blender and no GUI**.
All `bpy` code is therefore *unverified by the author* — it is delivered as small,
numbered test scripts under `tests/` that **you (the user) run in your Blender 5.0**
and report back on. Pure-Python modules (`rules/`, `texturegen/`, `postfx/`,
`labeltools/`) are unit-tested in Docker before integration.

We proceed **one phase at a time** and do not advance until you confirm the current
phase's test passed.

## Layout

```
rules/        pure python: combination rules, config sampling      [Docker-tested]
texturegen/   pure python: normal maps, holo masks, damage overlays[Docker-tested]
postfx/       pure python: sensor/compression/WB effects           [Docker-tested]
labeltools/   pure python: label writing, validation, visualization[Docker-tested]
blender/      bpy-only: assets_build, card_factory, layouts, lighting, camera, labeling, render loop
blender/addon/GUI panel add-on
tests/        numbered manual test scripts you run in Blender
assets/       built asset library .blend + source meshes
cards/        (your card images live here or wherever you point us)
out/          renders + labels from tests and production
```

## Conventions

- Units: **meters**, real-world scale. Card = 0.063 × 0.088 × 0.00045 m.
- All randomness flows from a single seeded `numpy.random.Generator`.
- Every image is reproducible from one integer seed (recorded in a manifest).

## Phase status

- [ ] **Phase 0** — API ground truth. Run `tests/t00_api_introspection.py`. ← *you are here*
- [ ] Phase 1 — One bare card + labels (end-to-end labeling skeleton)
- [ ] Phase 2 — Card factory + protection assets
- [ ] Phase 3 — Finishes & damage (texturegen)
- [ ] Phase 4 — Layouts (table → floating → binder → display case → hand)
- [ ] Phase 5 — Lighting & camera randomization
- [ ] Phase 6 — Post effects (postfx)
- [ ] Phase 7 — GUI + orchestration (start/pause, resume-safe)
- [ ] Phase 8 — Throughput + 50-image pilot

## Run Phase 0 now

```
blender -b -P tests/t00_api_introspection.py
```

Then paste the console output or attach `out/phase0_api_report.txt`.
```
