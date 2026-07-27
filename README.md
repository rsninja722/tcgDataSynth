# tcgDataSynth

Blender 5.0 synthetic-scene generator for training trading-card detectors. It builds randomized cards, protection, finishes, damage, layouts, and labels while keeping geometry-independent logic testable outside Blender.

## Current State

Phases 0-4 are implemented and accepted, including table, floating, binder,
display-case, and hand layouts with occlusion-aware labels. Phase 5 camera, lighting,
and non-sun simplex shadow masks are accepted. Phase 6 post effects is active; Phases
7-8 are not implemented.

See `PROJECT_STATUS.md` for the active checkpoint, validated decisions, and next work. See `LABEL_FORMAT.md` before consuming labels.

## Development Model

The development container has no Blender or GUI. Code is split accordingly:

```text
rules/          deterministic scene sampling and legality
texturegen/     OpenCV/NumPy texture generation
postfx/         render post-processing (not implemented yet)
labeltools/     label geometry, serialization, and visualization
blender/        bpy-only scene construction and rendering
tests/unit/     container-runnable tests
tests/t*.py     numbered Blender acceptance scripts
assets/         checked-in generated textures and compact Blender libraries
out/            generated output (ignored)
```

Substantial Blender changes are delivered through a focused numbered script, then paused for user feedback before the next major change.

## Container Setup

Python 3.11 is expected.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
bash run_unit_tests.sh
.venv/bin/python -m compileall -q .
```

`run_unit_tests.sh` fails early when required dependencies are absent. Shapely is mandatory because silently omitting occlusion would produce incorrect labels.

## Blender Setup

The known Blender executable is:

```text
C:\Program Files\Blender Foundation\Blender 5.0\blender.exe
```

Install runtime dependencies into Blender's bundled Python from an administrator terminal:

```bat
"C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe" -m pip install opencv-python-headless shapely
```

Card images default to the path in `config.py`. Override it without editing code:

```bat
set TCG_CARD_IMAGE_ROOT=C:\path\to\card-images
```

The image root must contain `back.png`. Optional `picture_regions.json` entries use `{card_id: [x0, y0, x1, y1]}` normalized from the image's top-left.

The hand layout loads `assets/hand_rig.blend` by default. `TCG_HAND_ASSET` remains an
optional diagnostic override for another compatible library.

The compact hand library passed `tests/t15_hand_asset_bundle.py` under Blender 5.0.0
and the five integrated `tests/t14_hand.py` cases passed from the bundled default.
Rebuild and validation instructions are in `assets/HAND_RIG.md`.

## Next Phase

Phase 6 will implement and Docker-test deterministic sensor noise, compression,
pixel-melt blur, white-balance shift, and tint shift before integrating them into the
Blender render pipeline. See `PROJECT_STATUS.md` for the active acceptance goals.

## Important Constraint

The current production path writes a custom occlusion-aware polygon format, not valid Ultralytics YOLO-pose input. `write_dataset_yaml()` remains only for the legacy fixed-corner path. A downstream training adapter or export decision is required before pilot-dataset generation.
