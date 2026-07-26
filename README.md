# tcgDataSynth

Blender 5.0 synthetic-scene generator for training trading-card detectors. It builds randomized cards, protection, finishes, damage, layouts, and labels while keeping geometry-independent logic testable outside Blender.

## Current State

Phases 0-3 are implemented. Phase 4 has table, floating, binder, and display-case layouts; the display-case and new occlusion-aware label path await a Blender verification run. The hand layout and Phases 5-8 are not implemented.

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
assets/         checked-in generated texture assets
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

## Active Acceptance Run

The next user verification is:

```bat
"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tests\t12_display_case.py
```

Then visualize each emitted pair, for example:

```bash
python3 labeltools/visualize.py out/t12_case_toploader_flat_5x5_in.png out/t12_case_toploader_flat_5x5_in.txt
```

Report the Blender console output and attach the `*_viz.png` files. Expected details are in the test script header and `PROJECT_STATUS.md`.

## Important Constraint

The current production path writes a custom occlusion-aware polygon format, not valid Ultralytics YOLO-pose input. `write_dataset_yaml()` remains only for the legacy fixed-corner path. A downstream training adapter or export decision is required before pilot-dataset generation.
