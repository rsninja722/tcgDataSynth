# Project Working Agreement

Read `PROJECT_STATUS.md` and `LABEL_FORMAT.md` before changing behavior.

## Environment

- The container has no Blender. Never claim bpy behavior was verified locally.
- Blender 5.0.0 runs on the user's Windows machine.
- Keep rules, texture generation, post-processing, and label math bpy-free and unit-tested.
- Every bpy change needs a focused numbered `tests/tXX_*.py` acceptance path.

## Workflow

- Work one phase and one significant acceptance goal at a time.
- After a significant Blender-visible change, give one compact run command, expected result, and requested artifacts, then wait for feedback.
- Update `PROJECT_STATUS.md` when a decision, checkpoint, or active risk changes. Do not keep a chronological session diary.
- Preserve deterministic sampling: one `numpy.random.Generator` derived from the scene seed; never use Python's randomized `hash()` or global random draws.

## Correctness

- Blender consumes validated `rules.combinations.SceneConfig`; it must not make legality decisions.
- OpenCV and Shapely are required generation dependencies. Do not silently downgrade finishes, damage, or occlusion when they are unavailable.
- Cycles is the production renderer. EEVEE is only an explicit Phase 8 comparison, not an automatic fallback.
- Units are meters. Card dimensions are 0.063 x 0.088 x 0.00045 m.
- Card IDs are image filename stems. Duplicate stems in a recursively discovered library are invalid.
- The active label path is the custom polygon format in `LABEL_FORMAT.md`; it is not directly Ultralytics-compatible.

## Validation

Run before handing off pure-Python changes:

```bash
bash run_unit_tests.sh
python3 -m compileall -q .
```

Keep generated output under `out/`. Do not commit caches, bytecode, Blender backup files, or ad hoc diagnostics after their issue is resolved.
