# Bundled Hand Rig

`hand_rig.blend` is the runtime hand asset. It is generated with Blender 5 by
`blender/hand_assets_build.py` from the user-supplied CC0 file
`Hands + armature.blend`.

The accepted Blender 5.0.0 bundle is 2,680,719 bytes with SHA-256:

```text
9c83a8936b1039d8f8d74912ae66677f851f8949651742f01e47ace180469d68
```

Source SHA-256:

```text
6cca25beb3f48460f977f1e47f76612eb87900b64b71cc8b4627d52e54561f40
```

The compact library retains only these objects and their dependency closure:

- `Hand.L` and `Hand_Left`
- `Hand.R` and `Hand_Right`

Source cameras, lights, scenes, text, images, and legacy materials are excluded.
Multires data is preserved to avoid changing the accepted hand geometry; runtime caps
evaluation at level 1. Generation replaces the legacy material with a seeded procedural
skin material.

Rebuild from the project root without overwriting the source:

```bat
"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P blender\hand_assets_build.py -- --source "C:\path\to\Hands + armature.blend"
```

Validate the result with `tests/t15_hand_asset_bundle.py`, then rerun
`tests/t14_hand.py` without setting `TCG_HAND_ASSET`.
