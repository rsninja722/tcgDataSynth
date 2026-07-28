# Label Formats

## Active Custom Polygon Format

Current Phase 4 layout scripts emit one line per card:

```text
<class> <x_min> <y_min> <x_max> <y_max> (<x> <y> <flag>)+ |<card_id>|<holo_tag>
```

Coordinates are normalized to `[0, 1]` with a top-left origin. Polygon points are perimeter-ordered and may be concave.

Ideal card corners are optically projected before frustum clipping and occlusion. Bulk
acrylic marked by the scene builder (currently slabs and display-case lids) uses a
finite-box Snell-ray approximation at nominal IOR 1.5. The geometric-normal ray
approximates the centroid of rough/scratched transmission; small normal-map, scratch,
and smudge deviations are intentionally ignored. The same apparent polygon is used
when a card acts as an occluder.

Classes:

- `0`: the card's four ideal corners are in the camera frustum.
- `1`: only part of the card intersects the camera frustum.

A front-facing card is omitted entirely when nearer cards cover more than 80% of its
original in-frustum projected area. Transparent protection does not count as an opaque
occluder; occlusion follows the embedded card's offset and rotation.

When any ideal corner has no converged finite-box refraction solution, all four corners
of that card fall back to their direct pre-refraction projections. This complete direct
polygon remains eligible for labeling and is used as the card's occluder. Production
generation records each fallback in `refraction_failures.txt` beside `manifest.jsonl`,
including image/label paths, seed, card ID, instance, protection, failed corner, solver
error, and fallback mode.

Hands are deliberately excluded from occlusion calculations. A hand-held card retains
its original frustum-clipped label even where fingers cover part of the rendered card.

Point flags:

- `1`: original top-left card corner.
- `2`: original top-right card corner.
- `3`: original bottom-left card corner.
- `4`: original bottom-right card corner.
- `5`: a point created by frustum clipping or occlusion carving.

Holo tags are `none`, `full`, `holo`, or `reverse`.

The bounding values are min/max corners, not YOLO `cx cy w h`. Flags are semantic point identities, not keypoint visibility values. Variable and concave polygons are not valid Ultralytics YOLO-pose labels.

Implementation:

- Geometry: `labeltools/occlusion.py`
- Refraction math: `labeltools/refraction.py`
- Serialization: `labeltools/yolo_pose.py::PolyLabel`
- Blender integration: `blender/labeling.py::label_scene`
- Visualization: `labeltools/visualize.py`

Current representation limits:

- Only one polygon ring is serialized.
- Interior holes are represented with a zero-width bridge.
- If carving creates disconnected components, only the largest is retained.
- Occluder ordering uses mean projected depth, not per-pixel depth.

## Legacy Pose Format

`CardLabel`, `write_label_file()`, and `write_dataset_yaml()` implement the earlier fixed-corner pose path:

```text
class cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4 |card_id|holo_tag
```

This path remains for Phase 1 regression tests. Do not combine `PolyLabel` files with its `dataset.yaml`.
