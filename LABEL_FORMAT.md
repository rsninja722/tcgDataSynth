# Label Formats

## Active Custom Polygon Format

Current Phase 4 layout scripts emit one line per card:

```text
<class> <x_min> <y_min> <x_max> <y_max> (<x> <y> <flag>)+ |<card_id>|<holo_tag>
```

Coordinates are normalized to `[0, 1]` with a top-left origin. Polygon points are perimeter-ordered and may be concave.

Classes:

- `0`: the card's four ideal corners are in the camera frustum.
- `1`: only part of the card intersects the camera frustum.

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
