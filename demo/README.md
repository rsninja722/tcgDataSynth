# Card Camera Demo

Run these commands from `tcgDataSynth/` after the YOLO and embedding training scripts have exported their model artifacts:

```bash
python demo/build_card_metadata.py
python demo/run_camera_demo.py
```

The metadata builder writes `models/card_metadata.json` from every JSON file beneath `../input/cardData`. Use `--card-data-root` to select another source directory.

The camera demo center-crops each frame to a square, draws each accepted segmentation polygon, rectifies the card by approximating that polygon to a quadrilateral, and retrieves the closest normalized gallery embedding. Press `q` to close it.

Configure the default display thresholds in `run_camera_demo.py` or pass them at launch:

```bash
python demo/run_camera_demo.py --camera 0 --yolo-confidence 0.65 --embedding-similarity 0.65
```

`--camera` accepts an integer camera index, a device path, or an OpenCV stream URL. A predicted name, set, and local number is shown only when both thresholds are met.
