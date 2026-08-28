# Run from tcgDataSynth/: python ml/train_yolo26_seg.py
"""Stage generated segmentation data, train YOLO26n-seg, and export best weights."""

from pathlib import Path
import random
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"
SOURCE_IMAGES = PROJECT_ROOT / "out" / "images"
SOURCE_LABELS = PROJECT_ROOT / "out" / "labels_yolo"
STAGED_DATASET = ML_DIR / "dataset"
DATA_YAML = ML_DIR / "dataset.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def stage_dataset() -> None:
    for source_dir in (SOURCE_IMAGES, SOURCE_LABELS):
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing generated dataset directory: {source_dir}")

    if STAGED_DATASET.exists():
        shutil.rmtree(STAGED_DATASET)

    images = sorted(
        path for path in SOURCE_IMAGES.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(images) < 3:
        raise ValueError("At least three image/label pairs are required for train, val, and test splits.")

    missing_labels = [path.name for path in images if not (SOURCE_LABELS / f"{path.stem}.txt").is_file()]
    if missing_labels:
        raise FileNotFoundError(f"Missing YOLO labels for {len(missing_labels)} images: {missing_labels[:5]}")

    rng = random.Random(42)
    rng.shuffle(images)
    test_count = max(1, round(len(images) * 0.10))
    val_count = max(1, round(len(images) * 0.10))
    train_count = len(images) - val_count - test_count
    if train_count < 1:
        raise ValueError("Not enough image/label pairs to create all three splits.")

    splits = {
        "train": images[:train_count],
        "val": images[train_count:train_count + val_count],
        "test": images[train_count + val_count:],
    }
    for split_name, split_images in splits.items():
        image_dir = STAGED_DATASET / "images" / split_name
        label_dir = STAGED_DATASET / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image in split_images:
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(SOURCE_LABELS / f"{image.stem}.txt", label_dir / f"{image.stem}.txt")
        print(f"{split_name}: {len(split_images)} images")

    DATA_YAML.write_text(
        f"path: {STAGED_DATASET.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: card\n",
        encoding="utf-8",
    )


def main() -> None:
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stage_dataset()
    model = YOLO("yolo26n-seg.pt")
    results = model.train(
        data=str(DATA_YAML),
        project=str(ML_DIR / "runs"),
        name="yolo26n-seg",
        exist_ok=True,
        epochs=100,
        imgsz=320,
        save_period=10,
        cache=True,
        max_det=18,
        patience=20,
        single_cls=True,
    )
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training did not produce best weights: {best_weights}")

    final_weights = MODELS_DIR / "yolo26n-seg-best.pt"
    shutil.copy2(best_weights, final_weights)
    print(f"Final weights: {final_weights}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
