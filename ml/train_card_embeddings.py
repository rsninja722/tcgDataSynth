# Run from tcgDataSynth/: python ml/train_card_embeddings.py
"""Rectify generated card crops, train MobileNetV3-small, and export gallery embeddings."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path
import json
import multiprocessing
import os
import shutil
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


ML_DIR = PROJECT_ROOT / "ml"
SOURCE_IMAGES = PROJECT_ROOT / "out" / "images"
SOURCE_LABELS = PROJECT_ROOT / "out" / "labels_yolo"
CUSTOM_LABELS = PROJECT_ROOT / "out" / "labels"
EXTRA_LABELS = PROJECT_ROOT / "out" / "extra_label"
CROP_DIR = ML_DIR / "card_crops"
CROP_MANIFEST = ML_DIR / "card_crop_manifest.json"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CARD_CROP_WIDTH = 160
CARD_CROP_HEIGHT = 224
IMAGE_SIZE = 224
EMBEDDING_DIMENSION = 512
EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
TRAIN_BATCH_SIZE = 128
INFERENCE_BATCH_SIZE = 256
DATA_LOADER_WORKERS = min(4, os.cpu_count() or 1)
SEED = 20260826
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.12, p=0.45),
    transforms.RandomApply([transforms.ColorJitter(0.22, 0.22, 0.12, 0.04)], p=0.8),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class CardEmbeddingDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(image)), label


class MobileNetSmallEmbedding(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = base.features
        self.pool = base.avgpool
        self.projection = nn.Sequential(
            nn.Linear(576, EMBEDDING_DIMENSION),
            nn.BatchNorm1d(EMBEDDING_DIMENSION),
        )
        self.classifier = nn.Linear(EMBEDDING_DIMENSION, class_count)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        values = self.pool(self.features(images)).flatten(1)
        return nn.functional.normalize(self.projection(values), dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(images))


def polygon_quad(yolo_row: str, image_width: int, image_height: int) -> tuple[np.ndarray, float] | None:
    fields = yolo_row.split()
    if not fields or fields[0] != "0":
        raise ValueError("Expected a class-0 YOLO segmentation row.")
    values = np.asarray([float(value) for value in fields[1:]], dtype=np.float32)
    if values.size < 6 or values.size % 2:
        raise ValueError("Invalid YOLO segmentation polygon.")

    points = np.clip(values.reshape(-1, 2), 0.0, 1.0)
    points[:, 0] *= image_width - 1
    points[:, 1] *= image_height - 1
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    for epsilon_fraction in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1):
        approximation = cv2.approxPolyDP(contour, epsilon_fraction * perimeter, True).reshape(-1, 2)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            return approximation.astype(np.float32), float(cv2.contourArea(contour))
    return None


def flagged_corners(custom_row: str, image_width: int, image_height: int) -> dict[int, np.ndarray]:
    fields = custom_row.split("|", 1)[0].split()
    if len(fields) < 14 or (len(fields) - 5) % 3:
        raise ValueError("Invalid custom polygon label.")

    corners = {}
    for offset in range(5, len(fields), 3):
        flag = int(float(fields[offset + 2]))
        if flag in (1, 2, 3, 4):
            corners[flag] = np.array([
                float(fields[offset]) * (image_width - 1),
                float(fields[offset + 1]) * (image_height - 1),
            ], dtype=np.float32)
    return corners


def orient_quad(quad: np.ndarray, corners: dict[int, np.ndarray]) -> np.ndarray:
    # Custom flags map the upright card frame to TL, TR, BR, BL.
    expected = np.stack([corners[1], corners[2], corners[4], corners[3]])
    ordering = min(
        permutations(range(4)),
        key=lambda candidate: sum(np.linalg.norm(quad[candidate[i]] - expected[i]) for i in range(4)),
    )
    return quad[list(ordering)]


def extract_crops() -> list[dict[str, object]]:
    if CROP_DIR.is_dir() and CROP_MANIFEST.is_file():
        try:
            existing_records = json.loads(CROP_MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_records = []
        if (
            isinstance(existing_records, list)
            and existing_records
            and all(
                isinstance(record, dict)
                and isinstance(record.get("crop_path"), str)
                and Path(record["crop_path"]).is_file()
                for record in existing_records
            )
        ):
            print(f"Reusing {len(existing_records)} existing rectified card crops from {CROP_DIR}.")
            return existing_records

    for source_dir in (SOURCE_IMAGES, SOURCE_LABELS, CUSTOM_LABELS, EXTRA_LABELS):
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing generated dataset directory: {source_dir}")

    if CROP_DIR.exists():
        shutil.rmtree(CROP_DIR)
    CROP_DIR.mkdir(parents=True)

    destination = np.array([
        [0, 0],
        [CARD_CROP_WIDTH - 1, 0],
        [CARD_CROP_WIDTH - 1, CARD_CROP_HEIGHT - 1],
        [0, CARD_CROP_HEIGHT - 1],
    ], dtype=np.float32)
    crop_records: list[dict[str, object]] = []
    skipped_instances = 0
    image_paths = sorted(
        path for path in SOURCE_IMAGES.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for image_path in image_paths:
        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"
        custom_label_path = CUSTOM_LABELS / f"{image_path.stem}.txt"
        extra_label_path = EXTRA_LABELS / f"{image_path.stem}.txt"
        if not label_path.is_file() or not custom_label_path.is_file() or not extra_label_path.is_file():
            raise FileNotFoundError(f"Missing segmentation, custom, or identity labels for {image_path.name}")

        yolo_rows = label_path.read_text(encoding="utf-8").splitlines()
        custom_rows = custom_label_path.read_text(encoding="utf-8").splitlines()
        identity_rows = extra_label_path.read_text(encoding="utf-8").splitlines()
        if len(yolo_rows) != len(custom_rows) or len(yolo_rows) != len(identity_rows):
            raise ValueError(f"Segmentation/custom/identity row counts differ for {image_path.name}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read generated image: {image_path}")
        height, width = image.shape[:2]
        for index, (yolo_row, custom_row, identity_row) in enumerate(zip(yolo_rows, custom_rows, identity_rows)):
            card_id = identity_row.split("|", 1)[0]
            custom_card_id = custom_row.split("|", 2)[1].strip() if "|" in custom_row else ""
            if not card_id or custom_card_id != card_id:
                raise ValueError(f"Custom/identity card IDs differ for {image_path.name}, instance {index}")

            corners = flagged_corners(custom_row, width, height)
            quad_and_area = polygon_quad(yolo_row, width, height)
            if not {1, 2, 3, 4}.issubset(corners) or quad_and_area is None:
                skipped_instances += 1
                continue

            quad, mask_area = quad_and_area
            transform = cv2.getPerspectiveTransform(orient_quad(quad, corners), destination)
            crop = cv2.warpPerspective(
                image,
                transform,
                (CARD_CROP_WIDTH, CARD_CROP_HEIGHT),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            crop_path = CROP_DIR / f"{image_path.stem}_{index:02d}.png"
            if not cv2.imwrite(str(crop_path), crop):
                raise OSError(f"Could not write crop: {crop_path}")
            crop_records.append({
                "card_id": card_id,
                "crop_path": str(crop_path.resolve()),
                "mask_area": mask_area,
            })

    if not crop_records:
        raise ValueError("No card instances were available for embedding training.")
    crop_records.sort(key=lambda record: float(record["mask_area"]), reverse=True)
    keep_count = max(1, (len(crop_records) + 1) // 2)
    for record in crop_records[keep_count:]:
        Path(str(record["crop_path"])).unlink()
    crop_records = crop_records[:keep_count]
    CROP_MANIFEST.write_text(json.dumps(crop_records, indent=2) + "\n", encoding="utf-8")
    print(
        f"Kept {len(crop_records)} largest rectified card crops for "
        f"{len({str(record['card_id']) for record in crop_records})} card IDs; "
        f"skipped {skipped_instances} partial/non-quadrilateral masks and excluded the smallest half by mask area."
    )
    return crop_records


def discover_gallery() -> tuple[list[Path], dict[str, Path]]:
    card_image_root = Path(config.card_image_root())
    if not card_image_root.is_dir():
        raise FileNotFoundError(f"card_image_root from config.json does not exist: {card_image_root}")

    gallery_paths = sorted(
        path for path in card_image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.name.casefold() != "back.png"
    )
    if not gallery_paths:
        raise ValueError(f"No card images found under {card_image_root}")

    gallery_by_id = {}
    gallery_keys = set()
    for path in gallery_paths:
        key = path.stem.casefold()
        if key in gallery_keys:
            raise ValueError(f"Duplicate card ID in card_image_root: {path.stem}")
        gallery_keys.add(key)
        gallery_by_id[path.stem] = path
    return gallery_paths, gallery_by_id


def train_embedding_model(crop_records: list[dict[str, object]], gallery_by_id: dict[str, Path]) -> tuple[MobileNetSmallEmbedding, list[str]]:
    training_ids = sorted({str(record["card_id"]) for record in crop_records} & set(gallery_by_id))
    if len(training_ids) < 2:
        raise ValueError("At least two card IDs with both gallery images and generated crops are required.")

    missing_gallery_ids = sorted({str(record["card_id"]) for record in crop_records} - set(gallery_by_id))
    if missing_gallery_ids:
        print(f"Skipping {len(missing_gallery_ids)} generated IDs not found in card_image_root.")
    print(f"Training identities: {len(training_ids)}")

    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    class_indices = {card_id: index for index, card_id in enumerate(training_ids)}
    training_samples = []
    for record in crop_records:
        card_id = str(record["card_id"])
        if card_id in class_indices:
            training_samples.append((gallery_by_id[card_id], class_indices[card_id]))
            training_samples.append((Path(str(record["crop_path"])), class_indices[card_id]))

    train_loader = DataLoader(
        CardEmbeddingDataset(training_samples, TRAIN_TRANSFORM),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=DATA_LOADER_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=DATA_LOADER_WORKERS > 0,
    )
    model = MobileNetSmallEmbedding(len(training_ids)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler("cuda", enabled=DEVICE.type == "cuda")
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct = total = 0
        loss_sum = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=DEVICE.type == "cuda"):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
            loss_sum += float(loss.detach()) * len(labels)

        mean_loss = loss_sum / total
        print(f"Epoch {epoch:02d}/{EPOCHS}: train loss={mean_loss:.4f}, train accuracy={correct / total:.4f}")
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping after {PATIENCE} epochs without a lower training loss.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), training_ids


def write_artifacts(model: MobileNetSmallEmbedding, training_ids: list[str], gallery_paths: list[Path]) -> None:
    gallery_loader = DataLoader(
        CardEmbeddingDataset([(path, 0) for path in gallery_paths], EVAL_TRANSFORM),
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=DATA_LOADER_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=DATA_LOADER_WORKERS > 0,
    )
    gallery_vectors = []
    with torch.inference_mode():
        for images, _ in gallery_loader:
            images = images.to(DEVICE, non_blocking=True)
            with autocast("cuda", enabled=DEVICE.type == "cuda"):
                gallery_vectors.append(model.embed(images).float().cpu().numpy())
    gallery_embeddings = np.concatenate(gallery_vectors).astype(np.float32)

    embedding_weights_path = MODELS_DIR / "mobilenet_v3_small_card_embedding.pt"
    gallery_embeddings_path = MODELS_DIR / "card_gallery_embeddings_512.npy"
    gallery_ids_path = MODELS_DIR / "card_gallery_ids.json"
    torch.save({
        "model": "mobilenet_v3_small",
        "state_dict": model.state_dict(),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "image_size": IMAGE_SIZE,
        "training_card_ids": training_ids,
    }, embedding_weights_path)
    np.save(gallery_embeddings_path, gallery_embeddings)
    gallery_ids_path.write_text(json.dumps([path.stem for path in gallery_paths], indent=2) + "\n", encoding="utf-8")
    print(f"Weights: {embedding_weights_path}")
    print(f"Gallery embeddings: {gallery_embeddings_path} with shape {gallery_embeddings.shape}")
    print(f"Gallery IDs: {gallery_ids_path}")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Embedding device: {DEVICE}; data loader workers: {DATA_LOADER_WORKERS}")
    crop_records = extract_crops()
    gallery_paths, gallery_by_id = discover_gallery()
    print(f"Gallery: {len(gallery_paths)} images")
    model, training_ids = train_embedding_model(crop_records, gallery_by_id)
    write_artifacts(model, training_ids, gallery_paths)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
