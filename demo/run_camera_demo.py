# Run from tcgDataSynth/: python demo/run_camera_demo.py
"""Identify cards from a camera using the exported YOLO and embedding models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_WEIGHTS = MODELS_DIR / "yolo26n-seg-best.pt"
EMBEDDING_WEIGHTS = MODELS_DIR / "mobilenet_v3_small_card_embedding.pt"
GALLERY_EMBEDDINGS = MODELS_DIR / "card_gallery_embeddings_512.npy"
GALLERY_IDS = MODELS_DIR / "card_gallery_ids.json"
CARD_METADATA = MODELS_DIR / "card_metadata.json"

CAMERA_DEVICE = "0"
YOLO_CONFIDENCE_THRESHOLD = 0.65
EMBEDDING_SIMILARITY_THRESHOLD = 0.65
CARD_CROP_WIDTH = 160
CARD_CROP_HEIGHT = 224
IMAGE_SIZE = 224
EMBEDDING_DIMENSION = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class MobileNetSmallEmbedding(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        # The saved checkpoint supplies all learned weights; do not download ImageNet weights.
        base = mobilenet_v3_small(weights=None)
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


def center_square(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


def approximate_quad(polygon: np.ndarray) -> np.ndarray | None:
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, True)
    for epsilon_fraction in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1):
        approximation = cv2.approxPolyDP(contour, epsilon_fraction * perimeter, True).reshape(-1, 2)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            return approximation.astype(np.float32)
    return None


def polygon_area(points: np.ndarray) -> float:
    return float(np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1)))


def order_quad(quad: np.ndarray) -> np.ndarray:
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    ordered = quad[np.argsort(angles)]
    # Image-coordinate clockwise order matches the target TL, TR, BR, BL order.
    if polygon_area(ordered) < 0:
        ordered = ordered[::-1]
    ordered = np.roll(ordered, -np.argmin(ordered.sum(axis=1)), axis=0)
    first_edge = np.linalg.norm(ordered[1] - ordered[0])
    second_edge = np.linalg.norm(ordered[2] - ordered[1])
    if first_edge > second_edge:
        ordered = np.roll(ordered, -1, axis=0)
    return ordered.astype(np.float32)


def rectify_card(frame: np.ndarray, polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    quad = approximate_quad(polygon)
    if quad is None:
        return None
    source = order_quad(quad)
    destination = np.array([
        [0, 0],
        [CARD_CROP_WIDTH - 1, 0],
        [CARD_CROP_WIDTH - 1, CARD_CROP_HEIGHT - 1],
        [0, CARD_CROP_HEIGHT - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    card = cv2.warpPerspective(
        frame,
        transform,
        (CARD_CROP_WIDTH, CARD_CROP_HEIGHT),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return card, quad


def load_embedding_model() -> MobileNetSmallEmbedding:
    checkpoint = torch.load(EMBEDDING_WEIGHTS, map_location=DEVICE, weights_only=True)
    training_card_ids = checkpoint.get("training_card_ids")
    if not isinstance(training_card_ids, list) or not training_card_ids:
        raise ValueError(f"Invalid embedding checkpoint: {EMBEDDING_WEIGHTS}")
    if checkpoint.get("embedding_dimension") != EMBEDDING_DIMENSION:
        raise ValueError(f"Unexpected embedding dimension in {EMBEDDING_WEIGHTS}")
    model = MobileNetSmallEmbedding(len(training_card_ids)).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def load_gallery() -> tuple[np.ndarray, list[str], dict[str, dict[str, str]]]:
    gallery_embeddings = np.load(GALLERY_EMBEDDINGS).astype(np.float32)
    gallery_ids = json.loads(GALLERY_IDS.read_text(encoding="utf-8"))
    metadata = json.loads(CARD_METADATA.read_text(encoding="utf-8"))
    if gallery_embeddings.ndim != 2 or gallery_embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(f"Invalid gallery embedding shape: {gallery_embeddings.shape}")
    if len(gallery_embeddings) != len(gallery_ids):
        raise ValueError("Gallery embedding and ID counts differ.")
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid card metadata file: {CARD_METADATA}")
    gallery_embeddings /= np.linalg.norm(gallery_embeddings, axis=1, keepdims=True).clip(min=1e-12)
    return gallery_embeddings, gallery_ids, metadata


def closest_card(model: MobileNetSmallEmbedding, card: np.ndarray, gallery_embeddings: np.ndarray) -> tuple[int, float]:
    # Test all four 90-degree orientations because segmentation alone has no pose head.
    rotations = [np.ascontiguousarray(np.rot90(card, rotation)) for rotation in range(4)]
    tensors = []
    for rotated_card in rotations:
        rgb = cv2.cvtColor(rotated_card, cv2.COLOR_BGR2RGB)
        tensors.append(EVAL_TRANSFORM(Image.fromarray(rgb)))
    images = torch.stack(tensors).to(DEVICE, non_blocking=True)
    with torch.inference_mode():
        embeddings = model.embed(images).float().cpu().numpy()
    similarities = embeddings @ gallery_embeddings.T
    rotation_index, gallery_index = np.unravel_index(np.argmax(similarities), similarities.shape)
    return int(gallery_index), float(similarities[rotation_index, gallery_index])


def draw_label(frame: np.ndarray, polygon: np.ndarray, text: str, accepted: bool) -> None:
    color = (0, 220, 0) if accepted else (0, 180, 255)
    points = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
    if not accepted:
        return
    x = int(points[:, 0, 0].min())
    y = max(20, int(points[:, 0, 1].min()) - 8)
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - text_height - 6), (x + text_width + 6, y + 3), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def parse_camera_device(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify cards from a webcam or OpenCV camera source.")
    parser.add_argument("--camera", default=CAMERA_DEVICE, help="Camera index, device path, or OpenCV stream URL.")
    parser.add_argument("--yolo-confidence", type=float, default=YOLO_CONFIDENCE_THRESHOLD)
    parser.add_argument("--embedding-similarity", type=float, default=EMBEDDING_SIMILARITY_THRESHOLD)
    args = parser.parse_args()

    required_paths = (YOLO_WEIGHTS, EMBEDDING_WEIGHTS, GALLERY_EMBEDDINGS, GALLERY_IDS, CARD_METADATA)
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing demo assets: {missing_paths}. Run demo/build_card_metadata.py after training.")

    detector = YOLO(YOLO_WEIGHTS)
    embedding_model = load_embedding_model()
    gallery_embeddings, gallery_ids, metadata = load_gallery()
    camera = cv2.VideoCapture(parse_camera_device(args.camera))
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera source: {args.camera}")

    window_name = "TCG Card Demo (press q to quit)"
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera source.")
            preview = center_square(frame)
            result = detector(preview, conf=args.yolo_confidence, verbose=False)[0]
            if result.boxes is not None and result.masks is not None:
                confidences = result.boxes.conf.detach().cpu().numpy()
                for polygon, confidence in zip(result.masks.xy, confidences):
                    if float(confidence) < args.yolo_confidence:
                        continue
                    polygon = np.asarray(polygon, dtype=np.float32)
                    rectified = rectify_card(preview, polygon)
                    if rectified is None:
                        draw_label(preview, polygon, "", False)
                        continue
                    card, quad = rectified
                    gallery_index, similarity = closest_card(embedding_model, card, gallery_embeddings)
                    accepted = similarity >= args.embedding_similarity
                    if accepted:
                        card_id = gallery_ids[gallery_index]
                        card_data = metadata.get(card_id, {})
                        text = " | ".join([
                            str(card_data.get("name", card_id)),
                            str(card_data.get("set", "Unknown set")),
                            f"#{card_data.get('localId', '?')}",
                        ])
                    else:
                        text = ""
                    draw_label(preview, quad, text, accepted)

            cv2.imshow(window_name, preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
