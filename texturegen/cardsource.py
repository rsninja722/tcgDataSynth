"""
Card image discovery & selection (bpy-FREE, Docker-testable).

Given a root folder (the user's Pokémon images dir), recursively find candidate
card face images, and deterministically pick one from a seeded RNG. The filename
stem is the card ID used in labels. Also resolves each card's normalized
picture-region coordinates (per-card input with a project default).

Nothing here imports bpy; Blender-side card_factory calls select_card() with the
scene RNG and then loads the returned path as an image texture.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Import as a top-level module when the project root is on sys.path, else fall
# back to a relative import when used as a package.
try:
    import config
except ImportError:  # pragma: no cover - package-style import
    from .. import config  # type: ignore


@dataclass(frozen=True)
class CardImage:
    """One discoverable card face image and its label identity."""
    path: str          # absolute or root-relative path to the image file
    card_id: str       # filename stem, used verbatim in YOLO label |<card_id>
    picture_region: Tuple[float, float, float, float]  # (x0,y0,x1,y1) normalized, top-down


def _is_card_image(filename: str, exts: Sequence[str]) -> bool:
    if os.path.splitext(filename)[1].lower() not in exts:
        return False
    # The generic card-back texture is not a selectable card face.
    if filename.lower() == config.BACK_IMAGE_FILENAME.lower():
        return False
    return True


def discover_card_paths(
    root: Optional[str] = None,
    exts: Sequence[str] = config.CARD_IMAGE_EXTS,
) -> List[str]:
    """Recursively collect card image paths under ``root``, sorted for determinism.

    Sorting matters: selection is seed-driven, so the candidate order must be
    stable across runs/machines for a seed to reproduce the same card.
    """
    root = root or config.card_image_root()
    found: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if _is_card_image(fn, exts):
                found.append(os.path.join(dirpath, fn))
    # Sort by path so the ordering is deterministic and platform-independent.
    found.sort()
    return found


def card_id_from_path(path: str) -> str:
    """Filename without extension = card ID (spec §3.1)."""
    return os.path.splitext(os.path.basename(path))[0]


def load_picture_regions(
    search_dirs: Sequence[str],
    filename: str = config.PICTURE_REGIONS_FILENAME,
) -> Dict[str, Tuple[float, float, float, float]]:
    """Load an optional {card_id: [x0,y0,x1,y1]} override map.

    Looks for ``filename`` in each of ``search_dirs`` (first hit wins). Missing
    file => empty map => every card uses the default region. Malformed entries
    are skipped rather than crashing a long generation run.
    """
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            out: Dict[str, Tuple[float, float, float, float]] = {}
            for cid, region in raw.items():
                try:
                    x0, y0, x1, y1 = (float(v) for v in region)
                    out[str(cid)] = (x0, y0, x1, y1)
                except (TypeError, ValueError):
                    continue  # skip malformed entry, keep going
            return out
    return {}


def resolve_region(
    card_id: str,
    overrides: Dict[str, Tuple[float, float, float, float]],
    default: Tuple[float, float, float, float] = config.DEFAULT_PICTURE_REGION,
) -> Tuple[float, float, float, float]:
    """Per-card region: override map if present, else the project default."""
    return overrides.get(card_id, default)


class CardLibrary:
    """Discovered card pool + region overrides, ready for seed-driven selection.

    Build once per generation run (discovery walks the disk); then call
    ``select(rng)`` per card instance with the scene's seeded RNG.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        exts: Sequence[str] = config.CARD_IMAGE_EXTS,
        region_search_dirs: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = root or config.card_image_root()
        self.paths: List[str] = discover_card_paths(self.root, exts)
        search = list(region_search_dirs) if region_search_dirs else [self.root, os.getcwd()]
        self.region_overrides = load_picture_regions(search)

    def __len__(self) -> int:
        return len(self.paths)

    def is_empty(self) -> bool:
        return len(self.paths) == 0

    def card_for_path(self, path: str) -> CardImage:
        cid = card_id_from_path(path)
        return CardImage(path=path, card_id=cid,
                         picture_region=resolve_region(cid, self.region_overrides))

    def select(self, rng) -> CardImage:
        """Pick one card using a numpy Generator (or anything with .integers).

        Raises if the library is empty so a misconfigured path fails loudly at
        setup rather than producing unlabeled black renders.
        """
        if self.is_empty():
            raise RuntimeError(
                f"No card images found under {self.root!r}. Check the path / "
                f"TCG_CARD_IMAGE_ROOT env var and that it contains "
                f"{config.CARD_IMAGE_EXTS} files."
            )
        idx = int(rng.integers(0, len(self.paths)))
        return self.card_for_path(self.paths[idx])
