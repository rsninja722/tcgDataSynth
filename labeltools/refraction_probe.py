"""Small image-difference helper used by optical marker diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerDetection:
    x: float
    y: float
    area_px: int
    energy: float


def detect_marker(baseline, probe, expected_xy: Optional[Tuple[float, float]] = None,
                  search_radius_px: Optional[int] = None) -> Optional[MarkerDetection]:
    """Locate the strongest connected component in a linear-image difference."""
    before = np.asarray(baseline, dtype=np.float32)
    after = np.asarray(probe, dtype=np.float32)
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] < 3:
        raise ValueError("baseline and probe must be same-sized RGB/RGBA images")
    energy = np.abs(after[..., :3] - before[..., :3]).sum(axis=2)
    if expected_xy is not None and search_radius_px is not None:
        yy, xx = np.ogrid[:energy.shape[0], :energy.shape[1]]
        ex, ey = expected_xy
        keep = (xx - ex) ** 2 + (yy - ey) ** 2 <= float(search_radius_px) ** 2
        energy = np.where(keep, energy, 0.0)
    peak = float(energy.max(initial=0.0))
    if peak <= 0.0:
        return None
    mask = (energy >= max(peak * 0.05, 1e-8)).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    for label in range(1, count):
        component = labels == label
        component_energy = float(energy[component].sum())
        if best is None or component_energy > best[0]:
            weights = energy[component]
            ys, xs = np.nonzero(component)
            best = (component_energy,
                    float(np.average(xs, weights=weights)),
                    float(np.average(ys, weights=weights)),
                    int(stats[label, cv2.CC_STAT_AREA]))
    if best is None:
        return None
    return MarkerDetection(x=best[1], y=best[2], area_px=best[3], energy=best[0])
