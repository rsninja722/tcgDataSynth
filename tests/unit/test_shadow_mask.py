"""Unit tests for deterministic simplex lighting shadow masks."""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from texturegen import shadow_mask as S  # noqa: E402


def test_face_centers_use_top_left_normalized_coordinates():
    u, v = S.face_sample_coordinates()
    assert u.shape == (50, 50) and v.shape == (50, 50)
    assert np.isclose(u[0, 0], 0.01) and np.isclose(v[0, 0], 0.01)
    assert np.isclose(u[-1, -1], 0.99) and np.isclose(v[-1, -1], 0.99)
    assert np.all(np.diff(u[0]) > 0.0)
    assert np.all(np.diff(v[:, 0]) > 0.0)


def test_shadow_brightness_is_deterministic_normalized_and_seeded():
    first = S.shadow_brightness(12345)
    second = S.shadow_brightness(12345)
    other = S.shadow_brightness(12346)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert first.shape == (50, 50)
    assert np.all(np.isfinite(first))
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0


def test_mask_threshold_is_exactly_fifty_percent():
    brightness = S.shadow_brightness(9)
    expected = brightness <= 0.50
    assert np.array_equal(S.retained_faces(9), expected)


def test_both_simplex_scales_affect_combined_brightness():
    u, v = S.face_sample_coordinates()
    seed = 8723
    coarse = S.simplex_noise_2d(u, v, S.COARSE_FREQUENCY, seed) * 0.5 + 0.5
    fine = S.simplex_noise_2d(u, v, S.FINE_FREQUENCY,
                              seed ^ 0x6A09E667F3BCC909) * 0.5 + 0.5
    combined = S.shadow_brightness(seed)
    assert np.allclose(combined, 0.65 * coarse + 0.35 * fine)
    assert not np.array_equal(combined, coarse)
    assert not np.array_equal(combined, fine)


def test_unit_grid_has_51_squared_vertices_and_positive_winding():
    retained = np.zeros((50, 50), dtype=bool)
    retained[0, 0] = True
    retained[-1, -1] = True
    vertices, faces = S.unit_grid_geometry(retained)
    assert vertices.shape == (51 * 51, 3)
    assert len(faces) == 2
    assert np.allclose(vertices[0], (-0.5, 0.5, 0.0))
    first = vertices[np.asarray(faces[0])]
    normal = np.cross(first[1] - first[0], first[2] - first[0])
    assert normal[2] > 0.0


def test_seed_audit_produces_nonempty_nonfull_planes():
    counts = np.asarray([np.count_nonzero(S.retained_faces(seed)) for seed in range(100)])
    assert np.all(counts > 0)
    assert np.all(counts < 2500)
    assert len(set(int(value) for value in counts)) > 20


def _run_all():
    tests = [value for key, value in sorted(globals().items())
             if key.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
