"""Bpy-free tests for idealized apparent-ray projection."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from labeltools.refraction import (RefractiveBox, RefractionError, refract,  # noqa: E402
                                   solve_camera_ray, trace_to_plane)


def _box(center=(0.0, 0.0, 0.0), half=(1.0, 1.0, 0.1), name="box"):
    return RefractiveBox(center, np.eye(3), half, 1.5, name)


def test_normal_incidence_is_unchanged():
    actual = refract((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), 1.0, 1.5)
    assert np.allclose(actual, (0.0, 0.0, -1.0), atol=1e-12)


def test_oblique_refraction_obeys_snell():
    incidence = math.radians(40.0)
    incoming = (math.sin(incidence), 0.0, -math.cos(incidence))
    actual = refract(incoming, (0.0, 0.0, 1.0), 1.0, 1.5)
    transmitted = math.asin(abs(actual[0]))
    assert abs(math.sin(incidence) - 1.5 * math.sin(transmitted)) < 1e-12
    assert transmitted < incidence


def test_total_internal_reflection_is_explicit():
    incidence = math.radians(60.0)
    incoming = (math.sin(incidence), 0.0, -math.cos(incidence))
    try:
        refract(incoming, (0.0, 0.0, 1.0), 1.5, 1.0)
    except RefractionError:
        pass
    else:
        raise AssertionError("expected total internal reflection")


def test_parallel_plate_restores_angle_and_adds_lateral_shift():
    thickness = 0.2
    incidence = math.radians(30.0)
    direction = np.array((math.sin(incidence), 0.0, -math.cos(incidence)))
    result = trace_to_plane((0.0, 0.0, 1.0), direction, (0.0, 0.0, -1.0),
                            (0.0, 0.0, 1.0), [_box(half=(1.0, 1.0, thickness / 2.0))])
    transmitted = math.asin(math.sin(incidence) / 1.5)
    expected_x = 2.0 * math.tan(incidence) + thickness * (
        math.tan(transmitted) - math.tan(incidence))
    assert result.crossed == ("box:enter", "box:exit")
    assert np.allclose(result.direction, direction, atol=1e-8)
    assert abs(result.hit[0] - expected_x) < 1e-7


def test_solver_reaches_target_through_parallel_plate():
    box = _box(half=(1.0, 1.0, 0.1), name="lid")
    camera, target = (0.0, 0.0, 1.0), (0.7, 0.1, -1.0)
    ray = solve_camera_ray(camera, target, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), [box])
    traced = trace_to_plane(camera, ray, target, (0.0, 0.0, 1.0), [box])
    assert np.allclose(traced.hit, target, atol=1e-7)
    assert traced.crossed == ("lid:enter", "lid:exit")


def test_target_inside_slab_stops_after_entry():
    slab = _box(half=(0.5, 0.5, 0.1), name="slab")
    camera, target = (0.0, 0.0, 1.0), (0.2, 0.0, 0.0)
    ray = solve_camera_ray(camera, target, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), [slab])
    traced = trace_to_plane(camera, ray, target, (0.0, 0.0, 1.0), [slab])
    assert np.allclose(traced.hit, target, atol=1e-7)
    assert traced.crossed == ("slab:enter",)


def test_stacked_lid_and_slab_compose():
    lid = _box(center=(0.0, 0.0, 0.55), half=(1.0, 1.0, 0.05), name="lid")
    slab = _box(center=(0.0, 0.0, 0.0), half=(0.6, 0.6, 0.1), name="slab")
    camera, target = (0.0, 0.0, 1.2), (0.25, 0.05, 0.0)
    ray = solve_camera_ray(camera, target, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                           [lid, slab])
    traced = trace_to_plane(camera, ray, target, (0.0, 0.0, 1.0), [lid, slab])
    assert np.allclose(traced.hit, target, atol=1e-7)
    assert traced.crossed == ("lid:enter", "lid:exit", "slab:enter")


def test_finite_box_miss_preserves_direct_ray():
    distant = _box(center=(10.0, 0.0, 0.0))
    camera, target = (0.0, 0.0, 1.0), (0.2, 0.0, -1.0)
    ray = solve_camera_ray(camera, target, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), [distant])
    direct = np.asarray(target) - np.asarray(camera)
    direct = direct / np.linalg.norm(direct)
    assert np.array_equal(ray, direct)


def test_grazing_side_entry_converges_without_branch_jump():
    slab = RefractiveBox((0.0, 0.0, 0.0), np.eye(3),
                         (0.04, 0.0675, 0.00335), 1.5, "slab")
    cases = [
        ((0.1614670131970429, 0.21847354447801884, 0.010816078008828828),
         (0.0003192726143474989, -0.0008864299864439415, 0.000225)),
        ((0.4506545916353599, 0.48180058913348645, 0.023057536419938247),
         (-0.028689886678068027, -0.009107690491883438, 0.000225)),
        ((0.6404174755295879, 0.43616953240687667, 0.34571740123664624),
         (0.029890058884223733, -0.006703279826177154, 0.000225)),
        ((0.628604039644353, 0.0733161616464152, 0.337455653302397),
         (0.030539025394979484, 0.0024254671375014206, 0.000225)),
    ]
    for camera, target in cases:
        ray = solve_camera_ray(camera, target, (1.0, 0.0, 0.0),
                               (0.0, 1.0, 0.0), [slab])
        traced = trace_to_plane(camera, ray, target, (0.0, 0.0, 1.0), [slab])
        assert np.allclose(traced.hit, target, atol=1e-7)


def test_solver_searches_neighbor_branch_when_direct_trial_has_tir():
    lid = _box(half=(0.5, 0.5, 0.05), name="display-case-lid")
    camera = (0.3631900856782384, -0.7564844164146461, 0.5795418936322998)
    target = (-0.6517616666704081, -0.3557103238427461, -0.3)
    direct = np.asarray(target) - np.asarray(camera)
    direct /= np.linalg.norm(direct)
    try:
        trace_to_plane(camera, direct, target, (0.0, 0.0, 1.0), [lid])
    except RefractionError as exc:
        assert "total internal reflection" in str(exc)
    else:
        raise AssertionError("regression geometry must make the direct trial undergo TIR")

    ray = solve_camera_ray(camera, target, (1.0, 0.0, 0.0),
                           (0.0, 1.0, 0.0), [lid])
    traced = trace_to_plane(camera, ray, target, (0.0, 0.0, 1.0), [lid])
    assert np.allclose(traced.hit, target, atol=1e-7)
    assert traced.crossed


def _run_all():
    tests = [value for key, value in sorted(globals().items())
             if key.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
