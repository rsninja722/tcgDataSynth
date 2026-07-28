"""Idealized refraction for apparent card-corner projection.

The renderer's roughness, scratches, and normal maps scatter a neighborhood of rays.
Labels use the geometric-normal Snell ray as an approximation of that neighborhood's
centroid.  Refractors are finite oriented boxes so lid/slab side exits are retained.
This module is bpy-free; Blender only converts marked objects into ``RefractiveBox``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


IOR_PROPERTY = "label_refractive_ior"
BOUNDS_MIN_PROPERTY = "label_refractive_bounds_min"
BOUNDS_MAX_PROPERTY = "label_refractive_bounds_max"

_EPS = 1e-9


class RefractionError(RuntimeError):
    """Raised when an apparent ray cannot be solved without inventing a fallback."""


class TotalInternalReflectionError(RefractionError):
    """Raised when one optical trial has no transmitted Snell ray."""


def _vec3(value: Sequence[float], name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (3,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain three finite values")
    return out


def _unit(value: Sequence[float], name: str) -> np.ndarray:
    out = _vec3(value, name)
    length = float(np.linalg.norm(out))
    if length <= _EPS:
        raise ValueError(f"{name} must be non-zero")
    return out / length


@dataclass
class RefractiveBox:
    """Finite homogeneous optical box in world space.

    ``axes`` are its three orthonormal local axes in world space and ``half_sizes``
    are the corresponding world-space half extents.
    """

    center: Sequence[float]
    axes: Sequence[Sequence[float]]
    half_sizes: Sequence[float]
    ior: float = 1.5
    name: str = "refractor"

    def __post_init__(self) -> None:
        self.center = _vec3(self.center, "center")
        axes = np.asarray(self.axes, dtype=np.float64)
        if axes.shape != (3, 3) or not np.all(np.isfinite(axes)):
            raise ValueError("axes must be a finite 3x3 array")
        axes = np.stack([_unit(axis, "axis") for axis in axes])
        if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-7):
            raise ValueError("axes must be orthonormal")
        self.axes = axes
        self.half_sizes = _vec3(self.half_sizes, "half_sizes")
        if np.any(self.half_sizes <= 0.0):
            raise ValueError("half_sizes must be positive")
        self.ior = float(self.ior)
        if not np.isfinite(self.ior) or self.ior <= 0.0:
            raise ValueError("ior must be positive and finite")


@dataclass(frozen=True)
class TraceResult:
    hit: np.ndarray
    direction: np.ndarray
    crossed: Tuple[str, ...]


def refract(direction: Sequence[float], incident_normal: Sequence[float],
            n_from: float, n_to: float) -> np.ndarray:
    """Return the ideal transmitted direction using Snell's law.

    ``incident_normal`` points into the medium containing the incident ray.  Total
    internal reflection has no transmitted centroid and raises ``RefractionError``.
    """
    d = _unit(direction, "direction")
    normal = _unit(incident_normal, "incident_normal")
    cos_i = -float(np.dot(normal, d))
    if cos_i < 0.0:
        normal = -normal
        cos_i = -cos_i
    eta = float(n_from) / float(n_to)
    discriminant = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    if discriminant < -1e-12:
        raise TotalInternalReflectionError(
            "total internal reflection has no transmitted ray")
    out = eta * d + (eta * cos_i - np.sqrt(max(0.0, discriminant))) * normal
    return _unit(out, "refracted direction")


def _contains(box: RefractiveBox, point: np.ndarray, tol: float = 1e-10) -> bool:
    local = box.axes @ (point - box.center)
    return bool(np.all(np.abs(local) < box.half_sizes - tol))


def _box_interval(box: RefractiveBox, origin: np.ndarray, direction: np.ndarray):
    local_o = box.axes @ (origin - box.center)
    local_d = box.axes @ direction
    t_enter, t_exit = -np.inf, np.inf
    enter_normal = exit_normal = None
    for i in range(3):
        if abs(local_d[i]) <= _EPS:
            if abs(local_o[i]) > box.half_sizes[i]:
                return None
            continue
        t_neg = (-box.half_sizes[i] - local_o[i]) / local_d[i]
        t_pos = (+box.half_sizes[i] - local_o[i]) / local_d[i]
        n_neg, n_pos = -box.axes[i], box.axes[i]
        if t_neg > t_pos:
            t_neg, t_pos = t_pos, t_neg
            n_neg, n_pos = n_pos, n_neg
        if enter_normal is not None and abs(t_neg - t_enter) <= 10.0 * _EPS:
            raise RefractionError(f"ray hits sharp edge of refractor {box.name!r}")
        if exit_normal is not None and abs(t_pos - t_exit) <= 10.0 * _EPS:
            raise RefractionError(f"ray hits sharp edge of refractor {box.name!r}")
        if t_neg > t_enter:
            t_enter, enter_normal = float(t_neg), n_neg
        if t_pos < t_exit:
            t_exit, exit_normal = float(t_pos), n_pos
        if t_enter > t_exit + _EPS:
            return None
    if enter_normal is None or exit_normal is None:
        return None
    return t_enter, t_exit, enter_normal, exit_normal


def ray_plane_intersection(origin: Sequence[float], direction: Sequence[float],
                           plane_point: Sequence[float], plane_normal: Sequence[float]):
    """Return the positive ray parameter for a plane, or ``None`` if not ahead."""
    o = _vec3(origin, "origin")
    d = _unit(direction, "direction")
    p = _vec3(plane_point, "plane_point")
    n = _unit(plane_normal, "plane_normal")
    denom = float(np.dot(d, n))
    if abs(denom) <= _EPS:
        return None
    distance = float(np.dot(p - o, n) / denom)
    return distance if distance > _EPS else None


def trace_to_plane(origin: Sequence[float], direction: Sequence[float],
                   plane_point: Sequence[float], plane_normal: Sequence[float],
                   boxes: Sequence[RefractiveBox] = ()) -> TraceResult:
    """Trace through finite boxes until the ray reaches the target plane."""
    pos = _vec3(origin, "origin").copy()
    ray = _unit(direction, "direction")
    target = _vec3(plane_point, "plane_point")
    normal = _unit(plane_normal, "plane_normal")
    crossed = []

    for _ in range(2 * len(boxes) + 3):
        active = [box for box in boxes if _contains(box, pos)]
        if len(active) > 1:
            names = ", ".join(box.name for box in active)
            raise RefractionError(f"overlapping refractive boxes are unsupported: {names}")
        target_t = ray_plane_intersection(pos, ray, target, normal)
        if target_t is None:
            raise RefractionError("candidate ray does not reach the target plane")

        event = None
        for box in boxes:
            interval = _box_interval(box, pos, ray)
            if interval is None:
                continue
            t_enter, t_exit, enter_normal, exit_normal = interval
            inside = any(box is active_box for active_box in active)
            if inside:
                distance, outward, entering = t_exit, exit_normal, False
            else:
                distance, outward, entering = t_enter, enter_normal, True
            if distance <= _EPS or distance >= target_t - _EPS:
                continue
            if event is not None and abs(distance - event[0]) <= 10.0 * _EPS:
                raise RefractionError(
                    f"coincident refractive boundaries are unsupported: "
                    f"{event[1].name}, {box.name}"
                )
            if event is None or distance < event[0]:
                event = (distance, box, outward, entering)

        if event is None or target_t <= event[0] + _EPS:
            return TraceResult(pos + ray * target_t, ray, tuple(crossed))

        distance, box, outward, entering = event
        boundary = pos + ray * distance
        if entering:
            if active:
                raise RefractionError(
                    f"overlapping refractive boxes are unsupported: "
                    f"{active[0].name}, {box.name}"
                )
            ray = refract(ray, outward, 1.0, box.ior)
            crossed.append(box.name + ":enter")
        else:
            ray = refract(ray, -outward, box.ior, 1.0)
            crossed.append(box.name + ":exit")
        pos = boundary + ray * (10.0 * _EPS)

    raise RefractionError("optical trace exceeded the expected number of boundaries")


def solve_camera_ray(camera_origin: Sequence[float], target: Sequence[float],
                     target_x_axis: Sequence[float], target_y_axis: Sequence[float],
                     boxes: Sequence[RefractiveBox] = (), tolerance: float = 1e-7,
                     max_iterations: int = 12) -> np.ndarray:
    """Find the camera ray whose refracted path reaches one target point.

    The two unknowns are small angular offsets around the unrefracted camera ray.
    Newton steps use finite differences of the hit residual in the target plane.
    """
    camera = _vec3(camera_origin, "camera_origin")
    point = _vec3(target, "target")
    x_axis = _unit(target_x_axis, "target_x_axis")
    y_axis = _unit(target_y_axis, "target_y_axis")
    plane_normal = _unit(np.cross(x_axis, y_axis), "target plane normal")
    direct = _unit(point - camera, "camera-to-target direction")

    try:
        initial = trace_to_plane(camera, direct, point, plane_normal, boxes)
    except TotalInternalReflectionError:
        # A direct ray near a finite-box edge can enter one face and encounter total
        # internal reflection at another. That invalid trial does not rule out a
        # neighboring transmitted branch which enters/exits a different face pair.
        initial = None
    if initial is not None and not initial.crossed:
        return direct

    reference = np.array((0.0, 0.0, 1.0))
    if abs(float(np.dot(reference, direct))) > 0.9:
        reference = np.array((0.0, 1.0, 0.0))
    tangent_x = _unit(np.cross(reference, direct), "ray tangent x")
    tangent_y = _unit(np.cross(direct, tangent_x), "ray tangent y")
    offset = np.zeros(2, dtype=np.float64)

    def evaluate(value):
        candidate = _unit(direct + value[0] * tangent_x + value[1] * tangent_y,
                          "candidate direction")
        traced = trace_to_plane(camera, candidate, point, plane_normal, boxes)
        delta = traced.hit - point
        residual = np.array((np.dot(delta, x_axis), np.dot(delta, y_axis)))
        return candidate, residual

    # A sharp finite-box edge separates front-entry and side-entry solution branches.
    # Seed Newton on the best nearby branch instead of assuming the unrefracted ray's
    # branch contains the apparent solution.
    best_offset = None
    _best_candidate = best_residual = None
    best_size = math.inf
    try:
        _best_candidate, best_residual = evaluate(offset)
    except RefractionError:
        pass
    else:
        best_offset = offset
        best_size = float(np.linalg.norm(best_residual))
        if float(np.linalg.norm(best_residual, ord=np.inf)) <= tolerance:
            return _best_candidate

    compass_directions = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    if initial is None:
        seed_directions = tuple(
            (math.cos(index * math.pi / 8.0), math.sin(index * math.pi / 8.0))
            for index in range(16))
        seed_radii = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
    else:
        seed_directions = compass_directions
        seed_radii = (0.0025, 0.005, 0.01, 0.02)
    for radius in seed_radii:
        for seed_direction in seed_directions:
            seed = np.asarray(seed_direction, dtype=np.float64)
            seed *= radius / float(np.linalg.norm(seed))
            try:
                _seed_candidate, seed_residual = evaluate(seed)
            except RefractionError:
                continue
            seed_size = float(np.linalg.norm(seed_residual))
            if seed_size < best_size:
                best_offset, best_size = seed, seed_size
                _best_candidate, best_residual = _seed_candidate, seed_residual
    if best_offset is None:
        raise RefractionError(
            "no transmitted camera-ray branch reaches the target plane")
    if float(np.linalg.norm(best_residual, ord=np.inf)) <= tolerance:
        return _best_candidate
    offset = best_offset

    for _ in range(max_iterations):
        candidate, residual = evaluate(offset)
        residual_size = float(np.linalg.norm(residual))
        if float(np.linalg.norm(residual, ord=np.inf)) <= tolerance:
            return candidate
        jacobian = np.empty((2, 2), dtype=np.float64)
        for axis in range(2):
            derivative = None
            for step in (1e-5, 2e-6, 5e-7):
                shifted = offset.copy()
                shifted[axis] += step
                try:
                    _candidate, shifted_residual = evaluate(shifted)
                except RefractionError:
                    shifted[axis] -= 2.0 * step
                    try:
                        _candidate, shifted_residual = evaluate(shifted)
                    except RefractionError:
                        continue
                    derivative = (residual - shifted_residual) / step
                else:
                    derivative = (shifted_residual - residual) / step
                break
            if derivative is None:
                raise RefractionError("could not sample the apparent-ray Jacobian")
            jacobian[:, axis] = derivative
        try:
            correction = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError as exc:
            raise RefractionError("apparent-ray solver has a singular Jacobian") from exc
        length = float(np.linalg.norm(correction))
        if length > 0.05:
            correction *= 0.05 / length
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            trial = offset - correction * scale
            try:
                _trial_candidate, trial_residual = evaluate(trial)
            except RefractionError:
                continue
            if float(np.linalg.norm(trial_residual)) < residual_size:
                offset = trial
                accepted = True
                break
        if not accepted:
            # Near a finite box edge, one finite-difference sample can cross onto a
            # different face and make the raw Newton direction unreliable. A damped
            # least-squares direction remains a descent direction on the current face.
            normal_matrix = jacobian.T @ jacobian
            rhs = jacobian.T @ residual
            matrix_scale = max(float(np.trace(normal_matrix)) / 2.0, 1e-12)
            for damping in (1e-6, 1e-4, 1e-2, 1.0, 100.0):
                try:
                    damped = np.linalg.solve(
                        normal_matrix + np.eye(2) * matrix_scale * damping, rhs)
                except np.linalg.LinAlgError:
                    continue
                damped_length = float(np.linalg.norm(damped))
                if damped_length > 0.02:
                    damped *= 0.02 / damped_length
                for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    trial = offset - damped * scale
                    try:
                        _trial_candidate, trial_residual = evaluate(trial)
                    except RefractionError:
                        continue
                    if float(np.linalg.norm(trial_residual)) < residual_size:
                        offset = trial
                        accepted = True
                        break
                if accepted:
                    break
        if not accepted:
            # Last resort at a branch boundary: a shrinking compass search can cross
            # the discontinuity without relying on a derivative sampled on one face.
            pattern_offset = best_offset.copy()
            _pattern_candidate, pattern_residual = evaluate(pattern_offset)
            pattern_size = float(np.linalg.norm(pattern_residual))
            pattern_step = 0.005
            for _pattern_iteration in range(40):
                improved = False
                for pattern_direction in compass_directions:
                    direction_2d = np.asarray(pattern_direction, dtype=np.float64)
                    direction_2d /= float(np.linalg.norm(direction_2d))
                    trial = pattern_offset + direction_2d * pattern_step
                    try:
                        trial_candidate, trial_residual = evaluate(trial)
                    except RefractionError:
                        continue
                    trial_size = float(np.linalg.norm(trial_residual))
                    if trial_size < pattern_size:
                        pattern_offset = trial
                        pattern_residual = trial_residual
                        pattern_size = trial_size
                        improved = True
                        if float(np.linalg.norm(trial_residual, ord=np.inf)) <= tolerance:
                            return trial_candidate
                if not improved:
                    pattern_step *= 0.5
                    if pattern_step < 1e-8:
                        break
            if pattern_size < residual_size:
                offset = pattern_offset
                continue
            raise RefractionError("apparent-ray solver could not find a decreasing step")

    _candidate, residual = evaluate(offset)
    raise RefractionError(
        f"apparent-ray solver did not converge; residual={np.linalg.norm(residual):.3g}m"
    )
