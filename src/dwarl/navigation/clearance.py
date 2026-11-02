"""Planar polygon distances and footprint/primitive clearance, using only NumPy."""

import numpy as np


def polygon_distance(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance of points (..., 2) to a convex/simple polygon's boundary, 0 inside."""
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip((((pts[..., None, :] - a) * ab).sum(-1)) / np.maximum((ab**2).sum(-1), 1e-12), 0, 1)
    d = np.linalg.norm(pts[..., None, :] - (a + t[..., None] * ab), axis=-1).min(-1)
    x, y = pts[..., 0, None], pts[..., 1, None]
    dy = np.where(ab[:, 1] == 0, 1e-12, ab[:, 1])
    cross = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < ab[:, 0] * (y - a[:, 1]) / dy + a[:, 0])
    inside = cross.sum(-1) % 2 == 1
    return np.where(inside, 0.0, d)


def point_polygon(points, polygon):
    return polygon_distance(np.asarray(points), polygon)


def _overlap_edges(a, b):
    aa, bb = np.roll(a, -1, axis=0) - a, np.roll(b, -1, axis=0) - b
    rel = b[None] - a[:, None]
    def cross(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    denom = cross(aa[:, None], bb[None])
    valid = np.abs(denom) > 1e-12
    safe = np.where(valid, denom, 1.)
    t = cross(rel, bb[None]) / safe
    u = cross(rel, aa[:, None]) / safe
    return np.any(valid & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1))


def footprint_clearance(xy, yaw, radius, footprint, primitives):
    primitives = np.asarray(primitives).reshape(-1, 6)
    if not len(primitives):
        return np.inf
    # Vectorized conservative broad phase keeps native-resolution building maps
    # practical. Exact polygon work is restricted to possible nearest objects.
    delta = np.asarray(xy) - primitives[:, 1:3]
    co, sn = np.cos(primitives[:, 3]), np.sin(primitives[:, 3])
    local = np.stack([co * delta[:, 0] + sn * delta[:, 1],
                      -sn * delta[:, 0] + co * delta[:, 1]], -1)
    box_gap = np.linalg.norm(np.maximum(np.abs(local) - primitives[:, 4:6], 0.), axis=-1)
    lower = np.where(primitives[:, 0] == 1, np.linalg.norm(delta, axis=-1) - primitives[:, 4], box_gap)
    lower = np.maximum(0., lower - radius)
    primitives = primitives[lower <= lower.min() + 2. * radius + 1e-9]
    c, s = np.cos(yaw), np.sin(yaw)
    robot = None if footprint is None else np.asarray(footprint) @ np.array([[c, s], [-s, c]]) + xy
    minimum = np.inf
    for kind, x, y, angle, hx, hy in primitives:
        center = np.array([x, y])
        if int(kind) == 1:
            distance = np.linalg.norm(xy - center) - radius - hx if robot is None else (
                point_polygon(center[None], robot)[0] - hx)
        else:
            ca, sa = np.cos(angle), np.sin(angle)
            box = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
            box = box @ np.array([[ca, sa], [-sa, ca]]) + center
            if robot is None:
                distance = point_polygon(np.asarray(xy)[None], box)[0] - radius
            else:
                distance = 0. if _overlap_edges(robot, box) else min(
                    point_polygon(robot, box).min(), point_polygon(box, robot).min())
        minimum = min(minimum, distance)
    return max(0., float(minimum))


def path_width_clear(
    path: np.ndarray,
    footprint,
    primitives: np.ndarray,
    eps: float = 1e-6,
) -> bool:
    """Check a disk of the footprint's lateral half-width along every path segment.

    This is the maze/rooms width approximation, not full-footprint turning feasibility.
    Distances are to the true primitives rather than rasterized cells.
    """
    path = np.asarray(path, dtype=np.float64)
    if not len(path):
        return True
    half_width = max(abs(y) for _, y in footprint)
    primitives = np.asarray(primitives).reshape(-1, 6)
    if not len(primitives):
        return True
    centers, extents = primitives[:, 1:3], primitives[:, 4:6]
    co, sn = np.cos(primitives[:, 3]), np.sin(primitives[:, 3])
    boxes = primitives[:, 0] == 0
    corners = extents[:, None] * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])

    def local(point):
        delta = point - centers
        return np.stack([co * delta[:, 0] + sn * delta[:, 1],
                         -sn * delta[:, 0] + co * delta[:, 1]], -1)

    for start, end in zip(path, np.concatenate([path[1:], path[-1:]]), strict=True):
        a, b = local(start), local(end)
        direction = b - a
        length2 = np.maximum((direction**2).sum(-1), 1e-30)
        t = np.clip(-(a * direction).sum(-1) / length2, 0., 1.)
        distances = np.linalg.norm(a + t[:, None] * direction, axis=-1) - extents[:, 0]
        # Segment-to-box distance: endpoint-to-box or corner-to-segment,
        # unless the segment intersects the box (slab intersection).
        endpoint = np.minimum(np.linalg.norm(np.maximum(np.abs(a) - extents, 0.), axis=-1),
                              np.linalg.norm(np.maximum(np.abs(b) - extents, 0.), axis=-1))
        t = np.clip(((corners - a[:, None]) * direction[:, None]).sum(-1) / length2[:, None], 0., 1.)
        corner = np.linalg.norm(corners - (a[:, None] + t[..., None] * direction[:, None]), axis=-1).min(-1)
        parallel = np.abs(direction) < 1e-15
        safe = np.where(parallel, 1., direction)
        t0, t1 = (-extents - a) / safe, (extents - a) / safe
        enter = np.where(parallel, -np.inf, np.minimum(t0, t1)).max(-1)
        leave = np.where(parallel, np.inf, np.maximum(t0, t1)).min(-1)
        intersects = (np.maximum(enter, 0.) <= np.minimum(leave, 1.)) & ~(
            parallel & (np.abs(a) > extents)).any(-1)
        distances[boxes] = np.where(intersects, 0., np.minimum(endpoint, corner))[boxes]
        if np.any(distances <= half_width + eps):
            return False
    return True
