"""Procedural map generators.

Every generator has the signature ``fn(rng, size, **params) -> primitives`` and returns the
interior obstacles (the border wall is added by the pipeline). Parameters are given as
``[low, high]`` ranges and sampled per map for diversity.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage

from dwarl.datasets.generation.geometry import grid_to_primitives
from dwarl.datasets.generation.mapdata import BOX, CYLINDER


def _u(rng: np.random.Generator, rng_range) -> float:
    lo, hi = rng_range
    return float(rng.uniform(lo, hi))


# ---------------------------------------------------------------------------- clutter
def clutter(
    rng: np.random.Generator,
    size: float,
    density=(0.03, 0.2),
    box_half=(0.1, 0.6),
    cyl_radius=(0.08, 0.4),
    cylinder_prob: float = 0.4,
    max_obstacles: int = 120,
) -> np.ndarray:
    """Random rotated boxes and cylinders until the target area fraction is reached."""
    target = _u(rng, density) * size * size
    area, prims = 0.0, []
    half = size / 2.0
    while area < target and len(prims) < max_obstacles:
        x, y = rng.uniform(-half, half, 2)
        if rng.random() < cylinder_prob:
            r = _u(rng, cyl_radius)
            prims.append([CYLINDER, x, y, 0.0, r, r])
            area += np.pi * r * r
        else:
            hx, hy = _u(rng, box_half), _u(rng, box_half)
            if rng.random() < 0.3:  # occasional wall-like slab
                hx, hy = _u(rng, (0.6, 1.5)), 0.05
            prims.append([BOX, x, y, rng.uniform(-np.pi, np.pi), hx, hy])
            area += 4 * hx * hy
    return np.asarray(prims, np.float32).reshape(-1, 6)


# ---------------------------------------------------------------------------- maze
def maze(
    rng: np.random.Generator,
    size: float,
    corridor=(1.0, 1.8),
    wall_thickness: float = 0.1,
    braid=(0.05, 0.35),
    plaza=0,
) -> np.ndarray:
    """Recursive-backtracker maze with variable corridor widths; ``braid`` removes extra walls (loops).

    Row heights and column widths are sampled independently from ``corridor`` and rescaled so they
    tile ``size`` exactly, giving hallways of different widths within a single map. ``plaza`` (a count
    or [lo, hi] range) merges that many 2x2 cell blocks into open areas, giving rectangular-footprint
    robots -- whose tight corridors are too narrow to host a random-heading spawn -- places to start.
    """
    lo, hi = float(corridor[0]), float(corridor[1])

    def spans() -> np.ndarray:
        """Boundaries (length n+1, 0..size) of n cells with widths sampled from [lo, hi]."""
        widths = []
        total = 0.0
        while total < size:
            widths.append(rng.uniform(lo, hi))
            total += widths[-1]
        return np.concatenate([[0.0], np.cumsum(np.asarray(widths) * (size / total))])

    xb = spans()  # column boundaries, left -> right
    yb = spans()  # row boundaries, bottom -> top
    nr, nc = len(yb) - 1, len(xb) - 1
    # walls: horizontal h[r, c] is the wall below cell (r, c) (between row r and r+1), vertical v[r, c] right of it
    h = np.ones((nr - 1, nc), bool)
    v = np.ones((nr, nc - 1), bool)
    visited = np.zeros((nr, nc), bool)
    stack = [(int(rng.integers(nr)), int(rng.integers(nc)))]
    visited[stack[0]] = True
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nbrs = [(a, b) for a, b in nbrs if 0 <= a < nr and 0 <= b < nc and not visited[a, b]]
        if not nbrs:
            stack.pop()
            continue
        a, b = nbrs[rng.integers(len(nbrs))]
        if a != r:
            h[min(a, r), c] = False
        else:
            v[r, min(b, c)] = False
        visited[a, b] = True
        stack.append((a, b))
    p = _u(rng, braid)
    h &= rng.random(h.shape) > p
    v &= rng.random(v.shape) > p

    # plazas: merge 2x2 blocks of cells into open spawn areas (tight mazes otherwise have none)
    if plaza:
        if isinstance(plaza, (tuple, list)):
            n_plaza = int(rng.integers(int(plaza[0]), int(plaza[1]) + 1))
        else:
            n_plaza = int(plaza)
        for _ in range(n_plaza):
            r = int(rng.integers(0, nr - 1))
            c = int(rng.integers(0, nc - 1))
            h[r, c] = h[r, c + 1] = False
            v[r, c] = v[r + 1, c] = False

    half, t = size / 2.0, wall_thickness / 2.0
    prims = []
    # merge collinear wall segments into single boxes
    for r in range(nr - 1):
        y = -half + yb[r + 1]
        c = 0
        while c < nc:
            if h[r, c]:
                c1 = c
                while c1 < nc and h[r, c1]:
                    c1 += 1
                x0, x1 = -half + xb[c], -half + xb[c1]
                prims.append([BOX, (x0 + x1) / 2, y, 0.0, (x1 - x0) / 2 + t, t])
                c = c1
            else:
                c += 1
    for c in range(nc - 1):
        x = -half + xb[c + 1]
        r = 0
        while r < nr:
            if v[r, c]:
                r1 = r
                while r1 < nr and v[r1, c]:
                    r1 += 1
                y0, y1 = -half + yb[r], -half + yb[r1]
                prims.append([BOX, x, (y0 + y1) / 2, 0.0, t, (y1 - y0) / 2 + t])
                r = r1
            else:
                r += 1
    return np.asarray(prims, np.float32).reshape(-1, 6)


# ---------------------------------------------------------------------------- rooms
def rooms(
    rng: np.random.Generator,
    size: float,
    min_room=(1.8, 2.8),
    door_width=(0.8, 1.3),
    wall_thickness: float = 0.1,
    extra_doors_prob: float = 0.3,
    furniture=(0, 3),
) -> np.ndarray:
    """Recursive division into rooms, each dividing wall gets at least one door; optional furniture."""
    min_r = _u(rng, min_room)
    t = wall_thickness / 2.0
    prims: list[list[float]] = []
    leaves: list[tuple[float, float, float, float]] = []

    def wall_with_doors(horizontal: bool, fixed: float, lo: float, hi: float) -> None:
        length = hi - lo
        ndoor = 1 + int(rng.random() < extra_doors_prob)
        gaps = []
        for _ in range(ndoor):
            dw = min(_u(rng, door_width), length * 0.6)
            s = rng.uniform(lo, hi - dw)
            gaps.append((s, s + dw))
        gaps.sort()
        cur = lo
        for g0, g1 in gaps + [(hi, hi)]:
            if g0 - cur > 0.05:
                m, hl = (cur + g0) / 2, (g0 - cur) / 2
                prims.append([BOX, m, fixed, 0.0, hl, t] if horizontal else [BOX, fixed, m, 0.0, t, hl])
            cur = max(cur, g1)

    def divide(x0: float, y0: float, x1: float, y1: float, depth: int = 0) -> None:
        w, hgt = x1 - x0, y1 - y0
        can_v, can_h = w >= 2 * min_r, hgt >= 2 * min_r
        if not (can_v or can_h) or depth > 6:
            leaves.append((x0, y0, x1, y1))
            return
        vertical = can_v and (not can_h or (w > hgt if abs(w - hgt) > 0.5 else rng.random() < 0.5))
        if vertical:
            xs = rng.uniform(x0 + min_r, x1 - min_r)
            wall_with_doors(False, xs, y0, y1)
            divide(x0, y0, xs, y1, depth + 1)
            divide(xs, y0, x1, y1, depth + 1)
        else:
            ys = rng.uniform(y0 + min_r, y1 - min_r)
            wall_with_doors(True, ys, x0, x1)
            divide(x0, y0, x1, ys, depth + 1)
            divide(x0, ys, x1, y1, depth + 1)

    half = size / 2.0
    divide(-half, -half, half, half)
    lo, hi = furniture
    for x0, y0, x1, y1 in leaves:
        for _ in range(int(rng.integers(lo, hi + 1))):
            hx, hy = rng.uniform(0.15, 0.45, 2)
            # keep furniture against walls-ish but away from room centre corridors
            x = rng.uniform(x0 + hx + 0.1, max(x0 + hx + 0.11, x1 - hx - 0.1))
            y = rng.uniform(y0 + hy + 0.1, max(y0 + hy + 0.11, y1 - hy - 0.1))
            prims.append([BOX, x, y, 0.0, hx, hy])
    return np.asarray(prims, np.float32).reshape(-1, 6)


# ---------------------------------------------------------------------------- caves
def caves(
    rng: np.random.Generator,
    size: float,
    cell=(0.2, 0.3),
    fill=(0.40, 0.48),
    steps: int = 5,
    min_blob_cells: int = 3,
) -> np.ndarray:
    """Cellular-automata caves (B5678/S45678 rule) on a coarse grid, decomposed into boxes."""
    cs = _u(rng, cell)
    n = int(round(size / cs))
    cs = size / n
    g = rng.random((n, n)) < _u(rng, fill)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = True
    kernel = np.ones((3, 3), int)
    for _ in range(steps):
        nb = ndimage.convolve(g.astype(int), kernel, mode="constant", cval=1)
        g = nb >= 5
    # strip border ring (added by pipeline) and tiny specks
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = False
    lab, num = ndimage.label(g)
    if num:
        counts = np.bincount(lab.ravel())
        small = counts < min_blob_cells
        small[0] = False
        g[small[lab]] = False
    return grid_to_primitives(g, cs, size)


GENERATORS: dict[str, Callable[..., np.ndarray]] = {
    "clutter": clutter,
    "maze": maze,
    "rooms": rooms,
    "caves": caves,
}
