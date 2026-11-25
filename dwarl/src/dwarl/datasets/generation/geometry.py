"""Rasterisation of primitives and rectangle decomposition of occupancy grids."""

from __future__ import annotations

import numpy as np

from dwarl.datasets.generation.mapdata import BOX, CYLINDER


def cell_centers(n: int, res: float, size: float) -> tuple[np.ndarray, np.ndarray]:
    coords = -size / 2.0 + (np.arange(n) + 0.5) * res
    xs, ys = np.meshgrid(coords, coords)  # xs[r, c] = x of column c, ys[r, c] = y of row r
    return xs, ys


def rasterize(primitives: np.ndarray, n: int, res: float, size: float) -> np.ndarray:
    """Return an (n, n) bool grid with cells whose centre lies inside any primitive.

    Primitives are grown by half a cell so thin walls are never lost.
    """
    xs, ys = cell_centers(n, res, size)
    occ = np.zeros((n, n), dtype=bool)
    pad = res * 0.5
    for t, x, y, yaw, hx, hy in primitives:
        if int(t) == CYLINDER:
            occ |= (xs - x) ** 2 + (ys - y) ** 2 <= (hx + pad) ** 2
        else:
            c, s = np.cos(yaw), np.sin(yaw)
            dx, dy = xs - x, ys - y
            lx = c * dx + s * dy
            ly = -s * dx + c * dy
            occ |= (np.abs(lx) <= hx + pad) & (np.abs(ly) <= hy + pad)
    return occ


def grid_to_rects(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Greedy decomposition of True cells into axis-aligned rectangles (r0, c0, r1, c1), exclusive end."""
    g = grid.copy()
    rows, cols = g.shape
    rects = []
    for r in range(rows):
        c = 0
        while c < cols:
            if not g[r, c]:
                c += 1
                continue
            c1 = c
            while c1 < cols and g[r, c1]:
                c1 += 1
            r1 = r + 1
            while r1 < rows and g[r1, c:c1].all():
                r1 += 1
            g[r:r1, c:c1] = False
            rects.append((r, c, r1, c1))
            c = c1
    return rects


def rects_to_primitives(rects, res: float, size: float) -> np.ndarray:
    half = size / 2.0
    out = []
    for r0, c0, r1, c1 in rects:
        x = -half + (c0 + c1) * 0.5 * res
        y = -half + (r0 + r1) * 0.5 * res
        out.append([BOX, x, y, 0.0, (c1 - c0) * 0.5 * res, (r1 - r0) * 0.5 * res])
    return np.asarray(out, dtype=np.float32).reshape(-1, 6)


def grid_to_primitives(grid: np.ndarray, res: float, size: float) -> np.ndarray:
    return rects_to_primitives(grid_to_rects(grid), res, size)


def border_primitives(size: float, thickness: float) -> np.ndarray:
    h = size / 2.0
    t = thickness / 2.0
    return np.array(
        [
            [BOX, 0.0, h - t, 0.0, h, t],
            [BOX, 0.0, -h + t, 0.0, h, t],
            [BOX, h - t, 0.0, 0.0, t, h - thickness],
            [BOX, -h + t, 0.0, 0.0, t, h - thickness],
        ],
        dtype=np.float32,
    )
