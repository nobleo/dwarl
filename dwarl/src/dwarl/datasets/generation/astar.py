"""Grid A* planner with clearance-aware costs, plus path post-processing."""

from __future__ import annotations

import heapq

import numpy as np

_SQRT2 = float(np.sqrt(2.0))
_MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2)]  # fmt: skip


def astar(
    free: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    cell_cost: np.ndarray | None = None,
) -> list[tuple[int, int]] | None:
    """8-connected A* on ``free`` (bool grid). ``cell_cost`` (>= 1) multiplies the step cost of
    entering a cell, which lets the planner prefer cells far from obstacles. Diagonal moves are
    only allowed when both adjacent orthogonal cells are free (no corner cutting).
    Returns a list of (row, col) cells from start to goal, or None when unreachable.
    """
    rows, cols = free.shape
    if not (free[start] and free[goal]):
        return None
    cost = np.ones_like(free, dtype=np.float64) if cell_cost is None else cell_cost
    g = np.full(free.shape, np.inf)
    parent = np.full(free.shape + (2,), -1, dtype=np.int32)
    closed = np.zeros(free.shape, dtype=bool)
    gr, gc = goal

    def h(r: int, c: int) -> float:  # octile distance, admissible because cost >= 1
        dr, dc = abs(r - gr), abs(c - gc)
        return (dr + dc) + (_SQRT2 - 2.0) * min(dr, dc)

    g[start] = 0.0
    heap = [(h(*start), 0.0, start[0], start[1])]
    while heap:
        _, gcur, r, c = heapq.heappop(heap)
        if closed[r, c]:
            continue
        closed[r, c] = True
        if (r, c) == goal:
            path = [(r, c)]
            while (r, c) != start:
                r, c = parent[r, c]
                path.append((int(r), int(c)))
            return path[::-1]
        for dr, dc, step in _MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or not free[nr, nc] or closed[nr, nc]:
                continue
            if dr and dc and not (free[r + dr, c] and free[r, c + dc]):
                continue
            ng = gcur + step * cost[nr, nc]
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                parent[nr, nc] = (r, c)
                heapq.heappush(heap, (ng + h(nr, nc), ng, nr, nc))
    return None


def smooth_path(points: np.ndarray, iterations: int = 3, window: int = 5) -> np.ndarray:
    """Moving-average smoothing that keeps endpoints fixed (removes grid staircase artefacts)."""
    if len(points) <= window:
        return points
    p = points.copy()
    k = np.ones(window) / window
    pad = window // 2
    for _ in range(iterations):
        ext = np.concatenate([np.repeat(p[:1], pad, 0), p, np.repeat(p[-1:], pad, 0)], 0)
        q = np.stack([np.convolve(ext[:, i], k, mode="valid") for i in range(2)], -1)
        q[0], q[-1] = points[0], points[-1]
        p = q
    return p


def resample_path(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline at (approximately) constant arc-length ``spacing``."""
    if len(points) < 2:
        return points.astype(np.float32)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(2, int(np.ceil(total / spacing)) + 1)
    t = np.linspace(0.0, total, n)
    return np.stack([np.interp(t, s, points[:, 0]), np.interp(t, s, points[:, 1])], -1).astype(np.float32)


def path_is_clear(path: np.ndarray, free: np.ndarray, size: float, resolution: float) -> bool:
    """True when every linearly-interpolated segment of ``path`` stays in the tile and in ``free``.

    ``free`` is the inflated free-space bool grid; ``size``/``resolution`` define the tile-to-cell
    mapping (identical to ``MapData.xy_to_cell``). Check every crossed cell,
    including cells touched at a grid edge or corner.
    """
    path = np.asarray(path, dtype=np.float64)
    half = size / 2.0
    rows, cols = free.shape
    if not len(path) or not np.isfinite(path).all() or (np.abs(path) >= half).any():
        return False
    grid = (path + half) / resolution
    for a, b in zip(grid, np.concatenate([grid[1:], grid[-1:]]), strict=True):
        delta = b - a
        times = [np.array([0., 1.])]
        for axis in range(2):
            if delta[axis] != 0:
                lo, hi = sorted((a[axis], b[axis]))
                boundaries = np.arange(np.ceil(lo), np.floor(hi) + 1)
                times.append((boundaries - a[axis]) / delta[axis])
        t = np.unique(np.clip(np.concatenate(times), 0., 1.))
        points = a + np.concatenate([t, (t[:-1] + t[1:]) / 2])[:, None] * delta
        # At exact boundaries both neighboring cells count (four at corners).
        rounded = np.rint(points)
        points = np.where(np.isclose(points, rounded, atol=1e-10, rtol=0), rounded, points)
        low, high = np.ceil(points).astype(int) - 1, np.floor(points).astype(int)
        for c in (low[:, 0], high[:, 0]):
            for r in (low[:, 1], high[:, 1]):
                if ((c < 0) | (c >= cols) | (r < 0) | (r >= rows)).any() or not free[r, c].all():
                    return False
    return True


def finalize_path(raw: np.ndarray, free: np.ndarray, size: float, resolution: float,
                  spacing: float) -> np.ndarray | None:
    """Prefer a smooth/resampled route; return None if even the raw route is invalid."""
    candidates = (resample_path(smooth_path(raw), spacing), resample_path(raw, spacing),
                  np.asarray(raw, dtype=np.float32))
    for path in candidates:
        if path_is_clear(path, free, size, resolution):
            return path
    return None
