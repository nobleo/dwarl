"""Obstacles on the reference path that are not on the map.

The reference path is planned on the map, so obstacles that the map does not contain lie on it: the
robot only sees them in its lidar, like a box left in an aisle. Training places them at random on the
path of a share of episodes (never in test layouts); test cases place their own, deterministically.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from dwarl.datasets.generation.geometry import rasterize
from dwarl.datasets.generation.mapdata import BOX, CYLINDER
from dwarl.navigation.options import SURPRISE_OBSTACLES as DEFAULTS  # noqa: F401  (re-exported)
from dwarl.navigation.options import surprise_obstacles as settings  # noqa: F401


# ------------------------------------------------------------------------------------------- geometry
def dilate(grid: np.ndarray, cells: int) -> np.ndarray:
    out = grid.copy()
    for dr in range(-cells, cells + 1):
        for dc in range(-cells, cells + 1):
            if dr * dr + dc * dc > cells * cells:
                continue
            shifted = np.zeros_like(grid)
            shifted[max(dr, 0):grid.shape[0] + min(dr, 0), max(dc, 0):grid.shape[1] + min(dc, 0)] = \
                grid[max(-dr, 0):grid.shape[0] + min(-dr, 0), max(-dc, 0):grid.shape[1] + min(-dc, 0)]
            out |= shifted
    return out


def reachable(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    """4-connected reachability of ``goal`` from ``start`` through free cells."""
    if blocked[start] or blocked[goal]:
        return False
    try:
        from scipy.ndimage import label
    except ImportError:  # pragma: no cover - scipy ships with the training stack
        return _reachable_bfs(blocked, start, goal)
    labels, _ = label(~blocked)  # default structure: 4-connectivity
    return bool(labels[start] == labels[goal])


def _reachable_bfs(blocked, start, goal):
    seen = np.zeros_like(blocked)
    seen[start] = True
    queue = deque([start])
    rows, cols = blocked.shape
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and not blocked[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return False


def half_width(robot: dict) -> float:
    footprint = robot.get("footprint")
    return max(abs(float(y)) for _, y in footprint) if footprint else float(robot["footprint_radius"])


# ------------------------------------------------------------------------------------------- placement
def sample_on_path(md, path: np.ndarray, robot: dict, rng: np.random.Generator, count: int,
                   s: dict | None = None) -> np.ndarray:
    """Random obstacles (≤ count, (M, 6)) on ``path`` that leave the goal reachable around them.

    Obstacles avoid mapped geometry, start and goal, and the goal must stay reachable around them for
    a disc of the robot's turning radius (circumscribed footprint radius, at least half its width plus
    ``margin``), so the robot can turn anywhere on the detour. Where the mapped route itself is tighter,
    no more room than the map leaves is required.
    """
    s = settings(s)
    path = np.asarray(path, np.float64)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    keepout, length = float(s["keepout"]), float(arc[-1])
    if count <= 0 or length < 2 * keepout + 0.3:
        return np.zeros((0, 6), np.float32)
    n, res = md.occupancy.shape[0], md.resolution
    width, radius_c = half_width(robot), float(robot["footprint_radius"])
    start, goal = (tuple(md.xy_to_cell(p).tolist()) for p in (path[0], path[-1]))
    mapped = md.occupancy
    cells = int(np.ceil(max(radius_c, width + float(s["margin"])) / res))
    while cells > 0 and not reachable(dilate(mapped, cells), start, goal):
        cells -= 1  # tight start or goal: require no more room than the mapped route has
    placed: list[np.ndarray] = []
    for _ in range(count):
        for _ in range(int(s["tries"])):
            along = rng.uniform(keepout, length - keepout)
            i = int(np.clip(np.searchsorted(arc, along) - 1, 0, len(path) - 2))
            tangent = (path[i + 1] - path[i]) / max(steps[i], 1e-9)
            radius = rng.uniform(*map(float, s["size"]))
            offset = rng.uniform(-1.0, 1.0) * float(s["lateral"]) * (radius + width)
            centre = path[i] + tangent * (along - arc[i]) + np.array([-tangent[1], tangent[0]]) * offset
            if rng.random() < 0.5:
                prim = np.array([CYLINDER, *centre, 0.0, radius, radius], np.float32)
            else:
                prim = np.array([BOX, *centre, rng.uniform(0, np.pi), radius, radius * rng.uniform(0.4, 1.0)],
                                np.float32)
            if min(np.linalg.norm(centre - path[0]), np.linalg.norm(centre - path[-1])) < keepout + radius:
                continue
            if any(np.linalg.norm(centre - other[1:3]) < float(s["spacing"]) for other in placed):
                continue
            footprint = rasterize(prim[None], n, res, md.size)
            if (footprint & mapped).any():
                continue
            occupied = mapped | rasterize(np.stack([*placed, prim]), n, res, md.size)
            if not reachable(dilate(occupied, cells), start, goal):
                continue
            placed.append(prim)
            break
    return np.stack(placed).astype(np.float32) if placed else np.zeros((0, 6), np.float32)


# ------------------------------------------------------------------------------------------- runtime
class SurpriseObstacles:
    """Per-environment obstacles that are not on the map."""

    def __init__(self, cfg, env):
        self.env = env
        self.s = settings(cfg)
        self.enabled = float(self.s["probability"]) > 0.0
        n = env.num_envs
        self.max_count = int(self.s["count"][1]) if self.enabled else 0
        self.capacity = max(self.max_count, 1)
        self.prims = np.zeros((n, self.capacity, 6))
        self.valid = np.zeros((n, self.capacity), bool)
        self.fixed: dict[int, np.ndarray] = {}  # explicit obstacles for test cases
        robot = {"footprint": env.footprint.tolist() if env.footprint is not None else None,
                 "footprint_radius": env.radius}
        self.robot = robot

    @property
    def active(self) -> bool:
        return self.enabled or bool(self.fixed)

    def set_fixed(self, env_id: int, prims) -> None:
        """Use these obstacles (tile-local [type, x, y, yaw, hx, hy]) for ``env_id``'s next episodes."""
        prims = np.asarray(prims, np.float64).reshape(-1, 6)
        if len(prims) > self.capacity:
            self.capacity = len(prims)
            n = self.env.num_envs
            self.prims = np.concatenate([self.prims, np.zeros((n, self.capacity - self.prims.shape[1], 6))], 1)
            self.valid = np.concatenate([self.valid, np.zeros((n, self.capacity - self.valid.shape[1]), bool)], 1)
        if len(prims):
            self.fixed[int(env_id)] = prims
        else:
            self.fixed.pop(int(env_id), None)

    def sample(self, env_id: int, md, path: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Choose this episode's obstacles for ``env_id``; returns them (M, 6)."""
        if int(env_id) in self.fixed:
            prims = self.fixed[int(env_id)]
        elif self.enabled and rng.random() < float(self.s["probability"]):
            lo, hi = (int(v) for v in self.s["count"])
            prims = sample_on_path(md, path, self.robot, rng, int(rng.integers(lo, hi + 1)), self.s)
        else:
            prims = np.zeros((0, 6))
        self.prims[env_id] = 0.0
        self.valid[env_id] = False
        self.prims[env_id, :len(prims)] = prims
        self.valid[env_id, :len(prims)] = True
        return np.asarray(prims, np.float32)

    def obstacles(self, env_id: int) -> np.ndarray:
        return self.prims[env_id, self.valid[env_id]].astype(np.float32)

