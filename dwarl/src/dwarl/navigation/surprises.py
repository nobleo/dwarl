"""Obstacles on the reference path that are not on the map, and the avoidance signal they create.

The reference path is planned on the map, so obstacles that the map does not contain lie on it: the
robot only sees them in its lidar, like a box left in an aisle. Training places them at random on the
path of a share of episodes (never in test layouts) and measures, with privileged geometry, how much
the path ahead is blocked. That measure, the avoidance level, relaxes the path-adherence rewards:

* far from obstacles the level is 0 and the policy is rewarded for following the path closely;
* as the robot approaches a blocked stretch the level rises towards 1 and leaving the path costs only
  progress, while proximity and collision penalties keep full weight (avoidance takes precedence);
* once the obstacle is behind the robot the level decays over ``release_time``, so path adherence
  returns smoothly and the robot converges back onto the path.

In potential-field terms the obstacle switches the path's attraction off locally instead of adding a
repulsion of its own; repulsion comes from the proximity reward, which already sees every obstacle.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from dwarl.datasets.generation.geometry import rasterize
from dwarl.datasets.generation.mapdata import BOX, CYLINDER
from dwarl.navigation.options import SURPRISE_OBSTACLES as DEFAULTS  # noqa: F401  (re-exported)
from dwarl.navigation.options import surprise_obstacles as settings  # noqa: F401


# ------------------------------------------------------------------------------------------- geometry
def primitive_distance(points: np.ndarray, prims: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Signed distance from points (N, P, 2) to the nearest valid primitive surface of (N, K, 6).

    Negative inside an obstacle; ``inf`` where an environment has no valid primitive.
    """
    delta = points[:, :, None, :] - prims[:, None, :, 1:3]  # (N, P, K, 2)
    c, s = np.cos(prims[:, None, :, 3]), np.sin(prims[:, None, :, 3])
    local = np.stack([c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]], -1)
    half = prims[:, None, :, 4:6]
    q = np.abs(local) - half
    box = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(-1), 0.0)
    cylinder = np.linalg.norm(delta, axis=-1) - prims[:, None, :, 4]
    distance = np.where(prims[:, None, :, 0] == CYLINDER, cylinder, box)
    return np.where(valid[:, None, :], distance, np.inf).min(-1)


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


def obstacle_size(robot: dict) -> tuple[float, float]:
    """Radius or larger box half-extent, in metres."""
    width = half_width(robot)
    return .5 * width, .9 * width


class LocalDetours:
    """Require a short, clear excursion that rejoins the same stretch of reference path."""

    def __init__(self, md, path, arc, robot, margin):
        self.md, self.path, self.arc = md, path, arc
        width = half_width(robot)
        clearance = max(float(robot["footprint_radius"]), width + margin)
        self.cells = int(np.ceil(clearance / md.resolution))
        self.blocked = dilate(md.occupancy, self.cells)
        self.excursion = clearance + width + margin
        self.reserved = np.zeros_like(self.blocked)

    def accept(self, footprint, along, tangent):
        """Keep a collision-free bypass for this obstacle and preserve earlier bypasses."""
        added = dilate(footprint, self.cells)
        if (added & self.reserved).any():
            return False
        span = 2 * self.excursion
        lo, hi = max(0., along - span), min(self.arc[-1], along + span)
        samples = np.linspace(lo, hi, int(np.ceil((hi - lo) / (.25 * self.md.resolution))) + 1)
        route = np.column_stack([np.interp(samples, self.arc, axis) for axis in self.path.T])
        phase = (samples - along) / np.where(samples < along, along - lo, hi - along)
        bump = (.5 + .5 * np.cos(np.pi * phase))[:, None] * np.array([-tangent[1], tangent[0]])
        blocked = self.blocked | added
        for factor in (.5, -.5, .75, -.75, 1., -1.):
            detour = route + factor * self.excursion * bump
            if np.abs(detour).max() >= self.md.size / 2:
                continue
            rows, cols = self.md.xy_to_cell(detour).T
            if not blocked[rows, cols].any():
                self.reserved[rows, cols] = True
                self.blocked = blocked
                return True
        return False


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
    local = LocalDetours(md, path, arc, robot, float(s["margin"]))
    size = s["size"] or obstacle_size(robot)
    placed: list[np.ndarray] = []
    for _ in range(count):
        for _ in range(int(s["tries"])):
            along = rng.uniform(keepout, length - keepout)
            i = int(np.clip(np.searchsorted(arc, along) - 1, 0, len(path) - 2))
            tangent = (path[i + 1] - path[i]) / max(steps[i], 1e-9)
            radius = rng.uniform(*size)
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
            if not local.accept(footprint, along, tangent):
                continue
            placed.append(prim)
            break
    return np.stack(placed).astype(np.float32) if placed else np.zeros((0, 6), np.float32)


# ------------------------------------------------------------------------------------------- runtime
class SurpriseObstacles:
    """Per-environment obstacles that are not on the map, and the avoidance level they cause."""

    POINTS = 32

    def __init__(self, cfg, env):
        self.env = env
        self.s = settings(cfg)
        self.enabled = float(self.s["probability"]) > 0.0
        from dwarl.navigation.avoidance_curriculum import AvoidanceCurriculum

        self.curriculum = AvoidanceCurriculum() if self.enabled and self.s["curriculum"] else None
        n = env.num_envs
        self.episode_stages = np.zeros(n, dtype=int)
        self.max_count = int(self.s["count"][1]) if self.enabled else 0
        self.capacity = max(self.max_count, 1)
        self.prims = np.zeros((n, self.capacity, 6))
        self.valid = np.zeros((n, self.capacity), bool)
        self.fixed: dict[int, np.ndarray] = {}  # explicit obstacles for test cases
        self.level = np.zeros(n)
        self.gap = np.full(n, np.inf)        # closest path-to-obstacle gap in the horizon [m]
        self.ahead = np.full(n, np.inf)      # arc distance to that blockage [m]
        robot = {"footprint": env.footprint.tolist() if env.footprint is not None else None,
                 "footprint_radius": env.radius}
        self.robot = robot
        if self.s["size"] is None:
            self.s["size"] = obstacle_size(robot)
        self.width = half_width(robot)
        lo, hi = (float(v) for v in self.s["horizon"])
        self.horizon = float(np.clip(env.v_max * float(self.s["horizon_time"]), lo, hi))
        self.offsets = np.linspace(-env.radius, self.horizon, self.POINTS)
        near = float(self.s["near"]) * self.horizon
        self.weight = np.clip((self.horizon - self.offsets) / max(self.horizon - near, 1e-6), 0.0, 1.0)

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
        s = self.curriculum.settings(self.s) if self.curriculum else self.s
        self.episode_stages[env_id] = self.curriculum.stage if self.curriculum else 0
        if int(env_id) in self.fixed:
            prims = self.fixed[int(env_id)]
        elif self.enabled and rng.random() < float(s["probability"]):
            lo, hi = (int(v) for v in s["count"])
            prims = sample_on_path(md, path, self.robot, rng, int(rng.integers(lo, hi + 1)), s)
        else:
            prims = np.zeros((0, 6))
        self.prims[env_id] = 0.0
        self.valid[env_id] = False
        self.prims[env_id, :len(prims)] = prims
        self.valid[env_id, :len(prims)] = True
        self.level[env_id] = 0.0
        self.gap[env_id], self.ahead[env_id] = np.inf, np.inf
        return np.asarray(prims, np.float32)

    def obstacles(self, env_id: int) -> np.ndarray:
        return self.prims[env_id, self.valid[env_id]].astype(np.float32)

    def report(self, ids, successes, collisions):
        if self.curriculum:
            self.curriculum.report(self.episode_stages[ids], self.valid[ids].any(-1), successes, collisions)

    def update(self, dt: float) -> None:
        """Measure the blockage of the path ahead and update the avoidance level (after each step)."""
        decay = np.exp(-dt / float(self.s["release_time"])) if dt > 0 else 1.0
        if not self.valid.any():
            self.level *= decay
            self.gap[:], self.ahead[:] = np.inf, np.inf
            return
        path = self.env.path_command
        points = np.stack([path.point_at(path.s + d) for d in self.offsets], 1)  # (N, P, 2)
        gap = primitive_distance(points, self.prims, self.valid) - self.width       # (N, P)
        clear, ramp = float(self.s["clear"]), float(self.s["ramp"])
        blocked = np.clip((clear - gap) / ramp, 0.0, 1.0) * self.weight              # (N, P)
        self.level = np.maximum(blocked.max(-1), self.level * decay)
        closest = gap.argmin(-1)
        rows = np.arange(len(gap))
        self.gap = gap[rows, closest]
        self.ahead = np.where(np.isfinite(self.gap), self.offsets[closest], np.inf)

    def features(self) -> np.ndarray:
        """Privileged critic features: avoidance level, path gap and distance to the blockage."""
        gap = np.where(np.isfinite(self.gap), np.clip(self.gap, -0.5, 2.0) / 2.0, 1.0)
        ahead = np.where(np.isfinite(self.ahead), np.clip(self.ahead / self.horizon, -1.0, 1.0), 1.0)
        return np.stack([self.level, gap, ahead], -1)
