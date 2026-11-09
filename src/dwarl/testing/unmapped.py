"""Obstacles that exist in the simulated world but not on the navigation map.

They test a local planner on what it can only see in its sensor data: the global plan
runs through them. Placement is deterministic per case, so every controller meets the
same obstacles, and every obstacle set leaves a passage that the robot fits through.
"""

from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

from dwarl.datasets.generation.geometry import rasterize
from dwarl.datasets.generation.mapdata import BOX, CYLINDER


def passage_radius(robot: dict, margin: float) -> float:
    """Half the robot's width plus a margin: the free half-width a detour must have."""
    footprint = robot.get("footprint")
    half_width = max(abs(float(y)) for _, y in footprint) if footprint else float(robot["footprint_radius"])
    return half_width + margin


def _reachable(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    if blocked[start] or blocked[goal]:
        return False
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


def _dilate(grid: np.ndarray, cells: int) -> np.ndarray:
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


def narrow_points(md, path: np.ndarray, robot: dict) -> np.ndarray:
    """Reference-path points in passages narrower than two robot diameters (circumscribed):
    doorways and tight gaps, where there is no room to manoeuvre around an obstacle."""
    cells = int(np.ceil(2 * float(robot["footprint_radius"]) / md.resolution))
    tight = _dilate(md.occupancy, cells)
    rc = md.xy_to_cell(path)
    return path[tight[rc[:, 0], rc[:, 1]]]


def place_unmapped(md, case: dict, count: int, robot: dict, size: tuple[float, float] = (0.2, 0.35),
                   margin: float = 0.1, keepout: float = 1.5, spacing: float = 2.0,
                   tries: int = 40) -> np.ndarray:
    """Return up to ``count`` obstacle primitives (N, 6) on the case's reference path.

    Obstacles are spread along the path, at least ``keepout`` from start and goal and
    ``spacing`` apart. Each one must not touch mapped geometry and must keep two robot
    radii (circumscribed) plus its own size away from passages on the path that are narrower
    than two robot diameters, such as doorways: there is no room there to manoeuvre around
    an obstacle that appears right behind the passage. With all of them
    added the goal must stay reachable for a robot of ``passage_radius`` (half its width
    plus ``margin``). Candidates that fail are resampled; an obstacle is skipped after
    ``tries`` failed candidates, so fewer than ``count`` may be returned on tight maps.
    """
    if count <= 0:
        return np.zeros((0, 6), np.float32)
    path = np.asarray(case["reference_path"], np.float64)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    length = float(arc[-1])
    if length < 2 * keepout + 0.5:
        return np.zeros((0, 6), np.float32)
    seed = int.from_bytes(hashlib.sha256(f"{case['id']}:{case.get('seed', 0)}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    n, res = md.occupancy.shape[0], md.resolution
    cells = int(np.ceil(passage_radius(robot, margin) / res))
    start = tuple(md.xy_to_cell(np.asarray(case["start"][:2])).tolist())
    goal = tuple(md.xy_to_cell(np.asarray(case["goal"][:2])).tolist())
    mapped = md.occupancy
    # Starts close to a wall: require no more passage than the mapped route itself has.
    while cells > 0 and not _reachable(_dilate(mapped, cells), start, goal):
        cells -= 1
    narrow = narrow_points(md, path, robot)
    turn_clearance = 2 * float(robot["footprint_radius"])
    placed: list[np.ndarray] = []
    usable = length - 2 * keepout
    for k in range(count):
        target = keepout + usable * (k + 0.5) / count
        for _ in range(tries):
            s = float(np.clip(target + rng.uniform(-0.25, 0.25) * usable / count, keepout, length - keepout))
            i = int(np.clip(np.searchsorted(arc, s) - 1, 0, len(path) - 2))
            tangent = (path[i + 1] - path[i]) / max(steps[i], 1e-9)
            point = path[i] + tangent * (s - arc[i])
            radius = rng.uniform(*size)
            centre = point + np.array([-tangent[1], tangent[0]]) * rng.uniform(-0.6, 0.6) * radius
            if rng.random() < 0.5:
                prim = np.array([CYLINDER, *centre, 0.0, radius, radius], np.float32)
            else:
                prim = np.array([BOX, *centre, rng.uniform(0, np.pi), radius, radius * rng.uniform(0.6, 1.0)],
                                np.float32)
            if min(np.linalg.norm(centre - np.asarray(case["start"][:2])),
                   np.linalg.norm(centre - np.asarray(case["goal"][:2]))) < keepout:
                continue
            if any(np.linalg.norm(centre - other[1:3]) < spacing for other in placed):
                continue
            if len(narrow) and np.linalg.norm(narrow - centre, axis=1).min() < turn_clearance + radius:
                continue
            footprint = rasterize(prim[None], n, res, md.size)
            if (footprint & mapped).any():
                continue
            occupied = mapped | rasterize(np.stack([*placed, prim]), n, res, md.size)
            if not _reachable(_dilate(occupied, cells), start, goal):
                continue
            placed.append(prim)
            break
    return np.stack(placed).astype(np.float32) if placed else np.zeros((0, 6), np.float32)
