"""Runtime map swapping by rewriting the per-tile geom pool of a compiled model."""

from __future__ import annotations

import mujoco
import numpy as np

from dwarl.datasets.generation.mapdata import CYLINDER
from dwarl.simulation.world import TiledWorld

_OBSTACLE_RGBA = np.array([0.45, 0.47, 0.52, 1.0])


def write_tile_map(world: TiledWorld, env: int, primitives: np.ndarray, height: float) -> None:
    """Place ``primitives`` ([type, x, y, yaw, hx, hy], tile-local) into tile ``env``'s geom pool."""
    m = world.model
    n = len(primitives)
    if n > world.pool_size:
        raise ValueError(f"map has {n} primitives but sim.pool_size is {world.pool_size}")
    ids = world.pool_geoms[env]
    used, unused = ids[:n], ids[n:]
    hz = height / 2.0

    if n:
        p = primitives.astype(np.float64)
        is_cyl = p[:, 0].astype(int) == CYLINDER
        m.geom_type[used] = np.where(is_cyl, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX)
        m.geom_pos[used] = np.stack([p[:, 1], p[:, 2], np.full(n, hz)], -1)  # relative to tile body
        yaw = np.where(is_cyl, 0.0, p[:, 3])
        m.geom_quat[used] = np.stack([np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)], -1)
        size = np.where(is_cyl[:, None], np.stack([p[:, 4], np.full(n, hz), np.zeros(n)], -1),
                        np.stack([p[:, 4], p[:, 5], np.full(n, hz)], -1))  # fmt: skip
        m.geom_size[used] = size
        half = np.where(is_cyl[:, None], np.stack([p[:, 4], p[:, 4], np.full(n, hz)], -1), size)
        m.geom_aabb[used] = np.concatenate([np.zeros((n, 3)), half], -1)
        m.geom_rbound[used] = np.linalg.norm(half, axis=-1)
        m.geom_contype[used] = 1
        m.geom_conaffinity[used] = 1
        m.geom_rgba[used] = _OBSTACLE_RGBA

    if len(unused):
        k = len(unused)
        m.geom_type[unused] = mujoco.mjtGeom.mjGEOM_BOX
        m.geom_pos[unused] = [0.0, 0.0, -1.0]
        m.geom_quat[unused] = [1.0, 0.0, 0.0, 0.0]
        m.geom_size[unused] = [0.01, 0.01, 0.01]
        m.geom_aabb[unused] = np.tile([0, 0, 0, 0.01, 0.01, 0.01], (k, 1))
        m.geom_rbound[unused] = 0.0175
        m.geom_contype[unused] = 0
        m.geom_conaffinity[unused] = 0
        m.geom_rgba[unused] = [0, 0, 0, 0]
