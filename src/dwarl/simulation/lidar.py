"""Planar 2D lidar via batched ray casting (``mj_multiRay``)."""

from __future__ import annotations

import numpy as np

from dwarl.simulation.world import ShardedWorld, TiledWorld


class Lidar2D:
    def __init__(
        self,
        world: TiledWorld | ShardedWorld,
        num_rays: int = 180,
        fov_deg: float = 360.0,
        min_range: float = 0.12,
        max_range: float = 8.0,
        noise_std: float = 0.0,
        dropout_prob: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.world = world
        self.num_rays = num_rays
        self.min_range, self.max_range = min_range, max_range
        self.noise_std, self.dropout_prob = noise_std, dropout_prob
        fov = np.deg2rad(fov_deg)
        if np.isclose(fov_deg, 360.0):
            self.angles = -np.pi + np.arange(num_rays) * (2 * np.pi / num_rays)
        else:
            self.angles = np.linspace(-fov / 2, fov / 2, num_rays)
        self._rng = np.random.default_rng(seed)

    def scan(self, yaw: np.ndarray) -> np.ndarray:
        """Ranges in metres, (num_envs, num_rays), ray 0 at -fov/2 (i.e. behind for 360 deg), CCW."""
        out = self.world.cast_rays(yaw, self.angles, self.max_range)
        out[out < 0] = self.max_range
        if self.noise_std > 0:
            out += self._rng.normal(0.0, self.noise_std, out.shape)
        if self.dropout_prob > 0:
            out[self._rng.random(out.shape) < self.dropout_prob] = self.max_range
        return np.clip(out, self.min_range, self.max_range)

    def normalize(self, ranges: np.ndarray) -> np.ndarray:
        return (ranges - self.min_range) / (self.max_range - self.min_range)
