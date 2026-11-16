"""The deployed collision monitor's STOP zones, for training against the safety layer the robot drives with.

``dwarl_bringup.nav.safety_zones`` builds velocity-dependent zones for Nav2's collision_monitor: the
footprint's bounding box grown by a margin, stretched in the driving direction by the braking distance
v^2 / (2 a) + |v| * latency at the top speed of each band, and an octagon around the whole footprint for
turning on the spot. The zone is chosen by the *commanded* velocity, and the monitor stops the robot when
``min_points`` scan returns lie inside it. :func:`zone_hits` reproduces that test for batches of scans.

Without it, training only learns that driving close to an obstacle is penalised, not that the robot then
stands still: on the robot the policy kept commanding the motion that the monitor blocked, and with an
obstacle behind it Nav2's backup recovery failed too.
"""

from __future__ import annotations

import math

import numpy as np

from dwarl.navigation.options import SAFETY_LAYER as DEFAULTS  # noqa: F401  (re-exported)
from dwarl.navigation.options import safety_layer as settings

# band limits of dwarl_bringup.nav.safety_zones (m/s, rad/s)
_CREEP = 0.05
_TURN = 0.2


def footprint_polygon(footprint, radius: float) -> np.ndarray:
    """The footprint polygon, or the octagon dwarl_bringup uses for a round robot."""
    if footprint is not None:
        return np.asarray(footprint, dtype=np.float64)
    k = np.arange(8) * math.pi / 4 + math.pi / 8
    return float(radius) * np.stack([np.cos(k), np.sin(k)], -1)


def zone_hits(points: np.ndarray, command: np.ndarray, polygon: np.ndarray, v_min: float, v_max: float,
              decel: float, margin: float = 0.035, latency: float = 0.15) -> np.ndarray:  # fmt: skip
    """Number of ``points`` (N, P, 2, base frame) inside the STOP zone selected by each ``command`` (N, 2)."""
    x0, y0 = polygon.min(0) - margin
    x1, y1 = polygon.max(0) + margin
    brake = lambda v: v * v / (2 * decel) + abs(v) * latency  # noqa: E731
    vmid = 0.5 * v_max
    v, w = np.asarray(command, dtype=np.float64).T
    px, py = points[..., 0], points[..., 1]
    lateral = (py >= y0) & (py <= y1)
    # forward_fast, forward_slow, backward and the stopped fallback are boxes that differ in their x extent
    front = np.where(v >= vmid, x1 + brake(v_max), np.where(v >= _CREEP, x1 + brake(vmid), x1))
    back = np.where(v <= -_CREEP, x0 - brake(min(v_min, -_CREEP)), x0)
    in_box = lateral & (px >= back[:, None]) & (px <= front[:, None])
    # turn_left / turn_right: octagon with circumradius R, vertices at pi/8 + k pi/4 (edge normals at k pi/4)
    radius = float(np.linalg.norm(polygon, axis=-1).max()) + margin
    normals = np.arange(8) * math.pi / 4
    reach = px[..., None] * np.cos(normals) + py[..., None] * np.sin(normals)
    in_octagon = reach.max(-1) <= radius * math.cos(math.pi / 8)
    turning = (np.abs(v) < _CREEP) & (np.abs(w) >= _TURN)
    inside = np.where(turning[:, None], in_octagon, in_box)
    return inside.sum(-1)


class SafetyLayer:
    """Stops an environment's robot while its commanded velocity's STOP zone holds lidar returns."""

    def __init__(self, cfg, env):
        s = settings(cfg)
        self.enabled = bool(s["enabled"])
        self.margin, self.latency, self.min_points = float(s["margin"]), float(s["latency"]), int(s["min_points"])
        self.env = env
        self.polygon = footprint_polygon(env.footprint, env.radius)
        self.blocked = np.zeros(env.num_envs, dtype=bool)

    def hits(self, command: np.ndarray) -> np.ndarray:
        e = self.env
        rays = np.stack([np.cos(e.lidar.angles), np.sin(e.lidar.angles)], -1)
        points = e.lidar_origin + e.ranges[..., None] * rays
        return zone_hits(points, command, self.polygon, e.v_min, e.v_max, e.a_max, self.margin, self.latency)

    def stops(self, command: np.ndarray) -> np.ndarray:
        """Boolean (N,): the monitor would stop the robot for this command and the latest scan."""
        return self.hits(command) >= self.min_points
