"""Path-based reward terms. Each term returns an unweighted (N,) array; weights come from config.

The training A* path supplies reward measurements and the critic's privileged path features.
With lookahead guidance, the actor also sees sampled path points in the robot frame; deployment
provides those points from the navigation stack's global plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardContext:
    progress: np.ndarray  # arc-length progress along the path this step [m]
    cross_track: np.ndarray  # |distance| to path [m]
    heading_error: np.ndarray  # robot yaw - path tangent [rad]
    speed: np.ndarray  # forward speed [m/s]
    action: np.ndarray  # (N, 2) normalised action
    prev_action: np.ndarray
    min_range: np.ndarray  # closest lidar return [m] (measured from the lidar origin)
    success: np.ndarray  # bool
    collision: np.ndarray  # bool (includes flipping)
    v_max: float
    dt: float
    footprint_radius: float
    stuck: np.ndarray | None = None  # bool: terminated for lack of progress
    clearance: np.ndarray | None = None  # footprint -> closest lidar return [m] (None: min_range - radius)
    # 1 - avoidance level: path-adherence weight, < 1 while obstacles that are not on the map block the path
    # ahead (dwarl.navigation.surprises). None or 1: follow the path.
    adherence: np.ndarray | float | None = None


def _adherence(c) -> np.ndarray | float:
    return 1.0 if c.adherence is None else c.adherence


def progress(env) -> np.ndarray:
    c = env.reward_context
    return c.progress / (c.v_max * c.dt)  # ~1 per step at full speed along the path


def cross_track(env) -> np.ndarray:
    c = env.reward_context
    # relaxed while the path ahead is blocked, so a detour costs only progress
    return -np.minimum(c.cross_track, 1.0) * _adherence(c)


def heading(env) -> np.ndarray:
    c = env.reward_context
    # only rewarded while moving forward, so spinning in place is not a reward source; relaxed like cross_track
    return np.cos(c.heading_error) * np.clip(c.speed / c.v_max, 0.0, 1.0) * _adherence(c)


def goal(env) -> np.ndarray:
    c = env.reward_context
    return c.success.astype(np.float64)


def collision(env) -> np.ndarray:
    c = env.reward_context
    return -c.collision.astype(np.float64)


def stuck(env) -> np.ndarray:
    c = env.reward_context
    return -(c.stuck.astype(np.float64) if c.stuck is not None else np.zeros_like(c.progress))


def time(env) -> np.ndarray:
    c = env.reward_context
    return -np.ones_like(c.progress)


def action_rate(env) -> np.ndarray:
    c = env.reward_context
    return -np.sum((c.action - c.prev_action) ** 2, axis=-1)


def proximity(env, band: float = 0.25, power: float = 1.0) -> np.ndarray:
    c = env.reward_context
    # soft penalty when the footprint is closer than `band` m to a (visible) obstacle surface; 1 at contact
    gap = c.clearance if c.clearance is not None else c.min_range - c.footprint_radius
    return -(np.clip((band - gap) / band, 0.0, 1.0) ** power)


def reverse(env, deadband: float = 0.02, power: float = 1.0) -> np.ndarray:
    c = env.reward_context
    # penalise measured backward speed beyond a deadband, normalised by v_max: -((|v_rev| - db) / v_max)^power
    rev = np.clip(-c.speed - deadband, 0.0, None) / c.v_max
    return -(rev**power)


def angular_effort(env, action_name="drive") -> np.ndarray:
    """Squared angular acceleration, normalised by the limit: weaving uses it constantly, smooth driving rarely.

    With the default ``window`` action mode the policy's action is the command change, so a penalty on action
    changes (``action_rate``) misses weaving made of long runs of full acceleration; this term does not.
    """
    term = env.action_manager.get_term(action_name)
    limit = max(env.alpha_max * env.control_dt, 1e-9)
    return -((term.command_delta[:, 1] / limit) ** 2)


def safety_zone(env, padding: float = 0.035, reaction_time: float = 0.15) -> np.ndarray:
    """-1 while a lidar return lies inside a collision monitor's STOP zone, as the deployed stack builds it.

    The zone is the footprint grown by ``padding``, stretched forward by the braking distance
    v^2 / (2 a_max) + v * reaction_time (dwarl_bringup's collision_monitor polygons). On the robot an entry
    stops it and triggers recoveries; in training it otherwise costs nothing, so the policy learns to pass
    obstacles with millimetres to spare.
    """
    from dwarl.navigation.clearance import polygon_distance

    c = env.reward_context
    rays = np.stack([np.cos(env.lidar.angles), np.sin(env.lidar.angles)], -1)
    points = env.lidar_origin + env.ranges[..., None] * rays
    if env.footprint is None:
        near = np.linalg.norm(points, axis=-1) <= env.radius + padding
        front, half = env.radius, env.radius
    else:
        near = polygon_distance(points, env.footprint) <= padding
        front, half = float(env.footprint[:, 0].max()), float(np.abs(env.footprint[:, 1]).max())
    speed = np.clip(c.speed, 0.0, None)
    reach = front + padding + speed**2 / (2 * env.a_max) + speed * reaction_time
    ahead = (points[..., 0] >= front) & (points[..., 0] <= reach[:, None]) & (np.abs(points[..., 1]) <= half + padding)
    return -(near | ahead).any(-1).astype(np.float64)
