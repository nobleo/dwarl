"""Policy and critic observation terms. All sensor sampling is owned by the environment."""

import numpy as np

from dwarl.navigation.commands import to_robot, wrap_angle
from dwarl.navigation.managers.base import ManagerTermBase


def normalize_ranges(ranges, minimum, maximum, mode="linear"):
    if mode == "linear":
        return (ranges - minimum) / (maximum - minimum)
    if mode == "log":
        return np.log1p(np.maximum(ranges - minimum, 0.) / .1) / np.log1p((maximum - minimum) / .1)
    raise ValueError(f"unknown lidar normalization {mode!r}")


class LidarHistory(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg["params"]
        self.history = [int(k) for k in params["history"]]
        if not self.history or self.history[0] != 0 or self.history != sorted(set(self.history)):
            raise ValueError("LiDAR history must be strictly ascending and start at 0")
        self.normalization = params["normalization"]
        self.buffer = None
        self.fresh = np.ones(env.num_envs, bool)

    def __call__(self, env, history, normalization):
        scan = normalize_ranges(env.ranges, env.lidar.min_range, env.lidar.max_range, normalization)
        if self.buffer is None:
            self.buffer = np.repeat(scan[:, None], self.history[-1] + 1, axis=1)
        else:
            self.buffer = np.concatenate([scan[:, None], self.buffer[:, :-1]], axis=1)
        self.buffer[self.fresh] = scan[self.fresh, None]
        self.fresh[:] = False
        return self.buffer[:, None, self.history]

    def reset(self, env_ids):
        self.fresh[env_ids] = True


def goal_points(env, command_name="path", max_distance=10.):
    xy, yaw = env.world.robot_pose()
    goals = env.command_manager.get_command(command_name)
    relative = to_robot(goals, xy[:, None], yaw[:, None])
    distance = np.linalg.norm(relative, axis=-1)
    angle = np.arctan2(relative[..., 1], relative[..., 0])
    return np.stack([np.minimum(distance, max_distance) / max_distance, np.sin(angle), np.cos(angle)], -1).reshape(
        env.num_envs, -1)


def last_action(env):
    return env.action_manager.action


def base_velocity(env):
    _, yaw = env.world.robot_pose()
    v, w = env.world.robot_velocity(yaw)
    return np.stack([v / env.v_max, w / env.w_max], -1)


def last_command(env, action_name="drive"):
    return env.action_manager.get_term(action_name).command / [env.v_max, env.w_max]


def path_features(env, command_name="path", lookahead=(.5, 1., 2., 3.)):
    path = env.command_manager.get_term(command_name)
    xy, yaw = env.world.robot_pose()
    points = np.stack([path.point_at(path.s + d) for d in lookahead], 1)
    points = to_robot(points, xy[:, None], yaw[:, None]) / 3.
    heading = wrap_angle(yaw - path.tangent)
    distance = np.linalg.norm(to_robot(path.main_goal, xy, yaw), axis=-1)
    remaining = np.clip((path.path_total - path.s) / 20., 0., 1.)
    return np.concatenate([
        np.stack([np.clip(path.cross_track, -1, 1), np.sin(heading), np.cos(heading), remaining,
                  np.minimum(distance, 20.) / 20.], -1), points.reshape(env.num_envs, -1),
    ], -1)
