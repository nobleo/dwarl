"""Path-following task command: local targets and progress along a reference polyline."""

import numpy as np

from dwarl.navigation.managers.command_manager import CommandTerm


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def to_robot(points, xy, yaw):
    delta = points - xy
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]], -1)


class PathCommand(CommandTerm):
    """Own path projection state and the points used as local navigation targets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg["params"]
        self.mode = params["goal_mode"]
        if self.mode not in ("lookahead", "final"):
            raise ValueError(f"unknown goal mode {self.mode!r}")
        self.lookahead = [float(d) for d in params["lookahead_distances"]]
        if not self.lookahead or any(d <= 0 for d in self.lookahead):
            raise ValueError("lookahead_distances must contain positive distances")
        self.main_distance = float(params["lookahead_distance"])
        self.window = tuple(params.get("projection_window", [-5, 30]))
        n, points = env.num_envs, env.dataset.max_path_points
        self.path = np.zeros((n, points, 2))
        self.path_s = np.zeros((n, points))
        self.path_len = np.full(n, 2, dtype=np.int64)
        self.path_total = np.zeros(n)
        self.seg_idx = np.zeros(n, dtype=np.int64)
        self.s_prev = np.zeros(n)
        self.s = np.zeros(n)
        self.progress = np.zeros(n)
        self.cross_track = np.zeros(n)
        self.tangent = np.zeros(n)
        self.start = np.zeros((n, 2))
        self.goal = np.zeros((n, 2))
        self.command = np.zeros((n, len(self.lookahead) if self.mode == "lookahead" else 1, 2))
        self.main_goal = np.zeros((n, 2))

    def reset(self, env_ids):
        for i in env_ids:
            path = self.env.dataset.maps[self.env.map_idx[i]].path(self.env.pair_idx[i]).astype(np.float64)
            length = len(path)
            self.path[i, :length], self.path[i, length:] = path, path[-1]
            self.path_s[i, :length] = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
            self.path_s[i, length:] = self.path_s[i, length - 1]
            self.path_len[i], self.path_total[i] = length, self.path_s[i, length - 1]
            self.start[i], self.goal[i] = path[0], path[-1]
        self.seg_idx[env_ids] = 0
        self.s_prev[env_ids] = 0.
        self.s[env_ids] = 0.
        self.progress[env_ids] = 0.

    def project(self, xy):
        """Bounded segment-window projection -> arc length, signed cross-track error, tangent."""
        n = self.env.num_envs
        offsets = np.arange(*self.window)
        last_seg = np.maximum(self.path_len - 2, 0)
        candidates = np.clip(self.seg_idx[:, None] + offsets[None], 0, last_seg[:, None])
        rows = np.arange(n)[:, None]
        a, b = self.path[rows, candidates], self.path[rows, candidates + 1]
        ab = b - a
        length2 = np.maximum((ab**2).sum(-1), 1e-12)
        t = np.clip(((xy[:, None] - a) * ab).sum(-1) / length2, 0., 1.)
        distance = np.linalg.norm(xy[:, None] - (a + t[..., None] * ab), axis=-1)
        arc = self.path_s[rows, candidates] + t * np.sqrt(length2)
        reach = 2. * self.env.v_max * self.env.control_dt + 0.1
        best = np.where(np.abs(arc - self.s_prev[:, None]) <= reach, distance, distance + 1e6).argmin(-1)
        rows = np.arange(n)
        self.seg_idx = candidates[rows, best]
        s = self.path_s[rows, self.seg_idx] + t[rows, best] * np.sqrt(length2[rows, best])
        direction, relative = ab[rows, best], xy - a[rows, best]
        error = np.sign(direction[:, 0] * relative[:, 1] - direction[:, 1] * relative[:, 0]) * distance[rows, best]
        return s, error, np.arctan2(direction[:, 1], direction[:, 0])

    def point_at(self, s):
        rows = np.arange(self.env.num_envs)
        s = np.clip(s, 0., self.path_s[rows, self.path_len - 1])
        hi = np.clip((self.path_s < s[:, None]).sum(1), 1, self.path_s.shape[1] - 1)
        lo = hi - 1
        span = self.path_s[rows, hi] - self.path_s[rows, lo]
        t = np.where(span > 1e-12, (s - self.path_s[rows, lo]) / np.maximum(span, 1e-12), 1.)
        return self.path[rows, lo] + t[:, None] * (self.path[rows, hi] - self.path[rows, lo])

    def compute(self, dt):
        xy, _ = self.env.world.robot_pose()
        self.s, self.cross_track, self.tangent = self.project(xy)
        bound = self.env.v_max * dt * 1.2
        self.progress = np.clip(self.s - self.s_prev, -bound, bound)
        self.s_prev = self.s.copy()
        if self.mode == "lookahead":
            self.command = np.stack([self.point_at(self.s + distance) for distance in self.lookahead], 1)
            self.main_goal = self.point_at(self.s + self.main_distance)
        else:
            self.command = self.goal[:, None].copy()
            self.main_goal = self.goal.copy()
