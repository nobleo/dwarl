"""Differential-drive action terms: policy actions -> limited velocities -> wheel targets."""

import numpy as np

from dwarl.navigation.managers.action_manager import ActionTerm


class DifferentialDriveAction(ActionTerm):
    action_dim = 2

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.mode = cfg.get("params", {}).get("mode", "window")
        if self.mode not in ("window", "absolute"):
            raise ValueError(f"unknown differential-drive action mode {self.mode!r}")
        self.command = np.zeros((env.num_envs, 2))
        self.command_delta = np.zeros_like(self.command)

    def process_actions(self, action):
        e = self.env
        next_command = command_from_action(action, self.command, self.mode, e.v_min, e.v_max, e.w_max,
                                           e.a_max, e.alpha_max, e.control_dt)
        self.command_delta[:] = next_command - self.command
        self.command[:] = next_command

    def apply_actions(self):
        e = self.env
        v, w = self.command.T
        half = 0.5 * e.wheel_sep
        e.world.set_ctrl(np.stack([(v - w * half) / e.wheel_r, (v + w * half) / e.wheel_r], -1))

    def reset(self, env_ids):
        self.command[env_ids] = 0.
        self.command_delta[env_ids] = 0.


def command_from_action(a: np.ndarray, cmd: np.ndarray, mode: str, v_min: float, v_max: float, w_max: float,
                        a_max: float, alpha_max: float, dt: float) -> np.ndarray:  # fmt: skip
    """Normalised action (N, 2) + current command (N, 2) -> next (v, w) command.

    ``window`` scales the reachable command change, then clips to velocity limits.
    ``absolute`` scales to a target velocity, then rate-limits the change.
    Actions must already be clipped to [-1, 1]; the caller validates the mode.
    """
    lim = np.array([a_max, alpha_max]) * dt
    if mode == "window":
        nxt = cmd + a * lim
        return np.stack([np.clip(nxt[:, 0], v_min, v_max), np.clip(nxt[:, 1], -w_max, w_max)], -1)
    target = np.stack([v_min + (a[:, 0] + 1) * 0.5 * (v_max - v_min), a[:, 1] * w_max], -1)
    return cmd + np.clip(target - cmd, -lim, lim)
