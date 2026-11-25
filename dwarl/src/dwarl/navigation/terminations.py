"""Navigation termination terms. Stateful terms reset only the selected environments."""

import numpy as np

from dwarl.navigation.managers.base import ManagerTermBase


def collision(env, min_upright=.5):
    return env.contacts | (env.world.robot_upright() < min_upright)


def flipped(env, min_upright=.5):
    return env.world.robot_upright() < min_upright


def goal_reached(env, tolerance=.3, command_name="path", min_upright=.5):
    xy, _ = env.world.robot_pose()
    goal = env.command_manager.get_term(command_name).goal
    return (np.linalg.norm(goal - xy, axis=-1) < tolerance) & ~collision(env, min_upright)


def goal_tolerance(terminations, *, require_standard=False):
    """Tolerance for standard path-goal visualization/export, independent of term names.

    Custom success predicates need not define a distance tolerance. Export requires
    the standard predicate; visualization uses only recognized path-goal terms.
    """
    success = [term for term in terminations.values() if term is not None and term.get("role") == "success"]
    standard = [term for term in success
                if term["func"] == "dwarl.navigation.terminations:goal_reached"
                and term.get("params", {}).get("command_name", "path") == "path"]
    if require_standard and (not standard or len(standard) != len(success)):
        raise ValueError("deployment requires standard path goal_reached terms with role=success")
    return max((float(term.get("params", {}).get("tolerance", .3)) for term in standard), default=0.)


class NoProgress(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.anchor = np.zeros(env.num_envs)
        self.steps = np.zeros(env.num_envs, dtype=np.int64)

    def __call__(self, env, steps=50, distance=.3, command_name="path"):
        s = env.command_manager.get_term(command_name).s
        advanced = s >= self.anchor + distance
        self.anchor = np.where(advanced, s, self.anchor)
        self.steps = np.where(advanced, 0, self.steps + 1)
        return (self.steps >= steps) if steps > 0 else np.zeros(env.num_envs, bool)

    def reset(self, env_ids):
        self.anchor[env_ids] = 0.
        self.steps[env_ids] = 0


class PathTimeLimit(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.max_steps = np.zeros(env.num_envs, dtype=np.int64)

    def __call__(self, env, min_steps=100, time_factor=2.5, command_name="path"):
        return env.episode_length_buf.cpu().numpy() >= self.max_steps

    def reset(self, env_ids):
        params = self.cfg["params"]
        path = self.env.command_manager.get_term(params.get("command_name", "path"))
        duration = params["time_factor"] * path.path_total[env_ids] / self.env.v_max / self.env.control_dt
        self.max_steps[env_ids] = int(params["min_steps"]) + duration.astype(np.int64)
