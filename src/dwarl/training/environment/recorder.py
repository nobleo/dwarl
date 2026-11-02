"""Navigation episode metrics and traces, consuming explicit control-tick measurements."""

from copy import deepcopy

import numpy as np

from dwarl.navigation.terminations import goal_tolerance
from dwarl.training.environment.episode import EpisodeInfo, EpisodeStep


class EpisodeRecorder:
    def __init__(self, env, *, clearance_method):
        self.env = env
        self.clearance_method = clearance_method
        self.enabled = False
        self.traces = [[] for _ in range(env.num_envs)]
        self.travelled = np.zeros(env.num_envs)
        self.min_clearance = np.full(env.num_envs, np.inf)
        self.command_variation = np.zeros(env.num_envs)
        self.stalled_time = np.zeros(env.num_envs)

    def pre_step(self, xy, goals, main_goal):
        self.xy_before = xy.copy()
        if self.enabled:
            self.goals_before = goals.copy()
            self.main_goal_before = main_goal.copy()

    def record_step(self, step: EpisodeStep):
        self.travelled += np.linalg.norm(step.xy - self.xy_before, axis=-1)
        self.command_variation += np.linalg.norm(step.command_delta, axis=-1)
        self.stalled_time += ((np.abs(step.speed) < .03) & (np.abs(step.omega) < .1)) * self.env.control_dt
        self.min_clearance = np.minimum(self.min_clearance, step.clearance)
        if not self.enabled:
            return
        imin = step.ranges.argmin(-1)
        for i in range(self.env.num_envs):
            self.traces[i].append(dict(
                xy=step.xy[i].copy(), yaw=float(step.yaw[i]), v=float(step.speed[i]), w=float(step.omega[i]),
                cmd=step.command[i].copy(), act=step.action[i].copy(),
                look=self.main_goal_before[i].copy(), goals=self.goals_before[i].copy(),
                min_range=float(step.ranges[i, imin[i]]), min_bearing=float(step.scan_angles[imin[i]]),
                scan=step.ranges[i].astype(np.float32),
                r={name: float(value[i]) for name, value in step.rewards.items()},
                clearance=float(step.clearance[i]), lidar_clearance=float(step.lidar_clearance[i]),
            ))

    def finish(self, ids) -> tuple[list[EpisodeInfo], dict[str, float]]:
        e, path = self.env, self.env.path_command
        flags = e.termination_manager
        outcomes = flags.outcomes
        completion = np.clip(path.s[ids] / np.maximum(path.path_total[ids], 1e-6), 0, 1)
        episodes = []
        for j, i in enumerate(ids):
            steps = int(e.episode_length_buf[i])
            info = EpisodeInfo(
                map=int(e.map_idx[i]), pair=int(e.pair_idx[i]),
                **{role: bool(value[i]) for role, value in outcomes.items()},
                terminated=bool(flags.terminated[i]),
                termination_terms={name: bool(value[i]) for name, value in flags.term_dones.items()},
                steps=steps, path_length=float(path.path_total[i]), travelled=float(self.travelled[i]),
                completion=float(completion[j]), generator=e.dataset.maps[e.map_idx[i]].generator,
                episode_id=f"{int(e.cfg.seed)}/{e.map_idx[i]}/{e.pair_idx[i]}/{e.episode_repeat[i]}",
                repeat=int(e.episode_repeat[i]), initial_yaw=float(e.initial_yaw[i]),
                initial_sensor_state=deepcopy(e.initial_sensor_state[i]), time_seconds=steps * e.control_dt,
                min_footprint_clearance=float(self.min_clearance[i]),
                clearance_method=self.clearance_method,
                command_variation=float(self.command_variation[i]), stalled_seconds=float(self.stalled_time[i]),
                interventions=0, fallback_steps=0,
                reward_components={name: float(value[i]) for name, value in e.reward_manager.episode_sums.items()},
                trace=self.pack_trace(i) if self.enabled else None,
            )
            episodes.append(info)
        return episodes, {
            "Metrics/path_completion": float(completion.mean()), "Curriculum/level": float(e.sampler.level),
        }

    def pack_trace(self, i):
        e, trace = self.env, self.traces[i]
        if not trace:
            return {}
        out = {key: np.array([row[key] for row in trace]) for key in
               ("xy", "yaw", "v", "w", "cmd", "act", "look", "goals", "min_range", "min_bearing",
                "clearance", "lidar_clearance")}
        tolerance = goal_tolerance(e.cfg.terminations)
        out.update(
            scan_last=trace[-1]["scan"], scan_angles=e.lidar.angles.astype(np.float32),
            start_yaw=float(e.initial_yaw[i]), lidar_origin=e.lidar_origin.astype(np.float32),
            v_min=e.v_min, action_mode=e.drive_action.mode, max_range=float(e.lidar.max_range), radius=e.radius,
            goal_tol=tolerance, dt=e.control_dt, v_max=e.v_max, w_max=e.w_max,
            clearance_method=self.clearance_method,
        )
        if e.footprint is not None:
            out["footprint"] = e.footprint.astype(np.float32)
        for name in trace[0]["r"]:
            out[f"r_{name}"] = np.array([row["r"][name] for row in trace])
        return out

    def reset(self, ids):
        self.travelled[ids] = 0.
        self.min_clearance[ids] = np.inf
        self.command_variation[ids] = 0.
        self.stalled_time[ids] = 0.
        for i in ids:
            self.traces[i] = []
