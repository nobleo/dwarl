"""Manager-based navigation environment with the RSL-RL VecEnv interface.

The environment owns simulation, sensors and episode scheduling. Configured
managers own actions, path commands, observations, rewards and terminations.
Completed episodes reset individually before the next observation is returned.
"""

from __future__ import annotations

import tempfile
import time
import warnings
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from dwarl.navigation.clearance import footprint_clearance, polygon_distance
from dwarl.navigation.commands import wrap_angle
from dwarl.navigation.managers import (
    ActionManager,
    CommandManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from dwarl.navigation.rewards import RewardContext
from dwarl.robot.config import is_materialized, materialize, validate_effective
from dwarl.simulation.lidar import Lidar2D
from dwarl.simulation.world import ShardedWorld, TiledWorld, default_num_shards
from dwarl.training.environment.dataset import MapDataset, PairSampler
from dwarl.training.environment.episode import EpisodeInfo, EpisodeStep
from dwarl.training.environment.recorder import EpisodeRecorder


class NavVecEnv(VecEnv):
    def __init__(self, cfg: Any, split="train", sequential=False, gui=None):
        self._robot_tmp = None
        self.world = None
        self.viewer = None
        # Optional live view: while this file exists, a passive viewer shows the first world shard.
        self.viewer_trigger = None
        self._viewer_poll = 0.0
        self._view_envs = 0
        self._episode_paths = {}
        try:
            self._initialize(cfg, split, sequential, gui)
        except BaseException:
            self.close()
            raise

    def _initialize(self, cfg, split, sequential, gui):
        from dwarl.config.managers import apply_manager_schemas

        cfg = apply_manager_schemas(cfg)
        self.cfg = cfg
        ec, sc, rc = cfg.env, cfg.sim, cfg.robot
        self.num_envs = int(ec.num_envs)
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(int(cfg.seed))
        self.control_dt = float(ec.control_dt)
        self.decimation = max(1, int(round(self.control_dt / float(sc.timestep))))
        if not np.isclose(self.decimation * float(sc.timestep), self.control_dt):
            raise ValueError("control_dt must be an integer multiple of the physics timestep")
        self.dataset = MapDataset(cfg.dataset.path, split)
        cc = cfg.curriculum
        self.sampler = PairSampler(
            self.dataset, enabled=bool(cc.enabled), init_level=cc.init_level, min_level=cc.min_level,
            step=cc.step, window=cc.window, promote_success=cc.promote_success, demote_success=cc.demote_success,
            sequential=sequential, seed=int(cfg.seed),
        )
        if not is_materialized(cfg):
            self._robot_tmp = tempfile.TemporaryDirectory(prefix="dwarl_robot_")
            materialize(cfg, self._robot_tmp.name)
        validate_effective(cfg)
        self.radius = float(rc.footprint_radius)
        self.footprint = np.asarray(rc.footprint, dtype=np.float64) if rc.get("footprint") else None
        self.v_min, self.v_max, self.w_max = float(rc.v_min), float(rc.v_max), float(rc.w_max)
        self.a_max, self.alpha_max = float(rc.a_max), float(rc.alpha_max)
        self._check_dataset()
        gui = bool(sc.gui) if gui is None else gui
        self.world = self._build_world(sc, rc, gui)
        robot = self.world.robot
        self.wheel_r, self.wheel_sep = robot.wheel_radius, robot.wheel_separation
        self.lidar_origin = np.asarray(robot.lidar_offset[:2], dtype=np.float64)
        lc = cfg.lidar
        self.lidar = Lidar2D(self.world, num_rays=lc.num_rays, fov_deg=lc.fov_deg, min_range=lc.min_range,
                             max_range=lc.max_range, noise_std=lc.noise_std, dropout_prob=lc.dropout_prob,
                             seed=int(cfg.seed))
        n = self.num_envs
        self.map_idx = np.full(n, -1, dtype=np.int64)
        self.pair_idx = np.zeros(n, dtype=np.int64)
        self.episodes_on_map = np.zeros(n, dtype=np.int64)
        self.episode_visits = {}
        self.episode_repeat = np.zeros(n, dtype=np.int64)
        self.initial_yaw = np.zeros(n)
        self.initial_sensor_state = [None] * n
        self.contacts = np.zeros(n, bool)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=self.device)
        self.last_episode_info: list[EpisodeInfo] = []
        self.command_manager = CommandManager(cfg.commands, self)
        self.action_manager = ActionManager(cfg.actions, self)
        self.num_actions = self.action_manager.total_action_dim
        self.observation_manager = ObservationManager(cfg.observations, self)
        self.termination_manager = TerminationManager(cfg.terminations, self)
        self.reward_manager = RewardManager(cfg.rewards, self)
        self.recorder = EpisodeRecorder(
            self, clearance_method="geometry_control_ticks" if sequential else "visible_lidar_footprint")
        if gui:
            from dwarl.simulation.viewer import SimViewer

            self.viewer = SimViewer(self.world, render_fps=float(sc.render_fps), realtime=bool(sc.realtime),
                                    speed=float(sc.realtime_speed))
            self._view_envs = self.num_envs
        self.reset()

    @property
    def path_command(self):
        """The navigation task's configured reference-path term."""
        return self.command_manager.get_term("path")

    @property
    def drive_action(self):
        return self.action_manager.get_term("drive")

    @property
    def max_episode_length(self):
        steps = self.termination_manager.episode_limits
        return torch.as_tensor(steps, device=self.device)

    @property
    def record(self):
        return self.recorder.enabled

    @record.setter
    def record(self, enabled):
        self.recorder.enabled = enabled

    def get_observations(self) -> TensorDict:
        return self._obs

    def reset(self):
        self._reset_idx(np.arange(self.num_envs))
        self.world.forward()
        self.last_episode_info = []
        self._obs = self._observe()
        return self._obs, {}

    def step(self, actions: torch.Tensor):
        xy_before, _ = self.world.robot_pose()
        self.recorder.pre_step(xy_before, self.path_command.command, self.path_command.main_goal)
        self.action_manager.process_action(actions.detach().cpu().numpy())
        self.action_manager.apply_action()
        self.contacts = self.world.step(self.decimation)
        self.episode_length_buf += 1
        self.command_manager.compute(self.control_dt)
        dones = self.termination_manager.compute()
        xy, yaw = self.world.robot_pose()
        speed, omega = self.world.robot_velocity(yaw)
        self.reward_context = self._measure_rewards(yaw, speed)
        reward = self.reward_manager.compute(self.control_dt)
        self.recorder.record_step(EpisodeStep(
            xy=xy, yaw=yaw, speed=speed, omega=omega,
            command=self.drive_action.command, command_delta=self.drive_action.command_delta,
            action=self.action_manager.action, ranges=self.ranges, scan_angles=self.lidar.angles,
            clearance=self._episode_clearance(xy, yaw), lidar_clearance=self.reward_context.clearance,
            rewards=self.reward_manager.term_values,
        ))
        extras = {
            "time_outs": torch.as_tensor(self.termination_manager.time_outs.copy(), device=self.device), "log": {},
        }
        self.last_episode_info = []
        done_ids = np.flatnonzero(dones)
        if len(done_ids):
            self.last_episode_info, extras["log"] = self.recorder.finish(done_ids)
            self.sampler.report(self.termination_manager.get_outcome("success")[done_ids])
            extras["log"].update(self._reset_idx(done_ids))
            self.world.forward(done_ids)
        self._obs = self._observe()
        if self.viewer_trigger is not None:
            self._poll_viewer()
        if self.viewer is not None:
            n = self._view_envs
            points = self.path_command.command[:n] + self.world.env_origins[:n, None]
            self.viewer.set_lookahead(points.reshape(-1, 2))
            self.viewer.sync(self.control_dt)
        return (self._obs, torch.as_tensor(reward, dtype=torch.float32, device=self.device),
                torch.as_tensor(dones, device=self.device), extras)

    def _observe(self):
        # Exactly one new scan/history update per control step, after any resets.
        _, yaw = self.world.robot_pose()
        self.ranges = self.lidar.scan(yaw)
        self.command_manager.compute(0.)
        return self.observation_manager.compute()

    def _measure_rewards(self, yaw, speed):
        # Progress uses the resulting pose; proximity uses the scan that informed the action.
        path = self.path_command
        points = self.lidar_origin + self.ranges[..., None] * np.stack(
            [np.cos(self.lidar.angles), np.sin(self.lidar.angles)], -1)
        distances = (np.linalg.norm(points, axis=-1) - self.radius if self.footprint is None
                     else polygon_distance(points, self.footprint))
        return RewardContext(
            progress=path.progress, cross_track=np.abs(path.cross_track), heading_error=wrap_angle(yaw - path.tangent),
            speed=speed, action=self.action_manager.action, prev_action=self.action_manager.prev_action,
            min_range=self.ranges.min(-1), clearance=distances.min(-1),
            success=self.termination_manager.get_outcome("success"),
            collision=self.termination_manager.get_outcome("collision"),
            stuck=self.termination_manager.get_outcome("stuck"), v_max=self.v_max, dt=self.control_dt,
            footprint_radius=self.radius,
        )

    def _episode_clearance(self, xy, yaw):
        """Evaluation measures resulting geometry; training reuses the action's lidar estimate."""
        if self.recorder.clearance_method == "visible_lidar_footprint":
            return self.reward_context.clearance
        clearance = np.array([
            footprint_clearance(xy[i], yaw[i], self.radius, self.footprint,
                                self.dataset.maps[self.map_idx[i]].primitives)
            for i in range(self.num_envs)
        ])
        clearance[self.contacts] = 0.
        return clearance

    def reset_case(self, map_index, pair_index, yaw, seed):
        """Start one explicit, materialized test case, independent of sampler order."""
        if self.num_envs != 1:
            raise ValueError("reset_case requires a single-environment test backend")
        self.rng = np.random.default_rng(seed)
        self.lidar._rng = np.random.default_rng(seed)
        self._reset_idx(np.array([0]), selection=(map_index, pair_index), initial_yaw=yaw)
        self.world.forward()
        self.last_episode_info = []
        self._obs = self._observe()
        return self._obs

    def _reset_idx(self, ids, selection=None, initial_yaw=None):
        log = self.reward_manager.reset(ids)
        for i in ids:
            keep = self.map_idx[i] >= 0 and self.episodes_on_map[i] < int(self.cfg.env.episodes_per_map)
            mi, pair = selection if selection is not None else self.sampler.sample(
                int(self.map_idx[i]) if keep and not self.sampler.sequential else None)
            md = self.dataset.maps[mi]
            count = self.episode_visits.get((mi, pair), 0)
            self.episode_visits[(mi, pair)] = count + 1
            self.episode_repeat[i] = count
            if mi != self.map_idx[i]:
                self.world.write_map(int(i), md.primitives, md.height)
                self.episodes_on_map[i] = 0
            self.map_idx[i], self.pair_idx[i] = mi, pair
            self.episodes_on_map[i] += 1
        self.command_manager.reset(ids)
        starts, yaws = [], []
        for i in ids:
            path = self.path_command.path[i, :self.path_command.path_len[i]]
            direction = path[min(5, len(path) - 1)] - path[0]
            yaw = (initial_yaw if initial_yaw is not None else
                   np.arctan2(direction[1], direction[0]) + self.rng.uniform(-1, 1) * self.spawn_yaw_noise())
            self.initial_yaw[i] = yaw
            self.initial_sensor_state[i] = deepcopy(self.lidar._rng.bit_generator.state)
            starts.append(path[0])
            yaws.append(yaw)
            if self.viewer_trigger is not None:
                self._episode_paths[int(i)] = path.copy()
            if self.viewer is not None and i < self._view_envs:
                self.viewer.set_episode(int(i), path[0], path[-1], path)
        self.world.set_robot_pose(ids, np.asarray(starts), np.asarray(yaws))
        self.action_manager.reset(ids)
        self.observation_manager.reset(ids)
        log.update(self.termination_manager.reset(ids))
        self.recorder.reset(ids)
        self.contacts[ids] = False
        self.episode_length_buf[torch.as_tensor(ids, device=self.device)] = 0
        return log

    def _poll_viewer(self):
        """Open or close the live view to follow ``viewer_trigger``; checked at most twice a second."""
        now = time.monotonic()
        if now < self._viewer_poll:
            return
        self._viewer_poll = now + 0.5
        wanted = self.viewer_trigger.exists()
        if self.viewer is not None and not (wanted and self.viewer.running):
            self.viewer.close()
            self.viewer, self._view_envs = None, 0
            self.viewer_trigger.unlink(missing_ok=True)  # also when the user closed the window
            print("Live view closed; training continues headless.", flush=True)
        elif wanted and self.viewer is None:
            # Sharded training keeps running at full speed; the view shows the first shard only.
            world = self.world.shards[0] if isinstance(self.world, ShardedWorld) else self.world
            try:
                from dwarl.simulation.viewer import SimViewer

                self.viewer = SimViewer(world, render_fps=float(self.cfg.sim.render_fps))
            except Exception as exc:  # e.g. no display: never interrupt training for a view
                self.viewer_trigger.unlink(missing_ok=True)
                warnings.warn(f"live view unavailable: {exc}", stacklevel=2)
                return
            self._view_envs = world.num_envs
            for i, path in self._episode_paths.items():
                if i < self._view_envs:
                    self.viewer.set_episode(i, path[0], path[-1], path)
            print(f"Live view: showing {self._view_envs} of {self.num_envs} environments.", flush=True)

    def spawn_yaw_noise(self):
        ec = self.cfg.env
        full = float(ec.spawn_yaw_noise)
        if not ec.get("spawn_yaw_curriculum", False):
            return full
        low = float(ec.get("spawn_yaw_min", .5))
        fraction = np.clip(self.sampler.level / float(ec.get("spawn_yaw_full_level", .6)), 0., 1.)
        return low + (full - low) * float(fraction)

    def _build_world(self, sc, rc, gui):
        fixed_map = self.dataset.maps[0] if self.dataset.index.get("static_map") else None
        if fixed_map is not None and len(self.dataset.maps) != 1:
            raise ValueError("static_map datasets must contain exactly one map")
        if fixed_map is None and self.dataset.max_primitives > int(sc.pool_size):
            raise ValueError(f"dataset needs pool_size >= {self.dataset.max_primitives}")
        kwargs = dict(tile_size=float(self.dataset.maps[0].size), tile_gap=float(sc.tile_gap),
                      pool_size=int(sc.pool_size), robot_xml=rc.xml, timestep=float(sc.timestep))
        if fixed_map is not None:
            kwargs["fixed_map"] = (fixed_map.primitives, fixed_map.height)
        shards = int(sc.get("num_shards", 0) or 0) or default_num_shards(self.num_envs)
        return ShardedWorld(self.num_envs, shards, **kwargs) if not gui and shards > 1 else TiledWorld(
            self.num_envs, **kwargs)

    def _check_dataset(self):
        radius = (self.dataset.index.get("config") or {}).get("robot_radius")
        if radius is not None and float(radius) + 1e-3 < self.radius:
            warnings.warn(f"dataset {self.cfg.dataset.path} was generated for robot_radius {float(radius):.2f} m < "
                          f"robot footprint radius {self.radius:.2f} m: paths may be infeasible", stacklevel=2)

    def close(self):
        try:
            if self.viewer is not None:
                self.viewer.close()
        finally:
            try:
                if self.world is not None:
                    self.world.close()
            finally:
                if self._robot_tmp is not None:
                    self._robot_tmp.cleanup()
