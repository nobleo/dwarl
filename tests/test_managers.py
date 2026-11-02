"""Manager contracts: term selection, units, partial resets and the actual environment boundary."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf, open_dict

from dwarl.navigation.managers import ActionManager, ObservationManager, RewardManager, TerminationManager
from dwarl.navigation.managers.base import ManagerTermBase
from dwarl.training.environment.nav_env import NavVecEnv


@pytest.mark.parametrize("configured,override,enabled", [
    (False, None, False), (True, None, True), (True, False, False), (False, True, True),
])
def test_gui_controls_viewer_and_world_selection(train_cfg, monkeypatch, configured, override, enabled):
    from unittest.mock import Mock

    from dwarl.simulation.world import ShardedWorld, TiledWorld

    viewer = Mock()
    monkeypatch.setattr("dwarl.simulation.viewer.SimViewer", viewer)
    train_cfg.sim.gui = configured
    train_cfg.sim.num_shards = 2
    env = NavVecEnv(train_cfg, gui=override)
    try:
        assert isinstance(env.world, TiledWorld if enabled else ShardedWorld)
        if enabled:
            viewer.assert_called_once()
            assert env.viewer is viewer.return_value
        else:
            viewer.assert_not_called()
            assert env.viewer is None
    finally:
        env.close()


def constant_reward(env, value):
    return np.full(env.num_envs, value)


def flags(env, name):
    return env.flags[name]


class CountObservation(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.count = np.zeros((env.num_envs, 1))

    def __call__(self, env):
        self.count += 1
        return self.count.copy()

    def reset(self, env_ids):
        self.count[env_ids] = 0


@pytest.mark.parametrize("mode,expected", [("window", [[.1, -.2], [.5, -1.]]),
                                          ("absolute", [[.1, -.2], [.5, -1.]])])
def test_action_processing_actuation_and_partial_reset(mode, expected):
    applied = []
    env = SimpleNamespace(num_envs=2, v_min=-.5, v_max=.5, w_max=1., a_max=1., alpha_max=2.,
                          control_dt=.1, wheel_r=.1, wheel_sep=.4,
                          world=SimpleNamespace(set_ctrl=lambda wheels: applied.append(wheels.copy())))
    manager = ActionManager({"drive": {"class_type": "dwarl.navigation.actions:DifferentialDriveAction",
                                       "params": {"mode": mode}, "clip": [-1., 1.]}}, env)
    drive = manager.get_term("drive")
    drive.command[1] = [.45, -.95]
    manager.process_action([[2., -2.], [1., -1.]])  # clipped before mapping
    np.testing.assert_allclose(drive.command, expected)
    manager.apply_action()
    np.testing.assert_allclose(applied[0], [[1.4, .6], [7., 3.]])
    manager.process_action(np.zeros((2, 2)))
    np.testing.assert_array_equal(manager.prev_action, [[1., -1.], [1., -1.]])
    command = drive.command.copy()
    manager.reset([0])
    np.testing.assert_array_equal(drive.command[0], [0., 0.])
    np.testing.assert_array_equal(drive.command[1], command[1])
    np.testing.assert_array_equal(manager.prev_action, [[0., 0.], [1., -1.]])
    with pytest.raises(ValueError, match="finite actions"):
        manager.process_action([[np.nan, 0], [0, 0]])


def test_custom_reward_term_dt_scaling_weight_changes_and_reset():
    cfg = OmegaConf.create({"rewards": {
        "custom": {"func": "tests.test_managers:constant_reward", "weight": 2., "params": {"value": 3.}},
        "disabled": None,
    }})
    env = SimpleNamespace(num_envs=2, cfg=cfg)
    manager = RewardManager(cfg.rewards, env)
    np.testing.assert_allclose(manager.compute(.1), [.6, .6])
    np.testing.assert_allclose(manager.compute(.05), [.3, .3])
    assert manager.reset([0])["Episode_Reward/custom"] == pytest.approx(.9)
    np.testing.assert_allclose(manager.episode_sums["custom"], [0., .9])
    manager.set_weights({"custom": 0.})
    assert cfg.rewards.custom.weight == 0.
    np.testing.assert_array_equal(manager.compute(.1), [0., 0.])
    manager.set_weights({"custom": 4.})
    np.testing.assert_allclose(manager.compute(.1), [1.2, 1.2])
    with pytest.raises(KeyError, match="unknown reward"):
        manager.set_weights({"typo": 1.})


def test_configured_observation_terms_order_scaling_and_stateful_reset():
    cfg = {"state": {"terms": {
        "counter": {"func": "tests.test_managers:CountObservation"},
        "scaled": {"func": "tests.test_managers:CountObservation", "clip": [0., 1.], "scale": 10.},
        "disabled": None,
    }}}
    manager = ObservationManager(cfg, SimpleNamespace(num_envs=2, device="cpu"))
    np.testing.assert_array_equal(manager.compute()["state"], [[1., 10.], [1., 10.]])
    manager.reset([0])
    np.testing.assert_array_equal(manager.compute()["state"], [[1., 10.], [2., 10.]])


def test_termination_takes_precedence_over_timeout_and_resets_individually():
    env = SimpleNamespace(num_envs=3, flags={"failure": np.array([True, False, False]),
                                           "timeout": np.array([True, True, False])})
    manager = TerminationManager({
        "timeout": {"func": "tests.test_managers:flags", "params": {"name": "timeout"}, "time_out": True},
        "failure": {"func": "tests.test_managers:flags", "params": {"name": "failure"}},
        "disabled": None,
    }, env)
    np.testing.assert_array_equal(manager.compute(), [True, True, False])
    np.testing.assert_array_equal(manager.terminated, [True, False, False])
    np.testing.assert_array_equal(manager.time_outs, [False, True, False])
    manager.reset([0])
    np.testing.assert_array_equal(manager.time_outs, [False, True, False])
    np.testing.assert_array_equal(env.flags["timeout"], [True, True, False])
    np.testing.assert_array_equal(env.flags["failure"], [True, False, False])


def test_auto_reset_clears_only_finished_environment_managers(train_cfg):
    train_cfg.env.num_envs = 2
    train_cfg.sim.num_shards = 1
    for name in ("success", "collision", "flipped", "stuck"):
        train_cfg.terminations[name] = None
    train_cfg.terminations.timeout.params.min_steps = 100
    train_cfg.terminations.timeout.params.time_factor = 0
    env = NavVecEnv(train_cfg)
    try:
        env.record = True
        env.termination_manager.terms["timeout"].func.max_steps[0] = 1
        obs, reward, done, extras = env.step(torch.ones((2, 2)))
        assert done.tolist() == [True, False]
        assert extras["time_outs"].tolist() == [True, False]
        assert torch.isfinite(reward).all()
        assert env.episode_length_buf.tolist() == [0, 1]
        np.testing.assert_array_equal(env.action_manager.action, [[0., 0.], [1., 1.]])
        np.testing.assert_array_equal(env.drive_action.command[0], [0., 0.])
        assert np.any(env.drive_action.command[1] != 0.)
        scan = obs["lidar"][0, 0].numpy()
        np.testing.assert_array_equal(scan, np.repeat(scan[:1], len(scan), axis=0))
        assert len(env.last_episode_info) == 1
        assert len(env.last_episode_info[0].trace["xy"]) == 1
        assert not env.recorder.traces[0] and len(env.recorder.traces[1]) == 1
        for values in env.reward_manager.episode_sums.values():
            assert values[0] == 0.
        env.step(torch.zeros((2, 2)))
        assert not env.last_episode_info  # completed episodes never leak into subsequent steps
    finally:
        env.close()


def test_export_rejects_unimplemented_runtime_observation_transform(train_cfg):
    from dwarl.artifacts.exporter import deploy_spec

    with open_dict(train_cfg):
        train_cfg.observations.state.terms.velocity.scale = 2.
    env = NavVecEnv(train_cfg)
    try:
        with pytest.raises(ValueError, match="unsupported deployment observation term"):
            deploy_spec(train_cfg, env)
    finally:
        env.close()


def test_checkpoint_restores_active_reward_weights(train_cfg):
    from dwarl.training.checkpoint import capture, restore

    env = NavVecEnv(train_cfg)
    try:
        env.reward_manager.set_weights({"cross_track": 0., "goal": 400.})
        state = capture(env)
        env.reward_manager.set_weights({"cross_track": 2., "goal": 300.})
        restore(env, state)
        assert env.reward_manager.weights["cross_track"] == 0.
        assert env.reward_manager.weights["goal"] == 400.
        assert train_cfg.rewards.goal.weight == 400.
    finally:
        env.close()
