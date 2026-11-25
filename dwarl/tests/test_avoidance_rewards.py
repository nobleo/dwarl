"""A smoothness reward that sees weaving in window action mode."""

from types import SimpleNamespace

import numpy as np
import pytest

from dwarl.navigation.rewards import angular_effort, dewobble


def test_angular_effort_penalises_full_angular_acceleration():
    env = SimpleNamespace(alpha_max=2.0, control_dt=0.1, action_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(command_delta=np.array([[0.1, 0.2], [0.0, 0.0], [0.0, -0.1]]))))
    assert np.allclose(angular_effort(env), [-1.0, 0.0, -0.25])


def steering_reward(error, command, adherence=1., horizon=.5):
    env = SimpleNamespace(w_max=1., reward_context=SimpleNamespace(
        steering_error=np.asarray(error), steering_command=np.asarray(command),
        steering_adherence=np.asarray(adherence)))
    return dewobble(env, horizon=horizon)


def test_dewobble_charges_aligned_turns_but_not_needed_corrections():
    # Aligned: left or right steering costs; still or useful corrective turns do not.
    np.testing.assert_allclose(steering_reward([0., 0., 0., .3, -.3], [.5, -.5, 0., .5, -.5]),
                               [-.25, -.25, 0., 0., 0.])


def test_dewobble_wrong_way_overshoot_and_braking():
    # Error .1 rad permits 0..0.2 rad/s over the .5-second horizon.
    np.testing.assert_allclose(steering_reward([.1, .1, .1, .1], [-.5, .5, .3, .2]),
                               [-.25, -.09, -.01, 0.])
    # Zero action in window mode keeps turning: evaluate the command, not its increment.
    assert steering_reward(0., .5) < steering_reward(0., .2) < steering_reward(0., 0.)


def test_dewobble_wraps_heading_and_relaxes_during_detours():
    assert steering_reward(2 * np.pi + .3, .5) == 0.
    np.testing.assert_allclose(steering_reward([0., 0., 0.], [1., 1., 1.], [1., .5, 0.]),
                               [-1., -.5, 0.])


@pytest.mark.parametrize("horizon", [0., -1., float("nan"), float("inf")])
def test_dewobble_rejects_invalid_horizon(horizon):
    with pytest.raises(ValueError, match="horizon"):
        steering_reward(0., 0., horizon=horizon)


def test_dewobble_uses_pre_action_heading_and_command_before_safety(train_cfg, monkeypatch):
    import torch

    from dwarl.training.environment.nav_env import NavVecEnv

    train_cfg.env.num_envs = 1
    train_cfg.sim.num_shards = 1
    train_cfg.env.safety_layer.enabled = True
    env = NavVecEnv(train_cfg)
    try:
        xy, _ = env.world.robot_pose()
        env.world.set_robot_pose(np.array([0]), xy, np.zeros(1))
        env.world.forward()
        # Aligned with its lookahead target, already turning; zero increment keeps that turn.
        env.path_command.main_goal[:] = xy + [1., 0.]
        env.drive_action.command[:, 1] = env.w_max / 2
        monkeypatch.setattr(env.safety, "stops", lambda command: np.ones(1, dtype=bool))
        env.step(torch.zeros((1, 2)))
        np.testing.assert_allclose(env.reward_context.steering_error, [0.], atol=1e-12)
        np.testing.assert_allclose(env.reward_context.steering_command, [env.w_max / 2])
        np.testing.assert_allclose(env.drive_action.command, 0.)
        np.testing.assert_allclose(dewobble(env), [-.25])
    finally:
        env.close()
