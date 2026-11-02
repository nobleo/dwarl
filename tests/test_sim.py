import mujoco
import numpy as np
import pytest
import torch

from dwarl.datasets.generation.mapdata import BOX, CYLINDER
from dwarl.simulation.lidar import Lidar2D
from dwarl.simulation.maps import write_tile_map
from dwarl.simulation.world import TiledWorld


def make_world(n=2):
    w = TiledWorld(n, tile_size=8.0, pool_size=8)
    mujoco.mj_forward(w.model, w.data)
    return w


def test_robot_params_from_mjcf():
    w = make_world(1)
    assert np.isclose(w.robot.wheel_radius, 0.05) and np.isclose(w.robot.wheel_separation, 0.30)
    assert np.isclose(w.robot.footprint_radius, 0.18)


def test_lidar_matches_analytic_wall():
    w = make_world(2)
    # wall 2 m in front of robot 0 (face at x = 2.0), a cylinder r=0.5 at 3 m left of robot 1
    write_tile_map(w, 0, np.array([[BOX, 2.1, 0.0, 0.0, 0.1, 3.0]]), 1.0)
    write_tile_map(w, 1, np.array([[CYLINDER, 0.0, 3.0, 0.0, 0.5, 0.5]]), 1.0)
    w.set_robot_pose(np.array([0, 1]), np.zeros((2, 2)), np.zeros(2))
    mujoco.mj_forward(w.model, w.data)
    lidar = Lidar2D(w, num_rays=360, max_range=8.0)
    _, yaw = w.robot_pose()
    r = lidar.scan(yaw)
    front = np.argmin(np.abs(lidar.angles))
    left = np.argmin(np.abs(lidar.angles - np.pi / 2))
    assert abs(r[0, front] - 2.0) < 1e-3
    assert abs(r[1, left] - 2.5) < 1e-3
    assert r[1, front] == 8.0  # nothing ahead; neighbouring tile is not visible
    # rotate robot 0 by 90 deg -> wall now on its right
    w.set_robot_pose(np.array([0]), np.zeros((1, 2)), np.array([np.pi / 2]))
    mujoco.mj_forward(w.model, w.data)
    _, yaw = w.robot_pose()
    right = np.argmin(np.abs(lidar.angles + np.pi / 2))
    assert abs(lidar.scan(yaw)[0, right] - 2.0) < 1e-3


def test_map_swap_updates_collisions():
    w = make_world(2)
    write_tile_map(w, 0, np.array([[BOX, 0.0, 0.0, 0.0, 0.5, 0.5]]), 0.5)  # box on top of robot 0
    write_tile_map(w, 1, np.zeros((0, 6)), 0.5)
    w.set_robot_pose(np.array([0, 1]), np.zeros((2, 2)), np.zeros(2))
    mujoco.mj_forward(w.model, w.data)
    assert w.collided_envs().tolist() == [True, False]
    write_tile_map(w, 0, np.zeros((0, 6)), 0.5)  # remove -> no collision
    mujoco.mj_forward(w.model, w.data)
    assert not w.collided_envs().any()


@pytest.mark.parametrize("shards", [1, 2])
@pytest.mark.parametrize("reset_neighbor", [False, True])
def test_scan_uses_final_physics_pose(shards, reset_neighbor):
    from dwarl.simulation.world import ShardedWorld

    w = TiledWorld(2, pool_size=8) if shards == 1 else ShardedWorld(2, shards, pool_size=8)
    try:
        for e in range(2):
            w.write_map(e, np.array([[BOX, 2.1, 0., 0., .1, 3.]]), 1.)
        w.set_robot_pose(np.arange(2), np.zeros((2, 2)), np.zeros(2))
        w.forward()
        w.set_ctrl(np.full((2, 2), 10.))
        w.step(100)
        if reset_neighbor:
            w.set_robot_pose(np.array([1]), np.zeros((1, 2)), np.zeros(1))
            w.forward(np.array([1]))
        xy, yaw = w.robot_pose()
        assert xy[0, 0] > .1  # enough movement to expose a one-substep lag
        scan = w.cast_rays(yaw, np.array([0.]), 8.)
        # A full forward pass is the reference for final-state ray geometry.
        w.forward()
        np.testing.assert_allclose(scan, w.cast_rays(yaw, np.array([0.]), 8.), atol=1e-10, rtol=0)
        np.testing.assert_array_equal(w.robot_pose()[0], xy)
    finally:
        w.close()


def test_failed_shard_build_shuts_down_workers(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from dwarl.simulation import world

    pools = []

    def executor(*args, **kwargs):
        pool = ThreadPoolExecutor(*args, **kwargs)
        pools.append(pool)
        return pool

    def fail_build(*args, **kwargs):
        raise ValueError("invalid robot")

    monkeypatch.setattr(world, "ThreadPoolExecutor", executor)
    monkeypatch.setattr(world, "TiledWorld", fail_build)
    with pytest.raises(ValueError, match="invalid robot"):
        world.ShardedWorld(2, 2)
    assert not any(t.is_alive() for t in pools[0]._threads)
    with pytest.raises(RuntimeError, match="shutdown"):
        pools[0].submit(lambda: None)


@pytest.mark.parametrize("gui", [False, True])
def test_failed_env_reset_releases_resources(train_cfg, monkeypatch, gui):
    from pathlib import Path

    from dwarl.simulation import viewer
    from dwarl.training.environment.nav_env import NavVecEnv

    captured = {}

    class Viewer:
        closed = False

        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            self.closed = True

    def fail_reset(self):
        captured.update(world=self.world, viewer=self.viewer, temporary=Path(self._robot_tmp.name))
        raise ValueError("reset failed")

    train_cfg.sim.num_shards = 2
    monkeypatch.setattr(viewer, "SimViewer", Viewer)
    monkeypatch.setattr(NavVecEnv, "reset", fail_reset)
    with pytest.raises(ValueError, match="reset failed"):
        NavVecEnv(train_cfg, gui=gui)
    assert not captured["temporary"].exists()
    if gui:
        assert captured["viewer"].closed
    else:
        assert not any(t.is_alive() for t in captured["world"]._pool._threads)
        with pytest.raises(RuntimeError, match="shutdown"):
            captured["world"].step(1)


def test_failed_materialization_releases_temporary_directory(train_cfg, monkeypatch):
    from pathlib import Path

    from dwarl.training.environment import nav_env

    directories = []

    def fail_materialize(cfg, directory):
        directories.append(Path(directory))
        raise ValueError("build failed")

    monkeypatch.setattr(nav_env, "materialize", fail_materialize)
    with pytest.raises(ValueError, match="build failed"):
        nav_env.NavVecEnv(train_cfg)
    assert not directories[0].exists()


def test_env_step_and_reset(train_cfg):
    from dwarl.training.environment.nav_env import NavVecEnv

    env = NavVecEnv(train_cfg)
    obs = env.get_observations()
    H = len(train_cfg.observations.lidar.terms.scan.params.history)
    G = len(train_cfg.commands.path.params.lookahead_distances)
    assert obs["lidar"].shape == (4, 1, H, 180) and obs["state"].shape == (4, 3 * G + 4)
    assert obs["privileged"].shape[0] == 4
    assert float(obs["lidar"].min()) >= 0 and float(obs["lidar"].max()) <= 1
    done_seen = False
    for _ in range(400):
        obs, rew, dones, extras = env.step(torch.rand(4, 2) * 2 - 1)
        assert torch.isfinite(rew).all() and "time_outs" in extras
        done_seen |= bool(dones.any())
    assert done_seen
    assert (env.episode_length_buf <= env.max_episode_length).all()


def test_following_path_gives_progress(train_cfg):
    """Teleport the robot along its A* path: progress reward positive, cross-track ~0, success at end."""
    from dwarl.training.environment.nav_env import NavVecEnv

    env = NavVecEnv(train_cfg)
    e = 0
    path = env.path_command.path[e, : env.path_command.path_len[e]]
    total = 0.0
    for k in range(1, len(path)):
        s, cte, _ = env.path_command.project(np.where(np.arange(4)[:, None] == e, path[k], env.world.robot_pose()[0]))
        assert abs(cte[e]) < 0.05
        env.path_command.s_prev = s  # projection is restricted to arc lengths reachable from s_prev
        total = s[e]
    assert abs(total - env.path_command.path_total[e]) < 0.05


def test_progress_does_not_jump_across_walls(train_cfg):
    """A pose next to a much later path stretch must not be matched to it (fake progress)."""
    from dwarl.training.environment.nav_env import NavVecEnv

    env = NavVecEnv(train_cfg)
    e = 0
    path = env.path_command.path[e, : env.path_command.path_len[e]]
    far = path[min(len(path) - 1, 25)]  # >= a few metres of arc length ahead
    env.path_command.s_prev[:] = 0.0
    s, _, _ = env.path_command.project(np.where(np.arange(4)[:, None] == e, far, env.world.robot_pose()[0]))
    assert s[e] <= 2.0 * env.v_max * env.control_dt + 0.1 + 1e-6


def test_diffdrive_forward_speed(train_cfg):
    from dwarl.training.environment.nav_env import NavVecEnv

    train_cfg.env.num_envs = 1
    train_cfg.terminations.timeout.params.min_steps = 10**6
    train_cfg.terminations.stuck = None
    train_cfg.actions.drive.params.mode = "absolute"
    env = NavVecEnv(train_cfg)
    write_tile_map(env.world, 0, np.zeros((0, 6)), 0.5)
    env.world.set_robot_pose(np.array([0]), np.array([[-3.0, 0.0]]), np.zeros(1))
    env.path_command.goal[:] = 100.0
    speeds = []
    for _ in range(30):
        env.step(torch.tensor([[1.0, 0.0]]))
        _, yaw = env.world.robot_pose()
        speeds.append(env.world.robot_velocity(yaw)[0][0])
    assert 0.65 < np.mean(speeds[15:]) < 0.85


def test_sharded_world_matches_single_world(train_cfg):
    from omegaconf import OmegaConf

    from dwarl.simulation.world import ShardedWorld
    from dwarl.training.environment.nav_env import NavVecEnv

    envs = []
    for shards in (1, 2):
        cfg = OmegaConf.merge(train_cfg, {"sim": {"num_shards": shards}, "terminations": {"stuck": None}})
        envs.append(NavVecEnv(cfg, split="train"))
    single, sharded = envs
    assert isinstance(single.world, TiledWorld) and isinstance(sharded.world, ShardedWorld)
    # Compare in free space: obstacle contact is solved against a monolithic vs sharded linear
    # system (different sizes/orderings), and random driving into walls amplifies that ~1e-6
    # noise chaotically. Clearing the tiles keeps the two worlds near-bit-exact, which is what
    # this test is actually checking (state/ctrl/map routing), not solver contact noise.
    for env in (single, sharded):
        for e in range(env.num_envs):
            env.world.write_map(e, np.zeros((0, 6)), 0.5)
        env.world.set_robot_pose(np.arange(env.num_envs), np.zeros((env.num_envs, 2)), np.zeros(env.num_envs))
        env.path_command.goal[:] = 1e6  # never reach the old goal in free space
        env.world.forward()
    rng = np.random.default_rng(0)
    for _ in range(60):
        a = torch.tensor(rng.uniform(-1, 1, (4, 2)), dtype=torch.float32)
        _, r1, d1, _ = single.step(a)
        _, r2, d2, _ = sharded.step(a)
        assert torch.equal(d1, d2)
        np.testing.assert_allclose(single.world.robot_pose()[0], sharded.world.robot_pose()[0], atol=1e-6)
    sharded.close()


def test_window_action_is_always_executable():
    from dwarl.navigation.actions import command_from_action

    cmd = np.array([[0.0, 0.0], [0.45, -0.95]])
    a = np.array([[1.0, -1.0], [1.0, -1.0]])
    nxt = command_from_action(a, cmd, "window", -0.5, 0.5, 1.0, 1.0, 2.0, 0.1)
    assert np.allclose(nxt, [[0.1, -0.2], [0.5, -1.0]])  # one accel step, clipped to the limits
    nxt = command_from_action(np.zeros((2, 2)), cmd, "window", -0.5, 0.5, 1.0, 1.0, 2.0, 0.1)
    assert np.allclose(nxt, cmd)  # a = 0 holds the command


def test_lidar_history_and_spawn_yaw_curriculum(train_cfg):
    from dwarl.training.environment.nav_env import NavVecEnv

    env = NavVecEnv(train_cfg)
    hist = env.observation_manager.get_term("lidar", "scan").history
    lid = [env.get_observations()["lidar"][0, 0].numpy()]
    assert np.allclose(lid[0], lid[0][:1])  # fresh episode: all rows are the current scan
    for _ in range(hist[-1]):
        obs, *_ = env.step(torch.tensor([[1.0, 0.3]] * env.num_envs))
        lid.append(obs["lidar"][0, 0].numpy())
    if int(env.episode_length_buf[0]) == hist[-1]:  # env 0 did not reset meanwhile
        for k, lag in enumerate(hist):
            assert np.allclose(lid[-1][k], lid[-1 - lag][0])
    env.cfg.env.spawn_yaw_curriculum = True
    env.sampler.level = 0.0
    lo = env.spawn_yaw_noise()
    env.sampler.level = 1.0
    assert lo < env.spawn_yaw_noise() == float(env.cfg.env.spawn_yaw_noise)
