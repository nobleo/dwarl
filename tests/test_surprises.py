"""Training obstacles that are not on the map: placement, avoidance level and relaxed path rewards."""

import numpy as np
import pytest
import torch
from omegaconf import open_dict

from dwarl.datasets.generation.geometry import border_primitives, rasterize
from dwarl.datasets.generation.mapdata import BOX, CYLINDER, MapData
from dwarl.navigation.surprises import primitive_distance, reachable, sample_on_path, settings

ROBOT = {"footprint_radius": 0.3, "footprint": [[0.1, 0.15], [-0.3, 0.15], [-0.3, -0.15], [0.1, -0.15]]}


def _map(walls=()):
    size, res = 10.0, 0.05
    prims = np.concatenate([border_primitives(size, 0.1), np.asarray(walls, np.float32).reshape(-1, 6)])
    return MapData(occupancy=rasterize(prims, int(size / res), res, size), resolution=res, size=size, height=0.5,
                   primitives=prims.astype(np.float32))


def _case():
    return {"reference_path": np.stack([np.linspace(-4, 4, 161), np.zeros(161)], -1).tolist()}


def test_primitive_distance_is_signed_and_ignores_padding():
    prims = np.array([[[CYLINDER, 0, 0, 0, 0.5, 0.5], [BOX, 3, 0, np.pi / 2, 1.0, 0.2]]])
    points = np.array([[[1.0, 0.0], [0.0, 0.0], [3.0, 0.9], [3.5, 0.0]]])
    d = primitive_distance(points, prims, np.array([[True, True]]))
    assert np.allclose(d, [[0.5, -0.5, -0.1, 0.3]])  # the box is rotated: 0.2 wide along x, 1.0 along y
    assert np.isinf(primitive_distance(points, prims, np.array([[False, False]]))).all()


def test_reachable_matches_four_connectivity():
    grid = np.zeros((5, 5), bool)
    grid[2, :] = True
    assert not reachable(grid, (0, 0), (4, 4))
    grid[2, 4] = False
    assert reachable(grid, (0, 0), (4, 4))
    grid[1, 4], grid[3, 3] = True, True  # only a diagonal link remains
    grid[2, 3] = True
    assert not reachable(grid, (0, 0), (4, 4))


def test_training_obstacles_sit_on_the_path_and_leave_a_detour():
    md, case = _map(), _case()
    path = np.asarray(case["reference_path"])
    rng = np.random.default_rng(1)
    s = settings({"probability": 1.0, "lateral": 0.5})
    found = 0
    for _ in range(10):
        prims = sample_on_path(md, path, ROBOT, rng, 3, s)
        found += len(prims)
        for prim in prims:
            assert np.linalg.norm(path - prim[1:3], axis=1).min() <= 0.5 * (prim[4] + 0.15) + 0.06
            assert min(np.linalg.norm(prim[1:3] - path[0]), np.linalg.norm(prim[1:3] - path[-1])) >= 1.0
    assert found >= 10
    walls = [[BOX, 0.0, 0.5, 0.0, 5.0, 0.05], [BOX, 0.0, -0.5, 0.0, 5.0, 0.05]]
    assert len(sample_on_path(_map(walls), path, ROBOT, rng, 3, s)) == 0  # no room for a detour


def test_settings_are_validated():
    assert settings(None)["probability"] == 0.0
    with pytest.raises(ValueError, match="unknown"):
        settings({"probabilty": 0.5})
    with pytest.raises(ValueError, match="count"):
        settings({"count": [2, 1]})


def _avoiding(train_cfg, **overrides):
    from dwarl.training.environment.nav_env import NavVecEnv

    train_cfg.env.num_envs = 2
    train_cfg.sim.num_shards = 1
    with open_dict(train_cfg):
        train_cfg.env.surprise_obstacles = {"probability": 1.0, "count": [1, 1], **overrides}
    return NavVecEnv(train_cfg)


def test_obstacle_ahead_relaxes_path_adherence_and_releases_after_it(train_cfg):
    env = _avoiding(train_cfg)
    try:
        sur = env.surprises
        assert sur.active and env.world.pool_size >= env.dataset.max_primitives + 1
        path = env.path_command
        # put one obstacle right on env 0's path, just ahead of the robot
        ahead = path.point_at(path.s + 0.6)[0]
        sur.prims[0] = 0.0
        sur.valid[0] = False
        sur.prims[0, 0] = [CYLINDER, *ahead, 0.0, 0.2, 0.2]
        sur.valid[0, 0] = True
        sur.valid[1] = False
        sur.level[:] = 0.0
        sur.update(0.1)
        assert sur.level[0] == pytest.approx(1.0) and sur.level[1] == 0.0
        assert sur.features()[0, 0] == pytest.approx(1.0) and sur.features()[0, 2] > 0
        _, yaw = env.world.robot_pose()
        env.reward_context = env._measure_rewards(yaw, np.zeros(2))
        assert env.reward_context.adherence[0] == pytest.approx(0.0)
        sur.valid[0] = False  # obstacle gone: the level decays with release_time instead of dropping
        sur.update(0.1)
        assert 0.9 < sur.level[0] < 1.0
        for _ in range(60):
            sur.update(0.1)
        assert sur.level[0] < 0.05
    finally:
        env.close()


def test_obstacles_are_written_into_the_world_and_seen_by_the_lidar(train_cfg):
    env = _avoiding(train_cfg, count=[2, 2])
    try:
        placed = [env.surprises.obstacles(i) for i in range(env.num_envs)]
        assert any(len(p) for p in placed)
        for i, prims in enumerate(placed):
            md = env.dataset.maps[env.map_idx[i]]
            used = env.world.shards[0].pool_geoms[i] if hasattr(env.world, "shards") else env.world.pool_geoms[i]
            model = env.world.shards[0].model if hasattr(env.world, "shards") else env.world.model
            active = int((model.geom_contype[used] > 0).sum())
            assert active == len(md.primitives) + len(prims)
        obs, reward, done, _ = env.step(torch.zeros((2, 2)))
        assert torch.isfinite(reward).all()
    finally:
        env.close()


def test_explicit_test_obstacles_replace_random_ones(train_cfg):
    env = _avoiding(train_cfg)
    env.close()
    from dwarl.training.environment.nav_env import NavVecEnv

    train_cfg.env.num_envs = 1
    with open_dict(train_cfg):
        train_cfg.env.surprise_obstacles = {"probability": 0.0}
    env = NavVecEnv(train_cfg, sequential=True)
    try:
        assert not env.surprises.active
        path = env.dataset.maps[0].path(0)
        prim = np.array([[CYLINDER, *path[len(path) // 2], 0.0, 0.15, 0.15]], np.float32)
        env.reset_case(0, 0, 0.0, 7, unmapped=prim)
        assert np.allclose(env.surprises.obstacles(0), prim)
        env.reset_case(0, 0, 0.0, 7)
        assert len(env.surprises.obstacles(0)) == 0
    finally:
        env.close()


def test_avoidance_preset_enables_obstacles_and_the_critic_signal():
    from dataclasses import asdict

    from dwarl.config import compose_config
    from dwarl.config.recipe import SurpriseObstacles
    from dwarl.navigation.surprises import DEFAULTS

    assert asdict(SurpriseObstacles()) == DEFAULTS  # the typed schema and the runtime defaults agree
    cfg = compose_config(assignments=["robot=diffbot"], preset="avoidance")
    assert cfg.recipe.env.surprise_obstacles.probability == 0.6
    assert "avoidance" in cfg.recipe.observations.privileged.terms
    assert "avoidance" not in compose_config().recipe.observations.privileged.terms
    with pytest.raises(ValueError, match="probability"):
        compose_config(assignments=["recipe.env.surprise_obstacles.probability=1.5"])


def test_saved_avoidance_config_composes_again(tmp_path):
    from omegaconf import OmegaConf

    from dwarl.config import compose_config

    saved = tmp_path / "config.yaml"
    OmegaConf.save(compose_config(assignments=["robot=diffbot"], preset="avoidance"), saved, resolve=True)
    cfg = compose_config(config=saved)  # what validation and dwarl test do with a run's config.yaml
    assert "avoidance" in cfg.recipe.observations.privileged.terms
    assert cfg.recipe.env.surprise_obstacles.probability == 0.6
