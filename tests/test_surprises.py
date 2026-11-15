"""Training obstacles that are not on the map: placement, world and test cases."""

import numpy as np
import pytest
import torch
from omegaconf import open_dict

from dwarl.datasets.generation.geometry import border_primitives, rasterize
from dwarl.datasets.generation.mapdata import BOX, CYLINDER, MapData
from dwarl.navigation.surprises import reachable, sample_on_path, settings

ROBOT = {"footprint_radius": 0.3, "footprint": [[0.1, 0.15], [-0.3, 0.15], [-0.3, -0.15], [0.1, -0.15]]}


def _map(walls=()):
    size, res = 10.0, 0.05
    prims = np.concatenate([border_primitives(size, 0.1), np.asarray(walls, np.float32).reshape(-1, 6)])
    return MapData(occupancy=rasterize(prims, int(size / res), res, size), resolution=res, size=size, height=0.5,
                   primitives=prims.astype(np.float32))


def _case():
    return {"reference_path": np.stack([np.linspace(-4, 4, 161), np.zeros(161)], -1).tolist()}


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

