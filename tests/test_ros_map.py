
import mujoco
import numpy as np
import pytest
import yaml
from PIL import Image
from scipy import ndimage

from dwarl.datasets.generation.ros_map import build_ros_map, read_map
from dwarl.simulation.world import TiledWorld
from dwarl.training.environment.dataset import MapDataset


def write_map(tmp_path, pixels, resolution=0.01):
    path = tmp_path / "building.yaml"
    Image.fromarray(pixels).save(tmp_path / "map.pgm")
    path.write_text(yaml.safe_dump({
        "image": "map.pgm", "resolution": resolution, "origin": [-2, -3, 0],
        "negate": 0, "free_thresh": 0.25, "occupied_thresh": 0.65,
    }))
    return path


@pytest.fixture()
def source(tmp_path):
    pixels = np.full((120, 120), 254, np.uint8)
    pixels[:3] = pixels[-3:] = 0
    pixels[:, :3] = pixels[:, -3:] = 0
    pixels[50:65, 50:65] = 0
    pixels[20:30, 80:90] = 205
    return write_map(tmp_path, pixels, 0.1)


def test_native_pixels_and_exact_geometry(tmp_path):
    pixels = np.array([[0, 0, 205, 254], [254, 0, 254, 0], [254, 254, 254, 0]], np.uint8)
    md, free, meta = read_map(write_map(tmp_path, pixels))
    assert md.resolution == 0.01 and md.size == 0.04
    assert meta["source_shape"] == [3, 4]
    np.testing.assert_array_equal(md.occupancy[:3], (pixels == 0)[::-1])
    np.testing.assert_array_equal(free[:3], (pixels == 254)[::-1])
    assert not free[3].any() and not md.occupancy[3].any()
    # Reconstruct rectangles without rasterize's conservative half-cell growth.
    cover = np.zeros_like(md.occupancy)
    for _, x, y, _, hx, hy in md.primitives:
        c0, c1 = np.rint((np.array([x - hx, x + hx]) + md.size / 2) / md.resolution).astype(int)
        r0, r1 = np.rint((np.array([y - hy, y + hy]) + md.size / 2) / md.resolution).astype(int)
        assert not cover[r0:r1, c0:c1].any()
        cover[r0:r1, c0:c1] = True
    np.testing.assert_array_equal(cover, md.occupancy)


def test_ros_map_routes_and_test_only_split(source, tmp_path):
    out = tmp_path / "building"
    index = build_ros_map(source, out, num_pairs=3, log=lambda _: None)
    assert set(index["splits"]) == {"test"} and index["static_map"]
    ds = MapDataset(out, "test")
    md = ds.maps[0]
    assert md.num_pairs == 3
    known = read_map(source)[1]
    clearance = ndimage.distance_transform_edt(np.pad(known, 1))[1:-1, 1:-1] * md.resolution
    for k in range(md.num_pairs):
        p = md.path(k)
        i, j = md.pairs[k]
        np.testing.assert_allclose(p[0], md.spawns[i], atol=1e-6)
        np.testing.assert_allclose(p[-1], md.spawns[j], atol=1e-6)
        assert (clearance[tuple(md.xy_to_cell(p).T)] >= 0.23).all()
    with pytest.raises(KeyError):
        MapDataset(out, "train")
    with pytest.raises(FileExistsError):
        build_ros_map(source, out)
    other = tmp_path / "same_seed"
    build_ros_map(source, other, num_pairs=3, log=lambda _: None)
    again = MapDataset(other, "test").maps[0]
    np.testing.assert_array_equal(md.path_points, again.path_points)
    np.testing.assert_array_equal(md.pairs, again.pairs)


def test_fixed_geometry_lidar_contacts_and_no_swapping():
    from dwarl.simulation.lidar import Lidar2D

    primitives = np.array([[0, 2.1, 0, 0, 0.1, 3]], dtype=np.float32)
    world = TiledWorld(1, fixed_map=(primitives, 2.0))
    assert world.pool_size == 1
    assert not world.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MIDPHASE
    world.set_robot_pose(np.array([0]), np.array([[0, 0]]), np.array([0]))
    world.forward()
    scan = Lidar2D(world, num_rays=360).scan(np.zeros(1))
    assert abs(scan[0, 180] - 2) < 1e-5
    world.write_map(0, primitives.copy(), 2.0)
    with pytest.raises(ValueError, match="immutable"):
        world.write_map(0, primitives + 1, 2.0)
    world.set_robot_pose(np.array([0]), np.array([[2.1, 0]]), np.array([0]))
    world.forward()
    assert world.collided_envs()[0]


@pytest.mark.parametrize("shards", [1, 2])
def test_fixed_dataset_env(source, tmp_path, train_cfg, shards):
    import torch

    from dwarl.training.environment.nav_env import NavVecEnv

    out = tmp_path / "sim"
    build_ros_map(source, out, num_pairs=3, log=lambda _: None)
    train_cfg.dataset.path = str(out)
    train_cfg.env.num_envs = 2
    train_cfg.sim.num_shards = shards
    train_cfg.sim.pool_size = 1  # fixed geometry is not constrained by the procedural pool
    env = NavVecEnv(train_cfg, split="test", sequential=True)
    try:
        obs, reward, _, _ = env.step(torch.zeros((2, 2)))
        assert torch.isfinite(obs["lidar"]).all() and torch.isfinite(reward).all()
        assert not env.world.collided_envs().any()
    finally:
        env.close()
