import numpy as np
import pytest
from scipy import ndimage

from dwarl.datasets.generation.astar import astar, finalize_path, path_is_clear, resample_path
from dwarl.datasets.generation.generators import GENERATORS
from dwarl.datasets.generation.geometry import grid_to_rects, rasterize, rects_to_primitives
from dwarl.datasets.generation.mapdata import MapData, path_length
from dwarl.datasets.generation.pipeline import generate_dataset, generate_map

from .conftest import datagen_cfg


def test_astar_straight_and_blocked():
    free = np.ones((20, 20), bool)
    p = astar(free, (0, 0), (0, 19))
    assert p[0] == (0, 0) and p[-1] == (0, 19) and len(p) == 20
    free[:, 10] = False
    assert astar(free, (0, 0), (0, 19)) is None
    free[15, 10] = True
    p = astar(free, (0, 0), (0, 19))
    assert (15, 10) in p and all(free[c] for c in p)


def test_astar_no_corner_cutting():
    free = np.ones((3, 3), bool)
    free[0, 1] = free[1, 0] = False
    assert astar(free, (0, 0), (1, 1)) is None


def test_rects_cover_grid_exactly(rng):
    g = rng.random((30, 30)) < 0.4
    cover = np.zeros_like(g)
    for r0, c0, r1, c1 in grid_to_rects(g):
        assert not cover[r0:r1, c0:c1].any()
        cover[r0:r1, c0:c1] = True
    assert (cover == g).all()
    prims = rects_to_primitives(grid_to_rects(g), 0.2, 6.0)
    # rasterisation is conservative (grown by half a cell): every occupied cell stays occupied
    assert rasterize(prims, 30, 0.2, 6.0)[g].all()


def test_resample_spacing():
    p = resample_path(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), 0.1)
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    assert abs(path_length(p) - 2.0) < 1e-2 and d.max() < 0.11


def test_path_is_clear_detects_corner_cutting():
    res = 0.1
    size = 11 * res
    free = np.ones((11, 11), bool)
    free[5, 5] = free[4, 5] = free[5, 4] = False

    def xy(r, c):
        return np.array([-size / 2 + (c + 0.5) * res, -size / 2 + (r + 0.5) * res])

    # Straight diagonal from (4, 4) to (6, 6) cuts the blocked corner at (5, 5).
    diagonal = np.stack([xy(4, 4), xy(6, 6)])
    assert not path_is_clear(diagonal, free, size, res)
    # A detour around the corner stays entirely on free cells.
    around = np.stack([xy(4, 4), xy(4, 3), xy(6, 3), xy(6, 6)])
    assert path_is_clear(around, free, size, res)


def test_path_checks_short_corner_crossings_and_boundary_contact():
    free = np.ones((4, 4), bool)
    free[2, 2] = False
    # Only 0.7 mm of this segment crosses the blocked cell, between old samples.
    path = np.array([[-.02, .0205], [.03, -.0295]])
    assert not path_is_clear(path, free, .2, .05)
    assert not path_is_clear(np.array([[-.02, .02], [.03, -.03]]), free, .2, .05)
    assert path_is_clear(path - .01, free, .2, .05)
    assert not path_is_clear(np.array([[.025, .025]]), free, .2, .05)


def test_finalization_checks_raw_fallback():
    free = np.ones((4, 4), bool)
    free[:, 2] = False
    raw = np.array([[-.075, .025], [.075, .025]])
    assert finalize_path(raw, free, .2, .05, .02) is None


def test_scenario_rejects_undersized_corridor():
    from dwarl.testing.scenarios import build_scenario

    cfg = datagen_cfg()
    cfg.update(passage_width=.2, passage_radius=.18, passage_inflation_margin=0.)
    with pytest.raises(ValueError, match="path violates clearance"):
        build_scenario("narrow_corridor", cfg)


@pytest.mark.parametrize("builder", ["procedural", "scenarios", "office"])
@pytest.mark.parametrize("existing", ["index.json", "train/map_00000.npz"])
def test_dataset_build_refuses_nonempty_destination(tmp_path, builder, existing):
    from dwarl.datasets.generation.ros_map import build_office
    from dwarl.testing.scenarios import build_scenarios

    file = tmp_path / existing
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(b"existing dataset")
    with pytest.raises(FileExistsError, match="must be empty"):
        if builder == "procedural":
            generate_dataset(datagen_cfg(), tmp_path, workers=1)
        elif builder == "scenarios":
            build_scenarios(tmp_path, previews=False)
        else:
            build_office(source=tmp_path / "source.yaml", out_dir=tmp_path)
    assert file.read_bytes() == b"existing dataset"
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == [file]


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_generators_produce_valid_maps(name):
    cfg = datagen_cfg()
    cfg["generators"] = {name: cfg["generators"][name]}
    md = next(m for m in (generate_map(cfg, s) for s in range(10)) if m is not None)
    assert md.generator == name and len(md.primitives) <= cfg["max_primitives"]
    assert md.num_pairs >= cfg["planner"]["min_pairs"]
    clearance = ndimage.distance_transform_edt(~md.occupancy) * md.resolution
    free = clearance >= float(cfg["robot_radius"]) + float(cfg["inflation_margin"])
    lab, num = ndimage.label(free, structure=np.ones((3, 3)))
    if num:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        free = lab == counts.argmax()
    for s in md.spawns:  # spawns are on free space
        assert not md.occupancy[tuple(md.xy_to_cell(s))]
    for k in range(md.num_pairs):
        path = md.path(k)
        i, j = md.pairs[k]
        assert np.allclose(path[0], md.spawns[i], atol=1e-4) and np.allclose(path[-1], md.spawns[j], atol=1e-4)
        assert not md.occupancy[tuple(md.xy_to_cell(path).T)].any()  # path never crosses an obstacle
        assert path_is_clear(path, free, md.size, md.resolution)  # segments too, not just points


def test_mapdata_roundtrip(tmp_path):
    md = generate_map(datagen_cfg(), 3)
    assert md is not None
    md.save(tmp_path / "m.npz")
    md2 = MapData.load(tmp_path / "m.npz")
    assert (md2.occupancy == md.occupancy).all() and np.allclose(md2.primitives, md.primitives)
    assert np.allclose(md2.path(1), md.path(1)) and md2.generator == md.generator
