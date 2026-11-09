"""Demo worlds for deploying a trained policy under Nav2.

They are larger than the 8 m benchmark tiles, with a long global route, doors and aisles. They are stored
as a DWARL map dataset (``scenarios`` split), so the same geometry can be scored with
``dwarl test --suite benchmark --set test.dataset=<dir> --sim gazebo``.

``surprises`` are obstacles that exist in Gazebo but not on the Nav2 map: the global plan runs through
them and only the local controller (from the lidar) can avoid them. Only the visual demo uses them.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import ndimage

from dwarl.datasets.generation.astar import astar, finalize_path
from dwarl.datasets.generation.geometry import border_primitives, rasterize
from dwarl.datasets.generation.mapdata import BOX, MapData, path_length, prepare_dataset_directory
from dwarl.testing.scenarios import cyl, export_gazebo, wall

SPLIT, RES, HEIGHT, BORDER = "scenarios", 0.05, 0.8, 0.1


def box(x, y, hx, hy, yaw=0.0):
    return [BOX, x, y, yaw, hx, hy]


def _with_doors(x0, x1, y, doors, width=1.0):
    """Horizontal wall from x0 to x1 at y with door gaps of ``width`` centred at ``doors``."""
    out, start = [], x0
    for d in sorted(doors):
        out.append(wall(start, y, d - width / 2 - 0.05, y))
        start = d + width / 2 + 0.05
    return out + [wall(start, y, x1, y)]


def office():
    size = 16.0
    obs = _with_doors(-8, 8, -0.9, [-5.5, 0.0, 5.5]) + _with_doors(-8, 8, 0.9, [-5.0, 0.0, 5.0])
    obs += [wall(-3.0, -8, -3.0, -0.9), wall(3.0, -8, 3.0, -0.9), wall(-2.5, 0.9, -2.5, 8), wall(2.5, 0.9, 2.5, 8)]
    # Furniture: desks along walls, meeting tables with chairs, cabinets.
    obs += [box(-6.0, -6.5, 1.0, 0.4), box(-6.6, -3.0, 0.4, 0.8), cyl(-5.0, -3.4, 0.22), cyl(-4.3, -5.4, 0.2),
            box(0.0, -5.0, 0.9, 0.5), cyl(-1.2, -5.0, 0.22), cyl(1.2, -5.0, 0.22), cyl(0.0, -6.0, 0.22),
            box(5.5, -6.6, 1.4, 0.35), box(7.2, -3.5, 0.3, 1.0), cyl(4.2, -3.0, 0.25),
            box(-5.2, 4.8, 1.0, 0.6), cyl(-6.5, 4.8, 0.22), cyl(-3.9, 4.8, 0.22), box(-7.4, 2.2, 0.3, 0.6),
            box(0.0, 6.6, 1.6, 0.4), cyl(-1.3, 3.0, 0.3), box(1.2, 3.6, 0.35, 0.35, 0.4),
            box(6.2, 3.0, 0.6, 1.0), cyl(4.6, 6.2, 0.25), cyl(7.0, 6.6, 0.22), box(3.6, 2.0, 0.3, 0.3)]
    pairs = [((-5.6, -4.4), (6.0, 5.2)), ((-6.0, 2.6), (5.6, -4.6))]
    surprises = [box(-2.2, 0.15, 0.25, 0.35), cyl(2.6, -0.25, 0.25), cyl(5.0, 1.6, 0.2)]
    return size, obs, pairs, surprises


def warehouse():
    size = 16.0
    obs = []
    for y in (-5.2, -2.6, 0.0, 2.6, 5.2):  # rack rows with a cross aisle at x ~ 1.9
        obs += [box(-2.0, y, 3.4, 0.4), box(5.0, y, 2.0, 0.4)]
    obs += [box(-6.9, -6.9, 0.45, 0.45), box(-6.9, -5.6, 0.45, 0.45, 0.2), box(-7.0, 6.9, 0.45, 0.45),
            box(7.0, -7.0, 0.5, 0.5), cyl(-6.6, 1.0, 0.25), box(7.3, 6.0, 0.35, 0.8), box(-7.2, 3.6, 0.35, 0.6)]
    pairs = [((-6.6, -3.9), (6.0, 6.6)), ((5.8, -6.6), (-6.0, 1.4))]
    surprises = [box(-2.0, 1.0, 0.3, 0.3, 0.3), cyl(1.9, 1.3, 0.25), cyl(2.3, -3.9, 0.2)]
    return size, obs, pairs, surprises


WORLDS = {"office": office, "warehouse": warehouse}


def passage_radius(profile):
    """Lateral half-width for footprint robots (as DWARL's passage scenarios), else the radius."""
    if profile is None:
        return 0.18
    if profile.footprint is not None:
        return max(max(abs(y) for _, y in profile.footprint), 0.12)
    return float(profile.footprint_radius)


def build_world(name, radius=0.18, margin=0.1):
    size, obs, pairs, surprises = WORLDS[name]()
    prims = np.concatenate([border_primitives(size, BORDER), np.asarray(obs, np.float32).reshape(-1, 6)], 0)
    n = int(round(size / RES))
    occ = rasterize(prims, n, RES, size)
    md = MapData(occupancy=occ, resolution=RES, size=size, height=HEIGHT, primitives=prims, generator=name,
                 density=float(occ.mean()))
    clearance = ndimage.distance_transform_edt(~occ) * RES
    free = clearance >= radius + margin
    cost = 1.0 + 4.0 * np.exp(-np.maximum(clearance - radius, 0.0) / 0.25)
    spawns, idx, paths = [], [], []
    for s, g in pairs:
        cells = astar(free, tuple(md.xy_to_cell(np.asarray(s))), tuple(md.xy_to_cell(np.asarray(g))), cost)
        if cells is None:
            raise ValueError(f"demo world {name}: no path {s} -> {g} for radius {radius:.2f} m")
        raw = md.cell_to_xy(np.asarray(cells))
        raw[0], raw[-1] = s, g
        pts = finalize_path(raw, free, size, RES, 0.1)
        if pts is None:
            raise ValueError(f"demo world {name}: path violates clearance {s} -> {g}")
        paths.append(pts)
        idx.append((len(spawns), len(spawns) + 1))
        spawns += [s, g]
    md.spawns, md.pairs = np.asarray(spawns, np.float32), np.asarray(idx, np.int32)
    md.set_paths(paths)
    md.difficulty = np.asarray([path_length(p) / size for p in paths], np.float32)
    return md, np.asarray(surprises, np.float32).reshape(-1, 6)


def build_worlds(out_dir, robot=None, names=None, previews=True):
    """Write a DWARL dataset (index.json, scenarios/*.npz) and gazebo/<name>[_surprise].{sdf,yaml,pgm,json}."""
    profile = None
    if robot is not None:
        from dwarl.robot.profile import load_profile

        profile = load_profile(robot)
    radius = passage_radius(profile)
    out = prepare_dataset_directory(out_dir)
    (out / SPLIT).mkdir(parents=True, exist_ok=True)
    gazebo = out / "gazebo"
    files = []
    for name in names or list(WORLDS):
        md, surprises = build_world(name, radius)
        md.save(out / SPLIT / f"{name}.npz")
        files.append(f"{name}.npz")
        export_gazebo(md, gazebo, name)
        # Visual demo: Gazebo gets the surprises; the Nav2 map stays the planned (static) map.
        shown = MapData(occupancy=md.occupancy, resolution=md.resolution, size=md.size, height=md.height,
                        primitives=np.concatenate([md.primitives, surprises]), generator=name)
        shown.spawns, shown.pairs = md.spawns, md.pairs
        shown.set_paths([md.path(k) for k in range(md.num_pairs)])
        export_gazebo(shown, gazebo, f"{name}_surprise")
        (gazebo / f"{name}_surprise.pgm").unlink()
        (gazebo / f"{name}_surprise.yaml").write_text((gazebo / f"{name}.yaml").read_text())
        if previews:
            from dwarl.datasets.generation.pipeline import save_preview

            (out / "previews").mkdir(exist_ok=True)
            save_preview(shown, out / "previews" / f"{name}.png")
    index = {"config": {"kind": "demo_worlds", "robot_id": profile.id if profile else "diffbot",
                        "robot_radius": float(profile.footprint_radius) if profile else 0.18,
                        "passage_radius": radius, "profile_sha256": profile.sha256 if profile else None},
             "splits": {SPLIT: {"maps": files}}}
    (out / "index.json").write_text(json.dumps(index, indent=2))
    return out


if __name__ == "__main__":
    import sys

    print(build_worlds(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
