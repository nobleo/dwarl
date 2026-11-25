"""Office and warehouse layouts for the standard benchmark builder.

Unmapped obstacles are materialized by the testing workflow, not embedded in these maps.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from dwarl.datasets.generation.astar import astar, finalize_path
from dwarl.datasets.generation.geometry import border_primitives, rasterize
from dwarl.datasets.generation.mapdata import BOX, MapData, path_length
from dwarl.testing.scenarios import cyl, wall

RES, HEIGHT, BORDER = 0.05, 0.8, 0.1


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
    hall = 1.45  # wall centrelines: 2.8 m of free corridor between the inner faces
    obs = _with_doors(-8, 8, -hall, [-5.5, 0.0, 5.5], width=1.8) + _with_doors(-8, 8, hall, [-5.0, 0.0, 5.0], width=1.8)
    obs += [wall(-3.0, -8, -3.0, -hall), wall(3.0, -8, 3.0, -hall), wall(-2.5, hall, -2.5, 8), wall(2.5, hall, 2.5, 8)]
    # Furniture: desks along walls, meeting tables with chairs, cabinets.
    obs += [box(-6.0, -6.5, 1.0, 0.4), box(-6.6, -3.0, 0.4, 0.8), cyl(-5.0, -3.4, 0.22),
            box(0.0, -5.0, 0.9, 0.5), cyl(-1.2, -5.0, 0.22), cyl(1.2, -5.0, 0.22), cyl(0.0, -6.0, 0.22),
            box(5.5, -6.6, 1.4, 0.35), box(7.2, -3.5, 0.3, 1.0), cyl(4.2, -3.0, 0.25),
            cyl(-6.5, 4.8, 0.22), cyl(-3.9, 4.8, 0.22), box(-7.4, 2.2, 0.3, 0.6),
            box(0.0, 6.6, 1.6, 0.4), cyl(-1.3, 3.0, 0.3), box(1.2, 3.6, 0.35, 0.35, 0.4),
            box(6.2, 3.0, 0.6, 1.0), cyl(4.6, 6.2, 0.25), cyl(7.0, 6.6, 0.22), box(3.6, 2.0, 0.3, 0.3)]
    # Clear a turning pocket beside each spawn and landing, keeping the furnished rooms.
    pairs = [((-5.0, -4.95), (5.7, 4.7)), ((-5.1, 3.2), (5.6, -4.6))]
    return size, obs, pairs


def warehouse():
    size = 16.0
    obs = []
    for y in (-5.2, -2.6, 0.0, 2.6, 5.2):  # rack rows with a cross aisle at x ~ 1.9
        obs += [box(-2.0, y, 3.4, 0.4), box(5.0, y, 2.0, 0.4)]
    obs += [box(-6.9, -6.9, 0.45, 0.45), box(-6.9, -5.6, 0.45, 0.45, 0.2), box(-7.0, 6.9, 0.45, 0.45),
            box(7.0, -7.0, 0.5, 0.5), cyl(-6.6, 1.0, 0.25), box(7.3, 6.0, 0.35, 0.8), box(-7.2, 3.6, 0.35, 0.6)]
    pairs = [((-6.6, -3.9), (6.0, 6.6)), ((5.8, -6.6), (-6.0, 1.4))]
    return size, obs, pairs


WORLDS = {"office": office, "warehouse": warehouse}


def build_world(name, radius=0.18, margin=0.1):
    size, obs, pairs = WORLDS[name]()
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
            raise ValueError(f"benchmark world {name}: no path {s} -> {g} for radius {radius:.2f} m")
        raw = md.cell_to_xy(np.asarray(cells))
        raw[0], raw[-1] = s, g
        pts = finalize_path(raw, free, size, RES, 0.1)
        if pts is None:
            raise ValueError(f"benchmark world {name}: path violates clearance {s} -> {g}")
        paths.append(pts)
        idx.append((len(spawns), len(spawns) + 1))
        spawns += [s, g]
    md.spawns, md.pairs = np.asarray(spawns, np.float32), np.asarray(idx, np.int32)
    md.set_paths(paths)
    md.difficulty = np.asarray([path_length(p) / size for p in paths], np.float32)
    return md
