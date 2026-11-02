"""Hand-crafted test scenarios, stored in the regular map-dataset format.

``dwarl scenarios build`` writes ``data/maps/scenarios/{index.json, scenarios/<name>.npz}`` plus
``gazebo/<name>.{sdf,yaml,pgm,json}`` (Gazebo world, Nav2 map, start/goal) for the ROS side, so the
existing eval/bench code (and the ROS MuJoCo sim) can load them like any generated split. The
``generator`` field of each map is the scenario name, so per-generator eval metrics are per-scenario.

Frame: 8 x 8 m tile centred on the origin, bordered by walls. Each scenario is a list of obstacles plus
one or more (start, goal) pairs; the reference path is planned with the same clearance-aware A*.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib import resources
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

from dwarl.datasets.generation.astar import astar, finalize_path
from dwarl.datasets.generation.geometry import border_primitives, rasterize
from dwarl.datasets.generation.mapdata import BOX, CYLINDER, MapData, path_length, prepare_dataset_directory

SPLIT = "scenarios"
SIZE, RES, HEIGHT = 8.0, 0.05, 0.6
LAYOUT_VERSION = 4
PASSAGE_SCENARIOS = {"corridor", "narrow_corridor", "doorway", "tight_gap"}
Pair = tuple[tuple[float, float], tuple[float, float]]


def wall(x0: float, y0: float, x1: float, y1: float, t: float = 0.1) -> list[float]:
    """Axis-aligned wall segment from (x0, y0) to (x1, y1) with thickness t."""
    return [BOX, (x0 + x1) / 2, (y0 + y1) / 2, 0.0, max(abs(x1 - x0) / 2, t / 2), max(abs(y1 - y0) / 2, t / 2)]


def cyl(x: float, y: float, r: float) -> list[float]:
    return [CYLINDER, x, y, 0.0, r, r]


# ---------------------------------------------------------------------------- scenario definitions
def open_space():
    return [], [((-3.0, -3.0), (3.0, 3.0)), ((3.0, -3.0), (-3.0, 3.0))]


def corridor(passage_width: float = 1.0):
    half = passage_width / 2 + 0.05  # wall centre -> free surface is 0.05 m closer
    return [wall(-4, half, 4, half), wall(-4, -half, 4, -half)], [((-3.3, 0.0), (3.3, 0.0))]


def narrow_corridor(passage_width: float = 0.6):
    half = passage_width / 2 + 0.05
    return [wall(-4, half, 4, half), wall(-4, -half, 4, -half)], [((-3.3, 0.0), (3.3, 0.0))]


def doorway(passage_width: float = 0.8):
    half = passage_width / 2 + 0.05
    return [wall(0, -4, 0, -half), wall(0, half, 0, 4)], [((-3.0, 1.0), (3.0, -1.0)), ((2.5, 2.5), (-2.5, -2.5))]


def l_turn(corridor_width: float = 1.1):
    half, x0 = corridor_width / 2 + 0.05, 1.0
    x1 = x0 + corridor_width + 0.1
    obs = [wall(-4, half, x0, half), wall(-4, -half, x1, -half),
           wall(x0, half, x0, 4), wall(x1, -half, x1, 4)]
    return obs, [((-3.2, 0.0), ((x0 + x1) / 2, 3.2))]


def s_turn():
    return [wall(-1.5, -4, -1.5, 1.8), wall(1.5, -1.8, 1.5, 4)], [((-3.0, -3.0), (3.0, 3.0))]


def u_trap():
    obs = [wall(0.6, -1.2, 0.6, 1.2), wall(-0.8, 1.2, 0.6, 1.2), wall(-0.8, -1.2, 0.6, -1.2)]
    return obs, [((-3.0, 0.0), (3.0, 0.0))]


def dead_end(corridor_width: float = 1.1):
    half = corridor_width / 2 + 0.05
    obs = [wall(-4, half, 2.0, half), wall(-2.0, -half, 2.0, -half), wall(2.0, -half, 2.0, half)]
    return obs, [((1.0, 0.0), (3.2, 0.0))]


def tight_gap(passage_width: float = 1.2):
    # An impassable 0.3 m centre gap and a traversable opening at y=2.8 m.
    blocked_half, opening_half = 0.15 + 0.05, passage_width / 2 + 0.05
    return [wall(0, -4, 0, -blocked_half), wall(0, blocked_half, 0, 2.8 - opening_half),
            wall(0, 2.8 + opening_half, 0, 4)], [((-3.0, 0.0), (3.0, 0.0))]  # fmt: skip


def clutter(num_obstacles: int = 26, min_separation: float = 0.95, extent: float = 3.6,
            spawn_clearance: float = .23):
    rng = np.random.default_rng(7)
    obs, pts = [], []
    spawns = np.array([[-extent, -extent], [extent, extent], [-extent, 0.], [extent, 0.]])
    for _ in range(10000):
        if len(obs) == num_obstacles:
            break
        p = rng.uniform(-3.0, 3.0, 2)
        if (np.linalg.norm(spawns - p, axis=1).min() > spawn_clearance + .43 and
                all(np.linalg.norm(p - q) > min_separation for q in pts)):
            pts.append(p)
            obs.append(cyl(*p, rng.uniform(0.1, 0.25)) if rng.random() < 0.5 else
                       [BOX, *p, rng.uniform(0, np.pi), *rng.uniform(0.1, 0.3, 2)])  # fmt: skip
    if len(obs) != num_obstacles:
        raise ValueError("clutter layout cannot fit the requested obstacles and spawn clearance")
    return obs, [((-extent, -extent), (extent, extent)), ((-extent, 0.0), (extent, 0.0))]


SCENARIOS: dict[str, Callable[[], tuple[list, list[Pair]]]] = {
    f.__name__: f
    for f in (open_space, corridor, narrow_corridor, doorway, l_turn, s_turn, u_trap, dead_end, tight_gap, clutter)
}


# ---------------------------------------------------------------------------- building
def _planner_cfg() -> dict:
    return yaml.safe_load((resources.files("dwarl") / "config/recipes/data.yaml").read_text())


def build_scenario(name: str, cfg: dict | None = None) -> MapData:
    cfg = cfg or _planner_cfg()
    scale = float(cfg.get("geometry_scale", 1.0))
    size, res = SIZE * scale, RES * scale
    tight_passage = name in PASSAGE_SCENARIOS and "passage_width" in cfg
    if tight_passage:
        obs, pairs = SCENARIOS[name](passage_width=float(cfg["passage_width"]) / scale)
    elif name in ("l_turn", "dead_end") and "turn_corridor_width" in cfg:
        obs, pairs = SCENARIOS[name](corridor_width=float(cfg["turn_corridor_width"]) / scale)
    elif name == "clutter" and "clutter_obstacles" in cfg:
        obs, pairs = SCENARIOS[name](num_obstacles=int(cfg["clutter_obstacles"]),
                                     min_separation=float(cfg["clutter_separation"]) / scale,
                                      extent=float(cfg.get("clutter_extent", 3.6)),
                                      spawn_clearance=float(cfg["robot_radius"]) + float(cfg["inflation_margin"]))
    else:
        obs, pairs = SCENARIOS[name]()
    prims = np.concatenate([border_primitives(SIZE, float(cfg["border_thickness"])),
                             np.asarray(obs, np.float32).reshape(-1, 6)], 0)  # fmt: skip
    prims[:, [1, 2, 4, 5]] *= scale
    pairs = [(np.asarray(s) * scale, np.asarray(g) * scale) for s, g in pairs]
    n = int(round(size / res))
    occ = rasterize(prims, n, res, size)
    md = MapData(occupancy=occ, resolution=res, size=size, height=HEIGHT, primitives=prims, generator=name,
                 density=float(occ.mean()))  # fmt: skip
    r = float(cfg["passage_radius"] if tight_passage else cfg["robot_radius"])
    margin = float(cfg.get("passage_inflation_margin", 0.0) if tight_passage else cfg["inflation_margin"])
    clearance = ndimage.distance_transform_edt(~occ) * res
    free = clearance >= r + margin
    pl = cfg["planner"]
    cost = 1.0 + float(pl["clearance_weight"]) * np.exp(-np.maximum(clearance - r, 0.0) / float(pl["clearance_decay"]))
    spawns, idx, paths = [], [], []
    for s, g in pairs:
        # Cross a tight doorway on its centreline and normal to the wall. A
        # point robot can cross diagonally, but a rectangular footprint cannot.
        waypoints = [s, g]
        if tight_passage and name == "doorway":
            xs = (-1.0, 1.0) if s[0] < g[0] else (1.0, -1.0)
            waypoints = [s, *[(np.array([x, 0.0]) * scale) for x in xs], g]
        elif tight_passage and name == "tight_gap":
            xs = (-1.0, 1.0) if s[0] < g[0] else (1.0, -1.0)
            waypoints = [s, *[(np.array([x, 2.8]) * scale) for x in xs], g]
        if tight_passage and name in ("corridor", "narrow_corridor"):
            raw = np.asarray([s, g])
        else:
            cells: list[tuple[int, int]] = []
            for a, b in zip(waypoints[:-1], waypoints[1:], strict=True):
                segment = astar(free, tuple(md.xy_to_cell(a)), tuple(md.xy_to_cell(b)), cost)
                if segment is None:
                    raise ValueError(f"scenario {name}: no path {s} -> {g}")
                cells += segment if not cells else segment[1:]
            raw = md.cell_to_xy(np.asarray(cells))
            raw[0], raw[-1] = s, g
        pts = finalize_path(raw, free, md.size, md.resolution, float(pl["waypoint_spacing"]) * scale)
        if pts is None:
            raise ValueError(f"scenario {name}: path violates clearance {s} -> {g}")
        paths.append(pts)
        idx.append((len(spawns), len(spawns) + 1))
        spawns += [s, g]
    md.spawns = np.asarray(spawns, np.float32)
    md.pairs = np.asarray(idx, np.int32)
    md.set_paths(paths)
    md.difficulty = np.asarray([path_length(p) / size for p in paths], np.float32)
    return md


def build_scenarios(out_dir: str | Path, names: list[str] | None = None, previews: bool = True,
                    robot: str | None = None) -> dict:
    from dwarl.datasets.generation.pipeline import save_preview

    out = prepare_dataset_directory(out_dir)
    cfg = _planner_cfg()
    robot_id, profile_hash = "diffbot", None
    if robot is not None:
        from dwarl.robot.profile import load_profile
        profile = load_profile(robot)
        robot_id, profile_hash = profile.id, profile.sha256
        r, margin = profile.footprint_radius, float(cfg["inflation_margin"])
        cfg.update(robot_radius=r,
                   turn_corridor_width=max(1.1, 2 * (r + margin + 0.05)),
                   clutter_obstacles=min(26, max(6, round(26 * (0.18 / r) ** 2))),
                   clutter_separation=max(0.95, 2 * r + 0.2), clutter_extent=3.2)
        if profile.footprint is not None:
            # Passages should test the actual lateral footprint, not become
            # spacious by inheriting its much larger circumscribed radius.
            half_width = max(abs(y) for _, y in profile.footprint)
            cfg.update(passage_width=2 * (half_width + 0.06), passage_radius=half_width,
                       passage_inflation_margin=0.0)
    (out / SPLIT).mkdir(parents=True, exist_ok=True)
    files = []
    for name in names or list(SCENARIOS):
        md = build_scenario(name, cfg)
        md.save(out / SPLIT / f"{name}.npz")
        files.append(f"{name}.npz")
        export_gazebo(md, out / "gazebo", name)
        if previews:
            (out / "previews").mkdir(exist_ok=True)
            save_preview(md, out / "previews" / f"{name}.png")
    index = {"config": {"kind": "scenarios", "layout_version": LAYOUT_VERSION, "robot_id": robot_id,
                        "robot_radius": float(cfg["robot_radius"]), "profile_sha256": profile_hash,
                        "geometry_scale": float(cfg.get("geometry_scale", 1.)),
                        "inflation_margin": float(cfg["inflation_margin"]),
                        "passage_width": cfg.get("passage_width")},
             "splits": {SPLIT: {"maps": files}}}
    (out / "index.json").write_text(json.dumps(index, indent=2))
    return index


# ---------------------------------------------------------------------------- Gazebo / Nav2 export
_WORLD = """<?xml version="1.0"?>
<sdf version="1.8">
  <world name="default">
    <physics name="1ms" type="ignored">
      <max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <light name="sun" type="directional"><pose>0 0 10 0 0 0</pose><direction>-0.5 0.1 -0.9</direction>
      <diffuse>0.8 0.8 0.8 1</diffuse></light>
    <model name="ground_plane"><static>true</static><link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>50 50</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>50 50</size></plane></geometry></visual>
    </link></model>
    <model name="obstacles"><static>true</static><link name="link">
{geoms}
    </link></model>
  </world>
</sdf>
"""


def _sdf_geom(i: int, prim: np.ndarray, h: float) -> str:
    t, x, y, yaw, hx, hy = (float(v) for v in prim)
    g = (f"<cylinder><radius>{hx}</radius><length>{h}</length></cylinder>" if int(t) == CYLINDER
         else f"<box><size>{2 * hx} {2 * hy} {h}</size></box>")  # fmt: skip
    pose = f"<pose>{x} {y} {h / 2} 0 0 {yaw}</pose>"
    return (f'      <collision name="c{i}">{pose}<geometry>{g}</geometry></collision>\n'
            f'      <visual name="v{i}">{pose}<geometry>{g}</geometry>'
            f"<material><diffuse>0.4 0.4 0.45 1</diffuse></material></visual>")  # fmt: skip


def export_gazebo(md: MapData, out_dir: str | Path, name: str) -> None:
    """Write ``<name>.sdf`` (Gazebo world), ``<name>.yaml/.pgm`` (Nav2 map) and ``<name>.json`` (start/goal)."""
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    geoms = "\n".join(_sdf_geom(i, p, md.height) for i, p in enumerate(md.primitives))
    (out / f"{name}.sdf").write_text(_WORLD.replace("{geoms}", geoms))
    Image.fromarray(np.where(md.occupancy, 0, 254).astype(np.uint8)[::-1]).save(out / f"{name}.pgm")
    (out / f"{name}.yaml").write_text(
        f"image: {name}.pgm\nmode: trinary\nresolution: {md.resolution}\n"
        f"origin: [{-md.size / 2}, {-md.size / 2}, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n"
    )
    pairs = []
    for k in range(md.num_pairs):
        p = md.path(k).astype(np.float64)
        j = min(5, len(p) - 1)
        d0, d1 = p[j] - p[0], p[-1] - p[max(0, len(p) - 6)]
        pairs.append({"start": [*p[0].tolist(), float(np.arctan2(d0[1], d0[0]))],
                      "goal": [*p[-1].tolist(), float(np.arctan2(d1[1], d1[0]))],
                      "path_length": path_length(p)})  # fmt: skip
    (out / f"{name}.json").write_text(json.dumps({"name": name, "pairs": pairs}, indent=2))
