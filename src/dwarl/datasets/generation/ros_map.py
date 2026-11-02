"""Import a native-resolution ROS occupancy map (map_server YAML + image) as an independent test dataset."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

import numpy as np
import yaml
from omegaconf import OmegaConf
from PIL import Image
from scipy import ndimage

from dwarl.datasets.generation.astar import astar, finalize_path
from dwarl.datasets.generation.geometry import grid_to_primitives
from dwarl.datasets.generation.mapdata import MapData, path_length, prepare_dataset_directory
from dwarl.datasets.generation.pipeline import sample_spawns, save_preview


def read_map(source: str | Path) -> tuple[MapData, np.ndarray, dict]:
    """Keep native cells exactly; square padding is unknown, never navigable.

    Rectangle merging preserves the occupied-cell union exactly, including isolated
    scan pixels. There is no resizing, denoising, inflation or mesh convex hull.
    """
    source = Path(source).expanduser().resolve()
    spec = yaml.safe_load(source.read_text())
    resolution = float(spec["resolution"])
    origin = np.asarray(spec["origin"], dtype=float)
    free_t, occupied_t = float(spec["free_thresh"]), float(spec["occupied_thresh"])
    if not np.isfinite(resolution) or resolution <= 0 or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("invalid map resolution/origin")
    if spec.get("mode", "trinary") != "trinary" or not 0 <= free_t < occupied_t <= 1:
        raise ValueError("expected a trinary ROS occupancy map")
    image = (source.parent / spec["image"]).resolve()
    with Image.open(image) as im:
        pixels = np.asarray(im.convert("L"), dtype=np.uint8)[::-1]
    probability = pixels.astype(float) / 255
    if not int(spec.get("negate", 0)):
        probability = 1 - probability
    rows, cols = pixels.shape
    n = max(rows, cols)
    occupied, free = np.zeros((n, n), bool), np.zeros((n, n), bool)
    occupied[:rows, :cols] = probability > occupied_t
    # map_saver's conventional unknown value must not become a spawn region
    # merely because this source's free_thresh (0.25) also admits gray 205.
    free[:rows, :cols] = (probability < free_t) & (pixels != 205)
    size = n * resolution
    primitives = grid_to_primitives(occupied, resolution, size)
    md = MapData(occupied, resolution, size, 2.0, primitives,
                 generator=source.stem, density=float(occupied.mean()))
    c, s = np.cos(origin[2]), np.sin(origin[2])
    center = origin[:2] + np.array([[c, -s], [s, c]]) @ np.full(2, size / 2)
    provenance = {
        "sources": {key: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for key, p in (("yaml", source), ("image", image))},
        "source_shape": [rows, cols],
        "resolution": resolution,
        "planning_unknown_pixel": 205,
        "geometry": "exact union of occupied ROS map cells, extruded to 2 m; unknown cells have no geometry",
        "tile_to_source": {"translation_xy": center.tolist(), "yaw": float(origin[2]),
                           "convention": "source_xy = R(yaw) @ tile_xy + translation_xy"},
    }
    return md, free, provenance


def build_ros_map(source: str | Path, out_dir: str | Path,
                 robot: str = "diffbot", seed: int = 0, num_pairs: int = 24, log=print) -> dict:
    """Build only a test split, planning paths on the native-resolution grid."""
    from dwarl.robot.config import datagen_overrides

    out = prepare_dataset_directory(out_dir)
    if num_pairs < 1:
        raise ValueError("num_pairs must be positive")
    md, known_free, provenance = read_map(source)
    md.seed = seed
    base = yaml.safe_load((resources.files("dwarl") / "config/recipes/data.yaml").read_text())
    cfg = OmegaConf.to_container(OmegaConf.merge(base, datagen_overrides(robot)), resolve=True)
    r, margin = float(cfg["robot_radius"]), float(cfg["inflation_margin"])
    clearance = ndimage.distance_transform_edt(np.pad(known_free, 1))[1:-1, 1:-1] * md.resolution
    free = clearance >= r + margin
    rng = np.random.default_rng(seed)
    sp = cfg["spawn"]
    cells = sample_spawns(clearance, free, max(r + float(sp["extra_clearance"]), r + margin),
                          float(sp["min_separation"]), max(32, num_pairs), md.resolution, rng)
    if len(cells) < 2:
        raise ValueError("not enough safe spawns for this robot")
    md.spawns = md.cell_to_xy(cells).astype(np.float32)
    labels, _ = ndimage.label(free)
    pl = cfg["planner"]
    cost = 1 + float(pl["clearance_weight"]) * np.exp(-np.maximum(clearance - r, 0) / float(pl["clearance_decay"]))
    candidates = [(i, j) for i in range(len(cells)) for j in range(len(cells))
                  if i != j and np.linalg.norm(md.spawns[i] - md.spawns[j]) >= float(pl["min_distance"])]
    rng.shuffle(candidates)
    pairs, paths = [], []
    log(f"[real-maps] {md.generator}: native {md.resolution:g} m grid, {len(md.primitives)} exact rectangles")
    for i, j in candidates:
        if labels[tuple(cells[i])] != labels[tuple(cells[j])]:
            continue
        route = astar(free, tuple(cells[i]), tuple(cells[j]), cost)
        if route is None:
            continue
        raw = md.cell_to_xy(np.asarray(route))
        path = finalize_path(raw, free, md.size, md.resolution, float(pl["waypoint_spacing"]))
        if path is None:
            continue
        pairs.append((i, j))
        paths.append(path)
        log(f"[real-maps] route {len(pairs)}/{num_pairs}: {path_length(path):.1f} m")
        if len(pairs) == num_pairs:
            break
    if len(pairs) != num_pairs:
        raise ValueError(f"found only {len(pairs)} valid routes; request fewer pairs or change seed")
    md.pairs = np.asarray(pairs, np.int32)
    md.set_paths(paths)
    md.difficulty = np.asarray([path_length(p) / md.size for p in paths], np.float32)
    index = {
        "config": {"kind": "real_buildings", "robot_radius": r, "robot_id": cfg["robot_id"],
                   "inflation_margin": margin, "seed": seed, "num_pairs": num_pairs},
        **provenance,
        "static_map": True,
        "splits": {"test": {"maps": [f"{md.generator}.npz"], "generators": {md.generator: 1}}},
    }
    (out / "test").mkdir(parents=True, exist_ok=True)
    md.save(out / "test" / f"{md.generator}.npz")
    (out / "previews").mkdir(exist_ok=True)
    save_preview(md, out / "previews" / f"{md.generator}.png", scale=1)
    (out / "index.json").write_text(json.dumps(index, indent=2))
    log(f"[real-maps] wrote {out} (test only; training defaults unchanged)")
    return index
