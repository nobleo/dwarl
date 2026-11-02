"""Spawn sampling, pair planning and dataset generation."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from dwarl.datasets.generation.astar import astar, finalize_path
from dwarl.datasets.generation.generators import GENERATORS
from dwarl.datasets.generation.geometry import border_primitives, rasterize
from dwarl.datasets.generation.mapdata import MapData, path_length, prepare_dataset_directory
from dwarl.navigation.clearance import path_width_clear


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, num = ndimage.label(mask, structure=np.ones((3, 3)))
    if num == 0:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == counts.argmax()


def sample_spawns(
    clearance: np.ndarray, free: np.ndarray, min_clearance: float, min_sep: float, num: int, res: float, rng
) -> np.ndarray:
    """Pick up to ``num`` free cells with clearance >= ``min_clearance`` that are >= ``min_sep`` apart."""
    cand = np.argwhere(free & (clearance >= min_clearance))
    rng.shuffle(cand)
    chosen: list[np.ndarray] = []
    sep_cells2 = (min_sep / res) ** 2
    for rc in cand:
        if all(((rc - q) ** 2).sum() >= sep_cells2 for q in chosen):
            chosen.append(rc)
            if len(chosen) >= num:
                break
    return np.asarray(chosen, dtype=np.int64).reshape(-1, 2)


def generate_map(cfg: dict, seed: int) -> MapData | None:
    """Generate one map with spawns and A* paths. Returns None when the sample is rejected."""
    rng = np.random.default_rng(seed)
    size, res = float(cfg["size"]), float(cfg["resolution"])
    n = int(round(size / res))
    names = list(cfg["generators"].keys())
    weights = np.array([float(cfg["generators"][k].get("weight", 1.0)) for k in names])
    name = names[rng.choice(len(names), p=weights / weights.sum())]
    params = {k: v for k, v in cfg["generators"][name].items() if k != "weight"}

    robot_r = float(cfg["robot_radius"])
    # Axis-aligned generators (maze walls, room doors) are traversed along their axes, so a rectangular
    # footprint only needs its lateral half-width there, not its circumscribed radius. Obstacles the robot
    # must rotate around (clutter, caves, ros maps) keep the circumscribed inflation.
    footprint = cfg.get("footprint")  # list of [x, y] or None for a circular footprint
    use_narrow = name in ("maze", "rooms") and footprint is not None
    r_free = float(cfg.get("passage_radius", robot_r)) if use_narrow else robot_r

    if name == "maze" and params.get("corridor") is None:
        # Size corridors to the robot: the A* inflation needs a free passage of at least
        # 2*(r_free + inflation_margin) + wall_thickness + resolution, so derive the cell width
        # from that floor plus a configurable clearance band (keeps mazes tight but traversable
        # for whatever robot the dataset is generated for, instead of one fixed range for all).
        lo, hi = params.pop("corridor_clearance", [0.1, 0.5])
        margin = float(cfg["inflation_margin"])
        base = 2.0 * (r_free + margin) + float(params.get("wall_thickness", 0.1)) + res
        params["corridor"] = (base + float(lo), base + float(hi))
    if name == "maze" and not use_narrow:
        # Circular robots already spawn inside ordinary corridors; plazas would only loosen the maze,
        # so only rectangular footprints (whose tight corridors cannot host a random-heading spawn) get
        # plaza cells.
        params["plaza"] = 0
    interior = GENERATORS[name](rng, size, **params)
    prims = np.concatenate([border_primitives(size, float(cfg["border_thickness"])), interior], 0)
    if len(prims) > int(cfg["max_primitives"]):
        return None
    height = float(rng.uniform(*cfg["height"]))
    occ = rasterize(prims, n, res, size)

    clearance = ndimage.distance_transform_edt(~occ) * res
    free = _largest_component(clearance >= r_free + float(cfg["inflation_margin"]))
    sp_cfg = cfg["spawn"]
    spawns_rc = sample_spawns(
        clearance, free, robot_r + float(sp_cfg["extra_clearance"]), float(sp_cfg["min_separation"]),
        int(sp_cfg["num"]), res, rng,
    )  # fmt: skip
    if len(spawns_rc) < int(sp_cfg["min_num"]):
        return None

    md = MapData(occupancy=occ, resolution=res, size=size, height=height, primitives=prims, generator=name,
                 seed=seed, density=float(occ.mean()))  # fmt: skip
    md.spawns = md.cell_to_xy(spawns_rc).astype(np.float32)

    pl = cfg["planner"]
    cell_cost = 1.0 + float(pl["clearance_weight"]) * np.exp(
        -np.maximum(clearance - r_free, 0.0) / float(pl["clearance_decay"])
    )
    pairs, paths, diffs = [], [], []
    m = len(spawns_rc)
    cand = [(i, j) for i in range(m) for j in range(m) if i != j]
    rng.shuffle(cand)
    for i, j in cand:
        if len(pairs) >= int(pl["pairs_per_map"]):
            break
        d = float(np.linalg.norm(md.spawns[i] - md.spawns[j]))
        if not (pl["min_distance"] <= d <= pl["max_distance"]):
            continue
        cells = astar(free, tuple(spawns_rc[i]), tuple(spawns_rc[j]), cell_cost)
        if cells is None:
            continue
        raw = md.cell_to_xy(np.asarray(cells))
        raw[0], raw[-1] = md.spawns[i], md.spawns[j]
        pts = finalize_path(raw, free, md.size, md.resolution, float(pl["waypoint_spacing"]))
        if pts is None:
            continue
        # A disk of the footprint's lateral half-width must clear the true primitives along the path; reject
        # the pair (and try another) when the robot's width cannot fit through the planned corridor.
        if footprint is not None and not path_width_clear(pts, footprint, prims):
            continue
        length = path_length(pts)
        clr = clearance[tuple(md.xy_to_cell(pts).T)]
        tortuosity = length / max(d, 1e-6)
        narrowness = float(np.mean(r_free / np.maximum(clr, 1e-3)))
        diffs.append(length / size + 0.5 * (tortuosity - 1.0) + narrowness + 2.0 * md.density)
        pairs.append((i, j))
        paths.append(pts)
    if len(pairs) < int(pl["min_pairs"]):
        return None
    md.pairs = np.asarray(pairs, np.int32)
    md.set_paths(paths)
    md.difficulty = np.asarray(diffs, np.float32)
    return md


def _worker(args: tuple[dict, int]) -> MapData | None:
    cfg, seed = args
    return generate_map(cfg, seed)


def save_preview(md: MapData, path: Path, pair: int = 0, scale: int = 3) -> None:
    from PIL import Image, ImageDraw

    img = np.where(md.occupancy, 30, 235).astype(np.uint8)[::-1]  # flip so +y is up
    im = Image.fromarray(img).convert("RGB").resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    n = md.occupancy.shape[0]

    def px(p):
        c = (p[0] + md.size / 2) / md.resolution * scale
        r = (n - (p[1] + md.size / 2) / md.resolution) * scale
        return (float(c), float(r))

    for s in md.spawns:
        x, y = px(s)
        dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(80, 80, 255))
    for k in range(min(md.num_pairs, 3)):
        p = md.path(k)
        dr.line([px(q) for q in p], fill=[(255, 0, 0), (0, 160, 0), (255, 140, 0)][k], width=2)
    im.save(path)


def generate_dataset(
    cfg: dict, out_dir: str | Path, workers: int = 0, log: Callable[[str], None] = print
) -> dict[str, Any]:
    """Generate train/val/test splits into ``out_dir/<split>/map_XXXXX.npz`` plus ``index.json``."""
    out = prepare_dataset_directory(out_dir)
    workers = workers or max(1, mp.cpu_count() - 1)
    summary: dict[str, Any] = {"config": cfg, "splits": {}}
    base_seed = int(cfg["seed"])
    for s_idx, (split, count) in enumerate(cfg["splits"].items()):
        count = int(count)
        sdir = out / split
        sdir.mkdir(exist_ok=True)
        maps: list[str] = []
        gens: dict[str, int] = {}
        seed = base_seed + s_idx * 10_000_000
        attempts = 0
        started = time.monotonic()
        log(f"[datagen] {split}: starting {count} maps with {workers} worker(s)")
        with mp.get_context("spawn").Pool(workers) as pool:
            while len(maps) < count:
                batch = [(cfg, seed + k) for k in range(max(workers * 2, count - len(maps)))]
                seed += len(batch)
                results = pool.imap(_worker, batch)
                remaining = len(batch)
                while remaining:
                    try:
                        md = results.next(timeout=10)
                    except mp.TimeoutError:
                        log(f"[datagen] {split}: {len(maps)}/{count} maps, planning paths "
                            f"({time.monotonic() - started:.0f}s elapsed)")
                        continue
                    remaining -= 1
                    attempts += 1
                    if md is None or len(maps) >= count:
                        continue
                    fname = f"map_{len(maps):05d}.npz"
                    md.save(sdir / fname)
                    if len(maps) < int(cfg.get("num_previews", 0)):
                        (out / "previews").mkdir(exist_ok=True)
                        save_preview(md, out / "previews" / f"{split}_{len(maps):05d}_{md.generator}.png")
                    maps.append(fname)
                    gens[md.generator] = gens.get(md.generator, 0) + 1
                    log(f"[datagen] {split}: {len(maps)}/{count} maps "
                        f"({attempts} attempts, {time.monotonic() - started:.0f}s elapsed)")
                log(f"[datagen] {split}: {len(maps)}/{count} maps (attempts {attempts})")
                if attempts > 50 * count + 100:
                    raise RuntimeError(f"too many rejected maps for split {split}; check datagen config")
        summary["splits"][split] = {"maps": maps, "generators": gens}
    (out / "index.json").write_text(json.dumps(summary, indent=2))
    return summary
