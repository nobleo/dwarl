"""Map container and (de)serialisation.

Conventions (tile-local frame, metres):
    * The tile is a square of side ``size`` centred on the origin.
    * Grid cell (row r, col c) has centre x = -size/2 + (c + 0.5) * res, y = -size/2 + (r + 0.5) * res.
    * ``primitives`` is an (N, 6) float32 array: [type, x, y, yaw, hx, hy] with type 0 = box
      (half extents hx, hy) and type 1 = cylinder (radius hx).
    * Paths are polylines in metres, stored flat with offsets; ``pairs[k] = (start_idx, goal_idx)``
      indexes into ``spawns`` and ``path(k)`` is the A* path for that pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BOX, CYLINDER = 0, 1


def prepare_dataset_directory(path: str | Path) -> Path:
    """Refuse to replace existing dataset files, including an unfinished build."""
    out = Path(path).expanduser()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"dataset destination must be empty: {out}; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class MapData:
    occupancy: np.ndarray  # (H, W) bool, True = occupied
    resolution: float
    size: float
    height: float
    primitives: np.ndarray  # (N, 6) float32
    generator: str = ""
    seed: int = 0
    spawns: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))
    path_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    path_offsets: np.ndarray = field(default_factory=lambda: np.zeros((1,), np.int64))
    difficulty: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))  # per pair
    density: float = 0.0

    # ------------------------------------------------------------------ geometry helpers
    def cell_to_xy(self, rc: np.ndarray) -> np.ndarray:
        rc = np.asarray(rc, dtype=np.float64)
        half, res = self.size / 2.0, self.resolution
        return np.stack([-half + (rc[..., 1] + 0.5) * res, -half + (rc[..., 0] + 0.5) * res], -1)

    def xy_to_cell(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        half = self.size / 2.0
        c = np.floor((xy[..., 0] + half) / self.resolution).astype(np.int64)
        r = np.floor((xy[..., 1] + half) / self.resolution).astype(np.int64)
        n = self.occupancy.shape[0]
        return np.stack([np.clip(r, 0, n - 1), np.clip(c, 0, n - 1)], -1)

    @property
    def num_pairs(self) -> int:
        return int(self.pairs.shape[0])

    def path(self, k: int) -> np.ndarray:
        return self.path_points[self.path_offsets[k] : self.path_offsets[k + 1]]

    def set_paths(self, paths: list[np.ndarray]) -> None:
        lengths = np.array([len(p) for p in paths], dtype=np.int64)
        self.path_offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        self.path_points = (np.concatenate(paths, 0) if paths else np.zeros((0, 2))).astype(np.float32)

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            occupancy=np.packbits(self.occupancy, axis=None),
            shape=np.array(self.occupancy.shape),
            resolution=self.resolution,
            size=self.size,
            height=self.height,
            primitives=self.primitives.astype(np.float32),
            generator=self.generator,
            seed=self.seed,
            spawns=self.spawns.astype(np.float32),
            pairs=self.pairs.astype(np.int32),
            path_points=self.path_points.astype(np.float32),
            path_offsets=self.path_offsets.astype(np.int64),
            difficulty=self.difficulty.astype(np.float32),
            density=self.density,
        )

    @classmethod
    def load(cls, path: str | Path) -> MapData:
        z = np.load(path, allow_pickle=False)
        shape = tuple(int(s) for s in z["shape"])
        occ = np.unpackbits(z["occupancy"], count=shape[0] * shape[1]).reshape(shape).astype(bool)
        return cls(
            occupancy=occ,
            resolution=float(z["resolution"]),
            size=float(z["size"]),
            height=float(z["height"]),
            primitives=z["primitives"],
            generator=str(z["generator"]),
            seed=int(z["seed"]),
            spawns=z["spawns"],
            pairs=z["pairs"],
            path_points=z["path_points"],
            path_offsets=z["path_offsets"],
            difficulty=z["difficulty"],
            density=float(z["density"]),
        )


def path_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
