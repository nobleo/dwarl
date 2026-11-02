"""Map dataset loading and curriculum-aware (map, pair) sampling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dwarl.datasets.generation.mapdata import MapData


class MapDataset:
    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root)
        index = json.loads((self.root / "index.json").read_text())
        self.index = index
        if split not in index["splits"]:
            raise KeyError(f"split '{split}' not in dataset {root} (have {list(index['splits'])})")
        self.split = split
        if index.get("schema") == "dwarl_encounters/1":
            raise ValueError("encounter datasets require the removed experimental dynamics; use procedural maps")
        for file in index["splits"][split]["maps"]:
            expected = index.get("files", {}).get(f"{split}/{file}")
            if expected and hashlib.sha256((self.root / split / file).read_bytes()).hexdigest() != expected:
                raise ValueError(f"dataset file hash mismatch: {split}/{file}")
        self.maps = [MapData.load(self.root / split / f) for f in index["splits"][split]["maps"]]
        if not self.maps:
            raise ValueError(f"dataset {root}/{split} is empty")
        # flat pair table: (map_idx, pair_idx)
        self.items = np.array([(mi, k) for mi, md in enumerate(self.maps) for k in range(md.num_pairs)], np.int64)
        diff = np.concatenate([md.difficulty for md in self.maps])
        order = np.argsort(np.argsort(diff, kind="stable"), kind="stable")
        self.rank = order / max(len(diff) - 1, 1)  # difficulty quantile in [0, 1]
        self.map_item_slices = np.concatenate([[0], np.cumsum([md.num_pairs for md in self.maps])])
        self.max_path_points = max(int(np.diff(md.path_offsets).max()) for md in self.maps)
        self.max_primitives = max(len(md.primitives) for md in self.maps)


@dataclass
class CurriculumState:
    level: float
    window: list[float]
    last_window_success: float | None = None


class PairSampler:
    """Samples (map, pair) items. With curriculum enabled only items whose difficulty quantile is
    <= level are drawn; the level moves up/down with the rolling success rate."""

    def __init__(
        self,
        dataset: MapDataset,
        enabled: bool = True,
        init_level: float = 0.2,
        min_level: float = 0.1,
        step: float = 0.05,
        window: int = 200,
        promote_success: float = 0.75,
        demote_success: float = 0.3,
        sequential: bool = False,
        seed: int = 0,
    ) -> None:
        self.ds = dataset
        self.enabled = enabled and not sequential
        self.level = init_level if self.enabled else 1.0
        self.min_level, self.step = min_level, step
        self.window_size = window
        self.promote, self.demote = promote_success, demote_success
        self.sequential = sequential
        self._cursor = 0
        self._outcomes: list[float] = []
        self.last_window_success: float | None = None  # diagnostics: rate that caused the latest level decision
        self.rng = np.random.default_rng(seed)
        # fixed shuffled order for sequential (evaluation) sampling, so partial evals are unbiased
        self._order = np.random.default_rng(seed).permutation(len(self.ds.items))

    @property
    def exhausted(self) -> bool:
        return self.sequential and self._cursor >= len(self.ds.items)

    @property
    def evaluation_order(self) -> tuple[tuple[int, int], ...]:
        """One complete seeded pass, exposed as immutable (map, pair) identifiers."""
        return tuple((int(self.ds.items[i, 0]), int(self.ds.items[i, 1])) for i in self._order)

    @property
    def window_episodes(self) -> int:
        return len(self._outcomes)

    def curriculum_state(self) -> CurriculumState:
        return CurriculumState(self.level, list(self._outcomes), self.last_window_success)

    def restore_curriculum(self, state: CurriculumState) -> None:
        self.level = state.level
        self._outcomes = list(state.window)
        self.last_window_success = state.last_window_success

    def sample(self, map_idx: int | None = None) -> tuple[int, int]:
        if self.sequential:
            i = self._order[self._cursor % len(self.ds.items)]
            self._cursor += 1
            mi, k = self.ds.items[i]
            return int(mi), int(k)
        if map_idx is not None:
            lo, hi = self.ds.map_item_slices[map_idx], self.ds.map_item_slices[map_idx + 1]
            ranks = self.ds.rank[lo:hi]
            ok = np.flatnonzero(ranks <= self.level)
            j = lo + (self.rng.choice(ok) if len(ok) else int(np.argmin(ranks)))
        else:
            ok = np.flatnonzero(self.ds.rank <= self.level)
            j = self.rng.choice(ok) if len(ok) else int(np.argmin(self.ds.rank))
        mi, k = self.ds.items[j]
        return int(mi), int(k)

    def report(self, success: np.ndarray) -> None:
        if not self.enabled:
            return
        self._outcomes.extend(success.astype(float).tolist())
        if len(self._outcomes) >= self.window_size:
            rate = float(np.mean(self._outcomes))
            self.last_window_success = rate
            self._outcomes.clear()
            if rate >= self.promote:
                self.level = min(1.0, self.level + self.step)
            elif rate <= self.demote:
                self.level = max(self.min_level, self.level - self.step)
