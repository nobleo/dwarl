"""Freeze geometry and starts once; simulators consume the same cases.json."""

import math
import shutil
import tempfile
from pathlib import Path

from dwarl.runs.store import identity, lock, sha256, write_json


def materialize(contract, request, out):
    from dwarl.datasets.generation.mapdata import path_length
    from dwarl.testing.scenarios import LAYOUT_VERSION, SPLIT, build_scenarios
    from dwarl.training.environment.dataset import MapDataset

    source = request.dataset or contract.dataset.path
    split = {"validation": "val", "test": "test", "benchmark": SPLIT}[request.suite]
    if request.suite == "benchmark" and not request.dataset:
        root = Path(".dwarl/scenarios").resolve()
        key = identity({"robot": contract.robot.profile_sha256, "layout": LAYOUT_VERSION})
        source = root / key
        with lock(root / f".{key}.lock"):
            if not source.exists():
                with tempfile.TemporaryDirectory(prefix=".scenarios-", dir=root) as tmp:
                    stage = Path(tmp) / "data"
                    build_scenarios(stage, robot=contract.robot.profile_file, previews=False)
                    stage.rename(source)
    dataset = MapDataset(source, split)
    names = sorted({m.generator for m in dataset.maps})
    if request.case and request.case not in names:
        raise ValueError(f"unknown case {request.case!r}; available: {', '.join(names)}")
    maps_dir = out / "maps" / split
    maps_dir.mkdir(parents=True)
    cases, files = [], []
    for mi, md in enumerate(dataset.maps):
        name = dataset.index["splits"][split]["maps"][mi]
        shutil.copyfile(Path(source) / split / name, maps_dir / name)
        files.append(name)
        if request.case and md.generator != request.case:
            continue
        for pi in range(md.num_pairs):
            path = md.path(pi)
            direction = path[min(5, len(path)-1)] - path[0]
            yaw = math.atan2(float(direction[1]), float(direction[0]))
            for repeat in range(request.repeats):
                params = contract.terminations.timeout.params
                budget = (params.min_steps + int(params.time_factor * path_length(path) /
                          contract.robot.v_max / contract.env.control_dt)) * contract.env.control_dt
                case = {"name": md.generator, "map": mi, "pair": pi, "map_file": f"maps/{split}/{name}",
                        "geometry": sha256(maps_dir / name), "start": [*path[0].tolist(), yaw],
                        "goal": path[-1].tolist(), "reference_path": path.tolist(), "repeat": repeat,
                        "seed": request.seed + repeat, "time_budget": budget,
                        "tolerance": contract.terminations.success.params.tolerance,
                        "robot": contract.robot.profile_sha256, "suite": request.suite}
                case["id"] = identity(case)
                cases.append(case)
    if request.max_cases is not None:
        cases = cases[:request.max_cases]
    if not cases:
        raise ValueError("selected workload contains no cases")
    index = {"config": dataset.index.get("config", {}), "splits": {split: {"maps": files}}}
    write_json(out / "maps/index.json", index)
    workload = {"schema": "dwarl_cases/1", "split": split, "cases": cases, "identity": identity(cases)}
    write_json(out / "cases.json", workload)
    return workload
