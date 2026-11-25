"""Freeze one obstruction layout per clean case, shared by every controller."""

from copy import deepcopy

from omegaconf import OmegaConf

from dwarl.runs.store import identity, write_json


def freeze_obstacles(contract, workload, directory, rows, count, doorways, checkpoint):
    from dwarl.datasets.generation.mapdata import MapData
    from dwarl.testing.unmapped import place_unmapped

    expected = {case["id"] for case in workload["cases"]}
    for name, episodes in rows.items():
        if {row["case_id"] for row in episodes if row["status"] == "completed"} != expected:
            raise ValueError(f"clean pass incomplete for {name}; obstructed pass was not started")
    routes = {}
    for name, episodes in rows.items():
        for row in episodes:
            plans = (row.get("diagnostics") or {}).get("global_plans") or []
            if plans and len(plans[0]) > 1 and not row.get("unmapped_obstacles"):
                routes.setdefault(row["case_id"], (name, [p[:2] for p in plans[0]]))
    missing = expected - routes.keys()
    if missing:
        raise ValueError(f"clean pass has no Nav2 global plan for {len(missing)} case(s); "
                         "obstructed pass was not started")
    robot = OmegaConf.to_container(contract.robot, resolve=True)
    obstructed, layouts = deepcopy(workload), {}
    for case in obstructed["cases"]:
        source, path = routes[case["id"]]
        md = MapData.load(directory / case["map_file"])
        obstacles = place_unmapped(md, dict(case, reference_path=path), count, robot, doorways=doorways)
        placement = {"route": "nav2_first_plan", "source_variant": source, "source": "clean",
                     "route_points": path}
        case.update(unmapped_obstacles=obstacles.tolist(), unmapped_placement=placement)
        layouts[case["id"]] = {"obstacles": obstacles.tolist(), "placement": placement}
    manifest = {"schema": "dwarl_obstacle_comparison/1", "checkpoint": checkpoint,
                "base_workload": workload["identity"], "requested_per_case": count,
                "cases": layouts, "identity": identity(layouts)}
    obstructed["base_identity"] = workload["identity"]
    obstructed["identity"] = identity(obstructed["cases"])
    write_json(directory / "obstacles.json", manifest)
    write_json(directory / "obstructed_cases.json", obstructed)
    return obstructed, manifest


def paired_results(rows, controllers):
    """Completion-aware differences; time differences only use cases successful in BOTH passes."""
    result = {}
    for controller in controllers:
        base = "gazebo" if controller == "dwarl" else f"gazebo-{controller}"
        clean = {r["case_id"]: r for r in rows.get(f"{base}-clean", [])}
        obstructed = {r["case_id"]: r for r in rows.get(f"{base}-obstructed", [])}
        pairs = [(a, obstructed[key]) for key, a in clean.items() if key in obstructed
                 and a["status"] == obstructed[key]["status"] == "completed"]
        both_success = [(a, b) for a, b in pairs if a.get("success") and b.get("success")]
        def mean_delta(key, selected):
            differences = [float(b[key]) - float(a[key]) for a, b in selected
                           if a.get(key) is not None and b.get(key) is not None]
            return sum(differences) / len(differences) if differences else None
        result[controller] = {
            "paired_completed": len(pairs), "paired_successful": len(both_success),
            "success_rate_delta": mean_delta("success", pairs),
            "collision_rate_delta": mean_delta("collision", pairs),
            "successful_time_delta_s": mean_delta("completion_time", both_success),
        }
    return result
