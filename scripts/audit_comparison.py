#!/usr/bin/env python3
"""Audit persisted shared-case navigation evidence before selecting or publishing results."""

import argparse
import json
from pathlib import Path

import yaml


def audit(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    cases = json.loads((directory / "cases.json").read_text())["cases"]
    episodes = json.loads((directory / "episodes.json").read_text())
    expected = [c["id"] for c in cases]
    errors, controllers, shared = [], {}, {}
    if summary["status"] != "completed":
        errors.append(f"Study status is {summary['status']}")
    for label, rows in episodes.items():
        if [r["case_id"] for r in rows] != expected:
            errors.append(f"{label}: selected case IDs/order do not match")
        completed = [r for r in rows if r["status"] == "completed"]
        if len(completed) != len(cases):
            errors.append(f"{label}: incomplete/infrastructure-failed episodes")
        successes = sum(r.get("success", False) for r in completed)
        stats = summary["simulators"][label]
        if stats["completed"] != len(completed):
            errors.append(f"{label}: completed count disagrees with episode records")
        rate = successes / len(completed) if completed else None
        if stats["success_rate"] != rate:
            errors.append(f"{label}: success rate disagrees with episode records")
        controllers[label] = {
            "completed": len(completed), "successes": successes,
            "collisions": sum(r.get("collision", False) for r in completed),
            "nav2_succeeded_outside_tolerance": sum(
                r.get("termination_reason") == "nav2_succeeded_outside_goal_tolerance" for r in completed),
        }
        for row in rows:
            for key in ("episode_file", "trace_file", "compute_file"):
                if row.get(key) and not (directory / row[key]).is_file():
                    errors.append(f"{label}/{row['case_id']}: missing {key}")
            if row.get("backend") != "gazebo":
                continue
            folder = directory / "gazebo" / row["controller"] / row["case_id"]
            params_path = folder / "nav2_params.yaml"
            if not params_path.is_file():
                errors.append(f"{label}/{row['case_id']}: missing Nav2 parameter snapshot")
                continue
            params = yaml.safe_load(params_path.read_text())
            params["controller_server"]["ros__parameters"].pop("FollowPath", None)
            reference = shared.setdefault(row["case_id"], params)
            if params != reference:
                errors.append(f"{label}/{row['case_id']}: shared Nav2 parameters differ between controllers")
            comparison = row.get("diagnostics", {}).get("final_pose_comparison")
            if comparison and row.get("final_pose"):
                if row["final_pose"] != comparison["ground_truth_pose"]:
                    errors.append(f"{label}/{row['case_id']}: terminal trace and scoring pose disagree")
    return {"directory": str(directory.resolve()), "checkpoint": summary["checkpoint"],
            "workload": summary["workload"], "cases_per_controller": len(cases),
            "controllers": controllers, "errors": errors, "evidence_consistent": not errors,
            "note": "Consistency audit, not a navigation quality or statistical significance test."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["evidence_consistent"] else 1)
