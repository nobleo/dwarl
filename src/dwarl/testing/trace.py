"""Small, explicit behavior traces shared by simulator backends and reports."""

import math
import re
from pathlib import Path

import numpy as np

from dwarl.runs.store import write_json

UNITS = {
    "time_s": "s since episode start (simulation time)",
    "x_m": "m", "y_m": "m", "yaw_rad": "rad",
    "command_v_m_s": "m/s", "command_w_rad_s": "rad/s",
    "measured_v_m_s": "m/s", "measured_w_rad_s": "rad/s",
    "clearance_m": "m (nonnegative footprint-to-obstacle distance; zero for contact/overlap)",
    "distance_to_goal_m": "m (Euclidean)", "path_remaining_m": "m",
    "path_completion": "fraction [0, 1]",
}


def finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def path_progress(reference, xy):
    """Geometric projection, independent of a backend's replanned global path."""
    path = np.asarray(reference, dtype=float)
    delta = np.diff(path, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if not len(lengths) or lengths.sum() <= 1e-12:
        return None, None
    fraction = np.clip(np.sum((np.asarray(xy) - path[:-1]) * delta, axis=1)
                       / np.maximum(lengths**2, 1e-12), 0, 1)
    projection = path[:-1] + fraction[:, None] * delta
    index = int(np.argmin(np.linalg.norm(projection - xy, axis=1)))
    along = float(lengths[:index].sum() + fraction[index] * lengths[index])
    total = float(lengths.sum())
    return max(0., total - along), min(1., along / total)


class BehaviorTrace:
    def __init__(self, case, *, pose_source, velocity_source, command_source, clearance_source):
        self.case = case
        self.data = {
            "schema": "dwarl_behavior_trace/1", "case_id": case["id"], "units": UNITS,
            "sources": {"pose": pose_source, "measured_velocity": velocity_source,
                        "command": command_source, "clearance": clearance_source},
            "path_progress_method": "nearest projection on canonical reference; may jump at path crossings",
            "sample_convention": ("pose/velocity/clearance at time_s; "
                                  "command applied for that step or latest fresh command"),
            "samples": [], "events": [],
        }

    def sample(self, time_s, pose, *, command=None, velocity=None, clearance=None):
        remaining, completion = path_progress(self.case["reference_path"], pose[:2])
        command = command if command is not None else (None, None)
        velocity = velocity if velocity is not None else (None, None)
        row = dict(zip(("time_s", "x_m", "y_m", "yaw_rad", "command_v_m_s", "command_w_rad_s",
                        "measured_v_m_s", "measured_w_rad_s", "clearance_m"),
                       map(finite, (time_s, *pose, *command, *velocity, clearance)), strict=True))
        row.update(distance_to_goal_m=finite(math.dist(pose[:2], self.case["goal"])),
                   path_remaining_m=remaining, path_completion=completion)
        self.data["samples"].append(row)
        return row

    def event(self, kind, time_s, pose=None, **details):
        self.data["events"].append({"type": kind, "time_s": finite(time_s),
                                    "pose": list(map(finite, pose)) if pose is not None else None,
                                    "details": details})


def persist_episode(directory, simulator, case, row, checkpoint):
    """Persist traces before plotting so report failures never discard measurements."""
    trace = row.pop("_trace", None)
    label = re.sub(r"[^a-zA-Z0-9_-]", "_", case["name"])[:80]
    target = Path("episodes") / f"{label}-{case['id'][:16]}" / simulator
    row.update(schema="dwarl_episode/1", simulator=simulator, checkpoint=checkpoint,
               case_name=case["name"], repeat=case["repeat"], map_file=case["map_file"],
               start=case["start"], goal=case["goal"], reference_path=case["reference_path"],
               episode_file=str(target / "episode.json"), trace_file=None)
    if row["status"] != "completed":
        row["outcome"] = row["status"]
        row["termination_reason"] = row.get("error") or row["status"]
    for name in ("success", "collision", "timeout", "stuck", "travelled_distance", "completion_time",
                 "clearance", "safety_stops", "recovery_actions", "final_distance_to_goal",
                 "path_completion", "final_pose", "termination_reason"):
        row.setdefault(name, None)
    for name in ("travelled_distance", "completion_time", "clearance"):
        row[name] = finite(row[name])
    if trace is not None:
        samples = trace["samples"]
        final = samples[-1] if samples else None
        pose = [final[k] for k in ("x_m", "y_m", "yaw_rad")] if final else None
        if final:
            row.update(final_pose=pose, final_distance_to_goal=final["distance_to_goal_m"],
                       path_completion=final["path_completion"])
        clearances = [s["clearance_m"] for s in samples if s["clearance_m"] is not None]
        if row["clearance"] is not None:
            clearances.append(row["clearance"])
        if clearances:
            row["clearance"] = min(clearances)
        events = trace["events"]
        if row.get("collision") and not any(e["type"] == "collision" for e in events):
            events.append({"type": "collision", "time_s": final["time_s"] if final else None,
                           "pose": pose, "details": {"location": "robot pose at detection, not a contact point"}})
        events.append({"type": "termination",
                       "time_s": final["time_s"] if final and row["status"] == "completed" else None,
                       "pose": pose, "details": {"outcome": row.get("outcome"), "reason": row["termination_reason"],
                                                 "last_observed_time_s": final["time_s"] if final else None}})
        trace.update(simulator=simulator, checkpoint=checkpoint, outcome=row.get("outcome"))
        row["trace_file"] = str(target / "trace.json")
        write_json(directory / row["trace_file"], trace)
    row["units"] = {"completion_time": "s", "travelled_distance": "m", "clearance": "m",
                    "final_distance_to_goal": "m", "path_completion": "fraction [0, 1]",
                    "final_pose": ["m", "m", "rad"]}
    write_json(directory / row["episode_file"], row)
    return row
