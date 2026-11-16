"""Readable episode reports: one trajectory and one motion timeline, never a collage."""

import json
import math
from pathlib import Path

import numpy as np

EVENT_COLORS = {"collision": "#c43c39", "safety_stop": "#b76b00", "safety_resume": "#38856c",
                "recovery": "#85358a", "termination": "#52525b"}


def number(value):
    return "unavailable" if value is None else f"{value:.2f}"


def caption(row):
    return (f"{row['outcome']} · {number(row.get('completion_time'))} s · "
            f"{number(row.get('travelled_distance'))} m travelled · "
            f"min clearance {number(row.get('clearance'))} m")


def trajectory_plot(plt, md, case, row, trace, destination):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Circle, Polygon

    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    unmapped = row.get("unmapped_obstacles") or []
    shapes = [(p, "#c9c9ce", None) for p in md.primitives]
    shapes += [(p, "#e0a458", "Not on the map") for p in unmapped]
    for (kind, x, y, angle, hx, hy), color, label in shapes:
        if int(kind) == 1:
            patch = Circle((x, y), hx, color=color)
        else:
            c, s = math.cos(angle), math.sin(angle)
            corners = [(x+c*u-s*v, y+s*u+c*v) for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
            patch = Polygon(corners, color=color)
        if label and label not in [h.get_label() for h in ax.patches]:
            patch.set_label(label)
        ax.add_patch(patch)
    ax.plot(*zip(*case["reference_path"], strict=True), "--", color="#85858f", lw=1.4, label="Reference path")
    samples = trace.get("samples", [])
    points = np.array([[s["x_m"], s["y_m"]] for s in samples]) if samples else np.array(row.get("trajectory") or [])
    if len(points):
        ax.plot(points[:, 0], points[:, 1], color="#85358a", lw=2, label="Actual route", zorder=3)
        speed = np.array([s["measured_v_m_s"] if s["measured_v_m_s"] is not None else np.nan for s in samples])
        if len(points) > 1 and len(speed) == len(points) and np.isfinite(speed[1:]).any():
            segments = np.stack([points[:-1], points[1:]], axis=1)
            norm = Normalize(0, max(.05, float(np.nanmax(np.abs(speed)))))
            colored = LineCollection(segments, cmap="viridis", norm=norm, linewidth=2.8, zorder=4)
            colored.set_array(np.ma.masked_invalid(np.abs(speed[1:])))
            ax.add_collection(colored)
            fig.colorbar(colored, ax=ax, shrink=.65, label="Measured forward speed magnitude |v| (m/s)")
        ax.scatter(*points[-1], marker="s", s=45, color="#24232a", zorder=6, label="Final observed position")
        pose = row.get("final_pose")
        if pose and pose[2] is not None:
            length = min(.5, md.size * .04)
            ax.annotate("", xy=(pose[0]+length*math.cos(pose[2]), pose[1]+length*math.sin(pose[2])),
                        xytext=pose[:2], arrowprops={"arrowstyle": "->", "color": "#24232a", "lw": 1.8})
    used = set()
    for event in trace.get("events", []):
        kind, pose = event["type"], event["pose"]
        if kind == "termination" or pose is None:
            continue
        ax.scatter(*pose[:2], marker="x", s=65, linewidths=2, color=EVENT_COLORS.get(kind, "#52525b"),
                   zorder=7, label=kind.replace("_", " ") if kind not in used else None)
        used.add(kind)
    ax.scatter(*case["start"][:2], marker="o", color="#24855e", s=65, zorder=6, label="Start")
    ax.scatter(*case["goal"], marker="*", color="#c43c39", s=140, zorder=6, label="Goal")
    ax.set(xlim=(-md.size/2, md.size/2), ylim=(-md.size/2, md.size/2), aspect="equal",
           xlabel="x (m)", ylabel="y (m)", title=f"{case['name']} · {row['simulator']}\n{caption(row)}")
    ax.grid(alpha=.15)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def motion_plot(plt, row, trace, destination):
    samples = trace["samples"]
    times = [s["time_s"] for s in samples]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout="constrained")
    channels = [("Linear speed (m/s)", [("command_v_m_s", "Command", "#85358a", "--"),
                                       ("measured_v_m_s", "Measured", "#227ba1", "-")]),
                ("Turn rate (rad/s)", [("command_w_rad_s", "Command", "#85358a", "--"),
                                       ("measured_w_rad_s", "Measured", "#227ba1", "-")]),
                ("Clearance (m)", [("clearance_m", "Footprint clearance", "#24855e", "-")])]
    for ax, (label, fields) in zip(axes, channels, strict=True):
        missing = []
        for key, name, color, style in fields:
            values = [s[key] if s[key] is not None else np.nan for s in samples]
            if np.isfinite(values).any():
                ax.plot(times, values, style, color=color, label=name, lw=1.5)
            else:
                missing.append(name)
        ax.set_ylabel(label)
        ax.axhline(0, color="#aaa", lw=.6)
        ax.grid(alpha=.18)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)
        if missing:
            ax.text(.01, .04, f"Unavailable: {', '.join(missing)}", transform=ax.transAxes, fontsize=8)
    used = set()
    for event in trace["events"]:
        kind, time_s = event["type"], event["time_s"]
        if time_s is None:
            continue
        for ax in axes:
            ax.axvline(time_s, color=EVENT_COLORS.get(kind, "#52525b"), lw=1, alpha=.6, linestyle=":")
        if kind not in used:
            axes[0].text(time_s, 1.02, kind.replace("_", " "), transform=axes[0].get_xaxis_transform(),
                         fontsize=7, rotation=30, color=EVENT_COLORS.get(kind, "#52525b"))
            used.add(kind)
    axes[-1].set_xlabel("Elapsed simulation time (s)")
    fig.suptitle(f"{row['case_name']} · {row['simulator']} · {row['outcome']}", fontsize=12)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def compute_plot(plt, compute, destination):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    for key, label in (("controller_ms", "Controller call"), ("inference_ms", "ONNX inference"),
                       ("observation_ms", "Observation building")):
        points = [(r["wall_time_s"], r[key]) for r in compute["timings"] if key in r]
        if points:
            axes[0].plot(*zip(*points, strict=True), label=label, lw=1)
    axes[0].axhline(compute["control_budget_ms"], color="red", linestyle="--", label="Control budget")
    axes[0].set_ylabel("Wall-clock duration (ms)")
    for group in ("controller_server", "gazebo", "other_stack"):
        samples = {}
        for row in compute["process_samples"]:
            if row["group"] == group and row["cpu_percent"] is not None:
                t = row["wall_time_s"]
                samples[t] = samples.get(t, 0.) + row["cpu_percent"]
        if samples:
            axes[1].plot(list(samples), list(samples.values()), label=group)
    axes[1].set(xlabel="Elapsed wall-clock time (s)", ylabel="CPU (% of one core)")
    for ax in axes:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


LOCALIZATION_WARNING_M = 0.1


def write_report(directory, workload, rows, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dwarl.datasets.generation.mapdata import MapData

    report = directory / "report"
    report.mkdir(exist_ok=True)
    comparison = ["## Controller comparison", "",
                  "Time and distance averages below include successful episodes only.", "",
                  "| Backend/controller | Completed | Success | Collision | Success time (s) | Success distance (m) |",
                  "|---|---|---|---|---|---|"]
    for name, stats in summary.get("simulators", {}).items():
        comparison.append(f"| {name} | {stats['completed']}/{stats['expected']} | "
                          f"{number(stats.get('success_rate'))} | {number(stats.get('collision_rate'))} | "
                          f"{number(stats.get('successful_mean_time_s'))} | "
                          f"{number(stats.get('successful_mean_distance_m'))} |")
    comparison.append("")
    for name, stats in summary.get("simulators", {}).items():
        drift = stats.get("mean_localization_error_m")
        if drift is not None and drift > LOCALIZATION_WARNING_M:
            comparison.extend([f"**Warning:** {name} ends {drift:.2f} m away from ground truth on average "
                               "(Nav2 pose vs simulator). The bench localizes with odometry, so these results "
                               "include odometry drift, not only control. See `final_pose_comparison` in the "
                               "episode diagnostics.", ""])
    lines = ["# Navigation test", "", f"Checkpoint: `{summary['checkpoint']}`", "",
             f"Workload: `{summary['workload']}`", "", f"Status: **{summary['status']}**", "",
             "Each episode has its own images and JSON files. Missing measurements are null, not zero.",
             "Path completion projects onto the canonical reference; it can jump at crossings. "
             "Nav2 can follow a different global path.",
             "Collision markers show the robot pose at detection, not an inferred contact point.", "",
             "[Summary JSON](../summary.json) · [All episodes JSON](../episodes.json)", "",
             "| Case / repeat | Simulator | Outcome | Time (s) | Distance (m) | Min clearance (m) | Goal gap (m) |",
             "|---|---|---|---|---|---|---|"]
    sections = []
    for case in workload["cases"]:
        for simulator, episodes in rows.items():
            row = next((r for r in episodes if r["case_id"] == case["id"]), None)
            if row is None:
                continue
            lines.append(f"| {case['name']} / {case['repeat']} | {simulator} | {row['outcome']} | "
                         f"{number(row.get('completion_time'))} | {number(row.get('travelled_distance'))} | "
                         f"{number(row.get('clearance'))} | {number(row.get('final_distance_to_goal'))} |")
            target = Path(row["episode_file"]).parent
            sections.extend(["", f"## {case['name']} · {simulator} · {case['id'][:8]}", "", caption(row), "",
                             f"[Episode JSON](../{row['episode_file']})", ""])
            if row.get("compute_file"):
                compute = row["compute"]
                sections.extend([f"[Compute measurements JSON](../{row['compute_file']})", "",
                                 "| Wall-clock metric | p50 | p95 | p99 | Max |",
                                 "|---|---|---|---|---|"])
                for key, label in (("controller_ms", "Controller call (ms)"),
                                   ("inference_ms", "ONNX inference (ms)"),
                                   ("observation_ms", "Observation building (ms)")):
                    stats = compute.get(key) or {}
                    sections.append(f"| {label} | " + " | ".join(number(stats.get(k))
                                    for k in ("p50", "p95", "p99", "max")) + " |")
                sections.extend(["", f"Calls exceeding control budget: {compute['calls_over_budget']}; "
                                 f"real-time factor: {number(compute['real_time_factor'])}.", "",
                                 "CPU usage (100% = one logical core) and RSS are in the compute JSON; "
                                 "Gazebo is measured separately from controller_server.", ""])
                raw_compute = json.loads((directory / row["compute_file"]).read_text())
                compute_plot(plt, raw_compute, directory / target / "compute.png")
                sections.extend([f"![{case['name']} {simulator} compute](../{target}/compute.png)", ""])
            trace = {"samples": [], "events": []}
            if row.get("trace_file"):
                trace = json.loads((directory / row["trace_file"]).read_text())
                sections.extend([f"[Timestamped trace JSON](../{row['trace_file']})", "",
                                 f"Command: {trace['sources']['command']}.", "",
                                 f"Measured velocity: {trace['sources']['measured_velocity']}.", "",
                                 f"Clearance: {trace['sources']['clearance']}.", ""])
            if row.get("error"):
                sections.extend([f"Infrastructure error: `{row['error']}`", ""])
            if row.get("trajectory") or trace["samples"]:
                md = MapData.load(directory / case["map_file"])
                trajectory_plot(plt, md, case, row, trace, directory / target / "trajectory.png")
                sections.extend([f"![{case['name']} {simulator} trajectory](../{target}/trajectory.png)", ""])
            else:
                sections.extend(["No trajectory measurements available.", ""])
            if trace["samples"]:
                motion_plot(plt, row, trace, directory / target / "motion.png")
                sections.extend([f"![{case['name']} {simulator} motion timeline](../{target}/motion.png)", ""])
    (report / "README.md").write_text("\n".join(lines[:2] + comparison + lines[2:] + sections) + "\n")
