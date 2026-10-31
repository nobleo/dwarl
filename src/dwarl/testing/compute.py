"""Wall-clock profiling; CPU percentages use one logical core = 100%."""

import os
import platform
import time
from pathlib import Path

import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {"count": len(values), "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "max": float(values.max())}


def hardware():
    return {"platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
            "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name")), platform.processor()),
            "thread_environment": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


class ComputeProfile:
    def __init__(self, root_pid, period_s):
        self.root_pid, self.period_s = root_pid, period_s
        self.started = time.monotonic()
        self.previous = {}
        self.samples, self.timings = [], []
        self.observations = []
        self.last_sample = 0.

    def timing(self, message, simulation_time):
        for status in message.status:
            values = {v.key: v.value for v in status.values}
            if "state" in values and "lidar" in values:
                self.observations.append({k: np.fromstring(values[k].rstrip(","), sep=",", dtype=np.float32)
                                          for k in ("state", "lidar")})
            row = {"wall_time_s": time.monotonic() - self.started, "simulation_time_s": simulation_time}
            for key in ("controller_ms", "inference_ms", "observation_ms",
                        "cycle_ms_excluding_metrics_publish", "failed", "reverse_limit_intervention"):
                if key in values:
                    if key == "inference_ms" and getattr(status, "message", "") not in (
                        "running", "nonfinite_policy_output"
                    ):
                        continue
                    row[key] = float(values[key])
            self.timings.append(row)

    def sample(self):
        now = time.monotonic()
        if now - self.last_sample < .5:
            return
        self.last_sample = now
        processes = {}
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
                pid = int(path.parent.name)
                command = (path.parent / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                processes[pid] = (int(fields[1]), fields[19], (int(fields[11]) + int(fields[12])) /
                                  os.sysconf("SC_CLK_TCK"), int(fields[21]) * os.sysconf("SC_PAGE_SIZE"), command)
            except (OSError, ValueError, IndexError):
                continue
        owned = {self.root_pid}
        while True:
            children = {pid for pid, p in processes.items() if p[0] in owned}
            if children <= owned:
                break
            owned |= children
        for pid in owned & processes.keys():
            _, start, cpu, rss, command = processes[pid]
            key = (pid, start)
            previous = self.previous.get(key)
            self.previous[key] = (now, cpu)
            group = "controller_server" if "nav2_controller/controller_server" in command else (
                "gazebo" if "gz sim" in command or "gz-sim" in command else "other_stack")
            self.samples.append({"wall_time_s": now - self.started, "pid": pid, "group": group,
                                 "rss_bytes": rss, "cpu_percent": (100 * (cpu - previous[1]) /
                                 (now - previous[0])) if previous else None})

    def result(self, simulation_elapsed):
        elapsed = time.monotonic() - self.started
        calls = [r["controller_ms"] for r in self.timings if "controller_ms" in r]
        groups = {}
        for group in ("controller_server", "gazebo", "other_stack"):
            batches = {}
            for row in self.samples:
                if row["group"] == group:
                    batch = batches.setdefault(row["wall_time_s"], [[], 0])
                    if row["cpu_percent"] is not None:
                        batch[0].append(row["cpu_percent"])
                    batch[1] += row["rss_bytes"]
            groups[group] = {"cpu_percent": distribution([sum(v[0]) for v in batches.values() if v[0]]),
                             "rss_bytes": distribution([v[1] for v in batches.values()])}
        return {"schema": "dwarl_compute/1", "hardware": hardware(), "wall_elapsed_s": elapsed,
                "simulation_elapsed_s": simulation_elapsed, "real_time_factor": simulation_elapsed / elapsed,
                "controller_ms": distribution(calls),
                "inference_ms": distribution([r["inference_ms"] for r in self.timings if "inference_ms" in r]),
                "observation_ms": distribution([r["observation_ms"] for r in self.timings if "observation_ms" in r]),
                "control_budget_ms": self.period_s * 1000,
                "calls_over_budget": sum(v > self.period_s * 1000 for v in calls) if calls else None,
                "failed_calls": sum(r.get("failed", 0) for r in self.timings) if calls else None,
                "reverse_limited_calls": sum(r["reverse_limit_intervention"] for r in self.timings
                                             if "reverse_limit_intervention" in r)
                if any("reverse_limit_intervention" in r for r in self.timings) else None,
                "processes": groups, "timings": self.timings, "process_samples": self.samples,
                "notes": ["CPU 100% = one logical core; RSS sums may double-count shared memory.",
                          "Controller timings exclude wrapper publication; include inner plugin diagnostics.",
                          "Timing statistics count received ROS samples; middleware may drop telemetry.",
                          "DWARL profiling records up to 128 inputs; serialization adds controller-call overhead.",
                          "Observation timing covers lidar/state construction, excluding TF and scan acquisition.",
                          "Over-budget calls are not a measurement of scheduler deadline misses.",
                          "Process sampling excludes startup and may miss short-lived processes."]}
