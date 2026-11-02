"""ROS/Torch-free synthetic full-cycle benchmark; run on the intended target CPU.

python -m dwarl_runtime.benchmark /path/to/bundle --iters 1000 --out runtime_bench.json
Includes preprocessing, tracking, history, ONNX and command mapping. Excludes
sensor transport, TF, middleware and actuator latency. Never sends robot commands.
"""

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np

from .core import Policy, Scan


def benchmark(bundle, iters=1000, warmup=100):
    if iters < 1 or warmup < 0:
        raise ValueError("positive iters and nonnegative warmup required")
    policy = Policy(bundle, threads=1)
    s = policy.spec
    policy.set_plan([[0., 0.], [10., 0.]])
    scan = Scan(np.full(s.num_rays, s.max_range), s.angle_min, s.angle_increment, s.min_range, s.max_range)
    samples = []
    cpu_start = time.process_time()
    for i in range(warmup + iters):
        t0 = time.perf_counter_ns()
        policy.step(scan, (*s.origin, 0.), (0., 0., 0.), (0., 0.))
        elapsed = (time.perf_counter_ns() - t0) * 1e-6
        if i >= warmup:
            samples.append(elapsed)
    # Linux ru_maxrss may include a launcher before exec. /proc describes this
    # executable's current address space; retain both for transparent reporting.
    memory = {}
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, value, _unit = line.split()
                memory[key[:-1]] = int(value)
    return {"scope": "synthetic_Policy.step_no_ROS_no_sensor_IO", "bundle": str(Path(bundle).resolve()),
            "platform": platform.platform(), "python": platform.python_version(), "onnx_threads": 1,
            "iters": iters, "warmup": warmup, "mean_ms": float(np.mean(samples)),
            "p50_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
            "p99_ms": float(np.percentile(samples, 99)), "max_ms": float(np.max(samples)),
            "cpu_seconds_including_warmup": time.process_time() - cpu_start,
            "address_space_rss_kib_linux": memory.get('VmRSS'),
            "address_space_peak_rss_kib_linux": memory.get('VmHWM'),
            "rusage_peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--out")
    args = parser.parse_args()
    text = json.dumps(benchmark(args.bundle, args.iters, args.warmup), indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
