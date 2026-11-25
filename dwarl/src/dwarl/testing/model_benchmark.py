"""Replay recorded policy inputs on a deployment CPU without ROS or Gazebo."""

import time
from pathlib import Path

import numpy as np
import yaml

from dwarl.runs.store import sha256, write_json
from dwarl.testing.compute import distribution, hardware


def execute(bundle, observations, output, threads=1, iterations=1000, warmup=100):
    import onnxruntime as ort

    if threads <= 0 or iterations <= 0 or warmup < 0:
        raise ValueError("threads/iterations must be positive and warmup nonnegative")
    bundle, observations = Path(bundle), Path(observations)
    deploy = yaml.safe_load((bundle / "deploy.yaml").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    session = ort.InferenceSession(str(bundle / "policy.onnx"), options, providers=["CPUExecutionProvider"])
    with np.load(observations, allow_pickle=False) as data:
        if "model_sha256" in data and str(data["model_sha256"]) != sha256(bundle / "policy.onnx"):
            raise ValueError("recorded observations belong to a different ONNX model")
        inputs = {entry["name"]: np.asarray(data[entry["group"]], dtype=np.float32).reshape(
            (-1, *entry["shape"][1:])) for entry in deploy["model"]["inputs"]}
    counts = {len(v) for v in inputs.values()}
    if len(counts) != 1 or not next(iter(counts)):
        raise ValueError("recorded inputs must have equal, nonzero sample counts")
    count = next(iter(counts))
    feeds = [{k: v[i:i+1] for k, v in inputs.items()} for i in range(count)]
    for i in range(warmup):
        session.run(None, feeds[i % count])
    timings = []
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    for i in range(iterations):
        start = time.perf_counter()
        session.run(None, feeds[i % count])
        timings.append((time.perf_counter() - start) * 1000)
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    report = {"schema": "dwarl_model_benchmark/1", "hardware": hardware(), "onnxruntime": ort.__version__,
              "model_sha256": sha256(bundle / "policy.onnx"), "observations_sha256": sha256(observations),
              "provider": session.get_providers(), "intra_op_threads": threads, "inter_op_threads": 1,
              "warmup": warmup, "iterations": iterations, "recorded_samples": count,
              "inference_ms": distribution(timings), "wall_elapsed_s": wall,
              "cpu_percent": cpu / wall * 100, "control_budget_ms": deploy["control"]["dt"] * 1000,
              "notes": "Batch-one Python ORT replay; excludes observation preparation, Nav2 and simulation."}
    write_json(Path(output), report)
    print(f"Inference p95: {report['inference_ms']['p95']:.3f} ms; results: {output}")
    return report
