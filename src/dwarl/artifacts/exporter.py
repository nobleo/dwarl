"""Policy export for deployment (ONNX / TorchScript + a ``deploy.yaml`` contract).

Follows the Isaac Lab convention: rsl-rl's own ``runner.export_policy_to_onnx/jit`` write the network
(normaliser + lidar encoder + MLP + deterministic head), and the run's resolved config is saved next to it.
``deploy.yaml`` is the small, explicit contract that non-Python consumers (the Nav2 controller) read to
rebuild observations and map actions exactly as in training.

The export directory is a self-contained *policy bundle* (``dwarl deploy`` installs it into a robot
stack)::

    <policy-bundle>/
        policy.onnx         network (state normaliser in-graph), batch 1
        policy.pt           TorchScript twin (Python / debugging)
        deploy.yaml         runtime contract: observation/action mapping, lidar geometry, robot id, ROS iface
        dwarl_robot.yaml   copy of the robot profile the policy was trained for
        env.yaml            full resolved training config
        golden.{npz,json}   optional reference episode (runtime parity tests)
        cpu_bench.json      optional inference benchmark
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

EXPORT_DIR = "exported"
DEPLOY_VERSION = 5  # v4 was the removed arc-travel experiment; never reinterpret it


def state_layout(num_goals: int, command_feedback: bool = False) -> list[str]:
    goals = [f"goal{j}_{k}" for j in range(num_goals) for k in ("dist_norm", "sin", "cos")]
    return goals + ["last_action_v", "last_action_w", "v_norm", "w_norm"] + (
        ["last_command_v_norm", "last_command_w_norm"] if command_feedback else [])


def deploy_spec(cfg: DictConfig, env: Any, iteration: int | None = None, checkpoint: str | None = None) -> dict:
    """Everything a runtime needs to reproduce the policy's observations and actions."""
    from dwarl.navigation.terminations import goal_tolerance

    _validate_deploy_terms(env)
    lc, rc = cfg.lidar, cfg.robot
    command = env.path_command
    scan = env.observation_manager.get_term("lidar", "scan")
    state_terms = env.observation_manager.cfg["state"]["terms"]
    n = int(lc.num_rays)
    fov = math.radians(float(lc.fov_deg))
    full = math.isclose(float(lc.fov_deg), 360.0)
    obs = env.get_observations()
    hist = scan.history
    goals = list(command.lookahead) if command.mode == "lookahead" else [0.0]
    origin = [float(v) for v in env.lidar_origin]
    pose = OmegaConf.to_container(rc.get("lidar_pose") or {}, resolve=True)
    feedback = state_terms.get("command_feedback") is not None
    return {
        "version": DEPLOY_VERSION if feedback else 3,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": {"checkpoint": checkpoint, "iteration": iteration},
        "robot": {
            "id": str(rc.get("id", "")),
            "profile_sha256": str(rc.get("profile_sha256", "")),
            "drive_type": "diff",
            "footprint_radius": float(env.radius),
            "footprint": OmegaConf.to_container(rc.footprint) if rc.get("footprint") else None,
            "wheel_separation": float(env.wheel_sep),
            "wheel_radius": float(env.wheel_r),
        },
        "model": {
            "onnx": "policy.onnx",
            "jit": "policy.pt",
            "inputs": [
                {"name": "obs", "shape": [1, int(obs["state"].shape[-1])], "group": "state"},
                {"name": "lidar", "shape": [1, 1, len(hist), n], "group": "lidar"},
            ],
            "outputs": [{"name": "actions", "shape": [1, 2]}],
            "normalization": "in_graph",  # state normaliser is baked into the ONNX graph
        },
        "control": {"dt": float(cfg.env.control_dt)},
        "lidar": {
            "num_rays": n,
            # ray i: from `origin` (base frame, metres) along base-frame bearing angle_min + i * angle_increment
            "origin": origin,
            "angle_min": -math.pi if full else -fov / 2,
            "angle_increment": 2 * math.pi / n if full else fov / (n - 1),
            "direction": "ccw",
            "full_circle": full,
            "min_range": float(lc.min_range),
            "max_range": float(lc.max_range),
            "no_return": "max_range",  # inf / nan / out of range -> max_range, then clip to [min, max]
            # linear: (r - min_range) / (max_range - min_range)
            # log:    log1p(max(r - min_range, 0) / 0.1) / log1p((max_range - min_range) / 0.1)
            "normalize": scan.normalization,
            # lidar input row k = normalised scan from history[k] control steps ago (row 0 = current scan);
            # before enough scans exist (start / plan reset) missing rows repeat the oldest available scan
            "history": hist,
            # training mount (informational): real scans are re-projected with the live TF instead
            "mount": pose or None,
        },
        "state": {
            "layout": state_layout(len(goals), feedback),
            **({"command_feedback": True} if feedback else {}),
            "max_goal_distance": float(state_terms["goals"]["params"]["max_distance"]),
            "v_scale": float(env.v_max),  # v_norm = measured v / v_scale
            "w_scale": float(env.w_max),  # w_norm = measured w / w_scale
            "last_action": "clipped policy output of the previous step (zeros after reset)",
        },
        "goal": {
            "mode": command.mode,
            "lookahead_distance": command.main_distance,
            "lookahead_distances": goals,  # state goal j = point at s + lookahead_distances[j] (final: the goal)
            "tolerance": goal_tolerance(env.cfg.terminations, require_standard=True),
            "projection_window": list(command.window),
            **({"tracker": "segment_window_v1"} if feedback else {}),
        },
        "action": {
            "clip": [-1.0, 1.0],
            # absolute: target v = v_min + (a0 + 1) / 2 * (v_max - v_min), w = a1 * w_max, then rate limited
            # window:   v = clip(cmd_v + a0 * a_max * dt, v_min, v_max), w = clip(cmd_w + a1 * alpha_max * dt, +-w_max)
            "mode": env.drive_action.mode,
            "v": {"min": float(rc.v_min), "max": float(rc.v_max)},  # v = v_min + (a0 + 1) / 2 * (v_max - v_min)
            "w": {"max": float(rc.w_max)},  # w = a1 * w_max
            "rate_limit": {"a_max": float(rc.a_max), "alpha_max": float(rc.alpha_max)},  # per dt, on the command
        },
        "ros": OmegaConf.to_container(rc.ros, resolve=True) if rc.get("ros") else {},
    }


def _validate_deploy_terms(env):
    """The v3/v5 runtimes implement the standard navigation terms, not arbitrary training experiments."""
    from dwarl.navigation.actions import DifferentialDriveAction
    from dwarl.navigation.commands import PathCommand

    if list(env.cfg.agent.obs_groups.actor) != ["lidar", "state"]:
        raise ValueError("deployment requires the standard lidar/state actor observation groups")
    if (list(env.action_manager.terms) != ["drive"] or type(env.drive_action) is not DifferentialDriveAction
            or type(env.path_command) is not PathCommand
            or env.action_manager.cfg["drive"].get("clip") != [-1., 1.]):
        raise ValueError("deployment requires the standard differential-drive action and path command")
    expected = {
        "lidar": {"scan": "LidarHistory"},
        "state": {"goals": "goal_points", "last_action": "last_action", "velocity": "base_velocity"},
    }
    if env.observation_manager.cfg["state"]["terms"].get("command_feedback") is not None:
        expected["state"]["command_feedback"] = "last_command"
    for group, terms in expected.items():
        configured = env.observation_manager.groups[group]
        if list(configured) != list(terms):
            raise ValueError(f"unsupported deployment observation layout: {group}")
        for name, function in terms.items():
            term = configured[name].cfg
            if (term["func"] != f"dwarl.navigation.observations:{function}" or term.get("clip") is not None
                    or term.get("scale", 1.) != 1.
                    or env.observation_manager.cfg[group].get("concatenate_dim", -1) != -1):
                raise ValueError(f"unsupported deployment observation term: {group}.{name}")
            params = term.get("params", {})
            if params.get("command_name", "path") != "path" or params.get("action_name", "drive") != "drive":
                raise ValueError(f"unsupported deployment term source: {group}.{name}")


def export_runner(runner, cfg: DictConfig, env: Any, out: Path, iteration: int | None = None,
                  checkpoint: str | None = None, jit: bool = False) -> Path:  # fmt: skip
    """Export a loaded ``OnPolicyRunner`` (rsl-rl built-ins) plus env.yaml / deploy.yaml into ``out``."""
    spec = deploy_spec(cfg, env, iteration=iteration, checkpoint=checkpoint)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"export staging directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_onnx(path=str(out), filename="policy.onnx")
    _inline_onnx_weights(out / "policy.onnx")
    if jit:
        runner.export_policy_to_jit(path=str(out), filename="policy.pt")
    else:
        spec["model"].pop("jit", None)
    OmegaConf.save(cfg, out / "env.yaml", resolve=True)
    from dwarl.runs.store import sha256
    if checkpoint and Path(checkpoint).is_file():
        import json
        ck = Path(checkpoint)
        spec["source"]["checkpoint_sha256"] = sha256(ck)
        training_config = ck.parent / "config.yaml"
        if training_config.exists():
            spec["source"]["training_config_sha256"] = sha256(training_config)
        provenance = ck.parent / "provenance.json"
        if provenance.exists():
            spec["source"]["training_provenance"] = json.loads(provenance.read_text())
    (out / "deploy.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    prof = env.cfg.robot.get("profile_file")
    if prof and Path(prof).is_file():
        shutil.copyfile(prof, out / "dwarl_robot.yaml")
        bundle = Path(prof).parent
        if (bundle / "manifest.json").is_file():
            # The frozen profile references its model and assets relative to itself.
            import json

            manifest = json.loads((bundle / "manifest.json").read_text())
            for relative in manifest["files"]:
                destination = out / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(bundle / relative, destination)
            shutil.copyfile(bundle / "manifest.json", out / "manifest.json")
    return out


def _inline_onnx_weights(path: Path) -> None:
    """torch's exporter may write weights to ``policy.onnx.data``; store a single self-contained file instead."""
    import onnx

    model = onnx.load(str(path))  # also loads external data
    onnx.save(model, str(path), save_as_external_data=False)
    Path(str(path) + ".data").unlink(missing_ok=True)


class OnnxPolicy:
    """Drop-in replacement for the torch inference policy that runs an exported ``policy.onnx``."""

    def __init__(self, path: str | Path, device: str = "cpu") -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.device = device

    def __call__(self, obs):
        import numpy as np
        import torch

        state = obs["state"].detach().cpu().numpy().astype(np.float32)
        lidar = obs["lidar"].detach().cpu().numpy().astype(np.float32)
        out = np.concatenate([self.session.run(["actions"], {"obs": state[i : i + 1], "lidar": lidar[i : i + 1]})[0]
                              for i in range(len(state))])  # fmt: skip  (exported with batch 1)
        return torch.as_tensor(out, device=self.device)


def benchmark_policy(export_dir: str | Path, num_rays: int = 180, iters: int = 1000, warmup: int = 100,
                     threads: list[int] | None = None) -> dict[str, Any]:
    """Benchmark exported ONNX and TorchScript policies on CPU.

    Args:
        export_dir: directory containing policy.onnx and policy.pt
        num_rays: lidar ray count (from deploy.yaml if needed)
        iters: iterations to time (after warmup)
        warmup: warmup iterations (discarded)
        threads: list of intra-op thread counts to test for ONNX (e.g., [0, 1])
                 0 = ORT default, 1 = single thread (often best for tiny models)

    Returns:
        dict with "onnx" and "jit" keys, each containing latency stats (ms, Hz) and metadata.
        If a backend is missing or fails, it is omitted with a warning logged.
    """
    import numpy as np

    if iters <= 0 or warmup < 0 or (threads is not None and (not threads or any(t < 0 for t in threads))):
        raise ValueError("benchmark requires iters > 0, warmup >= 0 and nonnegative thread counts")
    export_dir = Path(export_dir)
    results = {}

    # Read deploy.yaml to get input shapes
    deploy_path = export_dir / "deploy.yaml"
    if deploy_path.exists():
        with open(deploy_path) as f:
            deploy_spec = yaml.safe_load(f)
        obs_shape = deploy_spec["model"]["inputs"][0]["shape"]
        lidar_shape = deploy_spec["model"]["inputs"][1]["shape"]
        num_rays = lidar_shape[-1]
    else:
        obs_shape = [1, 7]
        lidar_shape = [1, 1, 1, num_rays]  # (no deploy.yaml: assume a single scan)

    # Synthetic inputs (fixed seed for reproducibility)
    rng = np.random.default_rng(0)
    obs_data = rng.standard_normal(obs_shape).astype(np.float32)
    lidar_data = rng.random(lidar_shape).astype(np.float32)

    cpu_model, num_cpus = _cpu_info()

    # Benchmark ONNX Runtime
    onnx_path = export_dir / "policy.onnx"
    if onnx_path.exists():
        try:
            import onnxruntime as ort
        except ImportError:
            print("[benchmark] skipping ONNX: onnxruntime not installed; run through pixi")
            onnx_path = None

    if onnx_path and onnx_path.exists():
        threads_to_test = threads if threads is not None else [0, 1]
        for thread_count in threads_to_test:
            key = f"onnx_{thread_count}t" if thread_count > 0 else "onnx_default"
            try:
                sess_opts = ort.SessionOptions()
                if thread_count > 0:
                    sess_opts.intra_op_num_threads = thread_count
                    sess_opts.inter_op_num_threads = 1
                sess = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])

                # Warmup
                for _ in range(warmup):
                    sess.run(["actions"], {"obs": obs_data, "lidar": lidar_data})

                # Timed iterations
                times_ns = []
                for _ in range(iters):
                    t0 = time.perf_counter_ns()
                    sess.run(["actions"], {"obs": obs_data, "lidar": lidar_data})
                    times_ns.append(time.perf_counter_ns() - t0)

                times_ms = np.array(times_ns) / 1e6
                results[key] = {
                    "mean_ms": float(np.mean(times_ms)),
                    "median_ms": float(np.median(times_ms)),
                    "p95_ms": float(np.percentile(times_ms, 95)),
                    "p99_ms": float(np.percentile(times_ms, 99)),
                    "max_ms": float(np.max(times_ms)),
                    "hz": float(1000.0 / np.mean(times_ms)),
                    "metadata": {
                        "backend": "onnx",
                        "threads": thread_count if thread_count > 0 else "default",
                        "version": ort.__version__,
                    },
                }
            except Exception as e:
                print(f"[benchmark] ONNX {key} failed: {e}")

    # Benchmark TorchScript
    jit_path = export_dir / "policy.pt"
    if jit_path.exists():
        try:
            import torch
        except ImportError:
            print("[benchmark] skipping JIT: torch not installed")
            jit_path = None

    if jit_path and jit_path.exists():
        try:
            jit = torch.jit.load(str(jit_path))
            jit.eval()

            # Prepare inputs: state, then list of 2D CNN inputs [lidar]
            obs_t = torch.from_numpy(obs_data)
            lidar_t = torch.from_numpy(lidar_data)

            # Warmup
            with torch.inference_mode():
                for _ in range(warmup):
                    jit(obs_t, [lidar_t])

            # Timed iterations
            times_ns = []
            with torch.inference_mode():
                for _ in range(iters):
                    t0 = time.perf_counter_ns()
                    jit(obs_t, [lidar_t])
                    times_ns.append(time.perf_counter_ns() - t0)

            times_ms = np.array(times_ns) / 1e6
            results["jit"] = {
                "mean_ms": float(np.mean(times_ms)),
                "median_ms": float(np.median(times_ms)),
                "p95_ms": float(np.percentile(times_ms, 95)),
                "p99_ms": float(np.percentile(times_ms, 99)),
                "max_ms": float(np.max(times_ms)),
                "hz": float(1000.0 / np.mean(times_ms)),
                "metadata": {
                    "backend": "torchscript",
                    "version": torch.__version__,
                },
            }
        except Exception as e:
            print(f"[benchmark] JIT failed: {e}")

    # Add system info to all results
    system_info = {
        "cpu_model": cpu_model,
        "num_cpus": num_cpus,
        "iters": iters,
        "warmup": warmup,
        "inputs": {
            "obs_shape": obs_shape,
            "lidar_shape": lidar_shape,
        },
    }

    if not results:
        raise RuntimeError("no inference backend completed the benchmark")
    return {
        "system": system_info,
        "backends": results,
    }


def _cpu_info() -> tuple[str, int]:
    model = platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return model, n


def dump_golden(env: Any, policy, path: str | Path, steps: int) -> int:
    """Record one noise-free episode (env 0) as reference data for re-implementations (C++ controller).

    Per step t (before acting): pose [x, y, yaw] and velocity [v, w] (tile frame), raw lidar ranges [m],
    last action, current command, the observation groups, and the policy action / resulting command.
    """
    import numpy as np
    import torch

    env.lidar.noise_std, env.lidar.dropout_prob = 0.0, 0.0
    env.reset()
    rec: dict[str, list] = {k: [] for k in ("pose", "vel", "ranges", "last_action", "cmd", "obs_state", "obs_lidar",
                                            "action", "cmd_next", "goal")}  # fmt: skip
    L = int(env.path_command.path_len[0])
    plan = env.path_command.path[0, :L].copy()
    obs = env.get_observations()
    with torch.no_grad():
        for _ in range(steps):
            xy, yaw = env.world.robot_pose()
            v, w = env.world.robot_velocity(yaw)
            rec["pose"].append([xy[0, 0], xy[0, 1], yaw[0]])
            rec["vel"].append([v[0], w[0]])
            rec["ranges"].append(env.ranges[0].copy())
            rec["last_action"].append(env.action_manager.action[0].copy())
            rec["cmd"].append(env.drive_action.command[0].copy())
            rec["goal"].append(env.path_command.main_goal[0].copy())
            rec["obs_state"].append(obs["state"][0].cpu().numpy())
            rec["obs_lidar"].append(obs["lidar"][0].reshape(-1).cpu().numpy())
            a = policy(obs)
            rec["action"].append(a[0].cpu().numpy())
            obs, _, dones, _ = env.step(a)
            if bool(dones[0]):
                break
            rec["cmd_next"].append(env.drive_action.command[0].copy())
    n = len(rec["cmd_next"])
    arrays = {k: np.asarray(v[:n], dtype=np.float64) for k, v in rec.items()}
    np.savez(path, plan=plan.astype(np.float64), **arrays)
    import json  # same data as JSON for non-Python consumers (C++ controller tests)

    js = {"plan": plan.tolist(), **{k: v.tolist() for k, v in arrays.items()}}
    Path(path).with_suffix(".json").write_text(json.dumps(js))
    return n
