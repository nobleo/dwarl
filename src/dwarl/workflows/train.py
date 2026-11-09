"""Training owns its environment; consumers only see completed checkpoints."""

import os
import random
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
from omegaconf import OmegaConf

from dwarl.runs.store import complete_checkpoint, lock, resolve_checkpoint, resolve_run, write_json


def execute(cfg, assignments=()):
    print("Training: loading dependencies and preparing inputs…", flush=True)
    import torch

    from dwarl.artifacts.exporter import deploy_spec
    from dwarl.datasets.cache import prepare
    from dwarl.robot.config import materialize
    from dwarl.robot.profile import load_profile
    from dwarl.training.checkpoint import attach, restore
    from dwarl.training.contract import build_contract, device, load_contract
    from dwarl.training.environment.nav_env import NavVecEnv
    from dwarl.training.runner_adapter import RunnerAdapter
    from dwarl.training.validation import EarlyStop, attach_validation

    checkpoint = None
    if cfg.train.resume:
        run = resolve_run(cfg.train.resume)
        checkpoint = resolve_checkpoint(run)
        saved = OmegaConf.load(run / "config.yaml")
        from dwarl.config.schema import TrainConfig

        saved.train = OmegaConf.merge(OmegaConf.structured(TrainConfig), saved.train)
        permitted = {"train.resume", "train.device", "train.max_iterations", "train.show", "train.detached",
                     "train.early_stop.enabled"}
        for assignment in assignments:
            key = assignment.partition("=")[0]
            if key not in permitted:
                if OmegaConf.select(saved, key) == OmegaConf.select(cfg, key):
                    continue
                raise ValueError(f"resume cannot change {key}; supported changes: {sorted(permitted)}")
            OmegaConf.update(saved, key, OmegaConf.select(cfg, key))
        cfg = saved
        from dwarl.config.compose import validate_training

        validate_training(cfg.train)
        contract = load_contract(run)
        print("Resume: saved recipe restored; simulator episodes restart.")
    else:
        if not cfg.robot:
            raise ValueError("select a robot: dwarl train <integration> or robot in dwarl.yaml")
        profile = load_profile(cfg.robot)
        name = cfg.train.name
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("train.name must be a simple directory name")
        run = Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}_{uuid4().hex[:6]}"
        run = run.resolve()
        run.mkdir(parents=True)
        print(f"Training run: {run}", flush=True)
        print("Training: building and freezing robot assets…", flush=True)
        contract = build_contract(cfg)
        materialize(contract, run / "robot")
        dataset, dataset_id = prepare(cfg, profile, Path(".dwarl/datasets"))
        contract.dataset.path = str(dataset)
        OmegaConf.save(cfg, run / "config.yaml", resolve=True)
        OmegaConf.save(contract, run / "contract.yaml", resolve=True)
        from dwarl.runs.provenance import source_identity, versions

        write_json(run / "provenance.json", {"schema": "dwarl_run/1", "created_ns": time.time_ns(),
                   "source_identity": source_identity(), "versions": versions(),
                   "robot": profile.id, "source_profile": str(profile.path), "source_sha256": profile.sha256,
                   "dataset": dataset_id, "deployment_ready": cfg.train.deployment_ready})
        write_json(run / "checkpoint_manifest.json", {"schema": "dwarl_checkpoints/1", "completed": [], "best": None})
    with lock(run / ".training.lock", blocking=False):
        contract.device = device(cfg.train.device)
        contract.sim.gui = cfg.train.show
        torch.manual_seed(contract.seed)
        random.seed(contract.seed)
        np.random.seed(contract.seed)
        from dwarl.runs.process import process_identity

        write_json(run / "status.json", {"state": "running", "pid": os.getpid(),
                                         "process_identity": process_identity(os.getpid())})
        env = None
        try:
            from rsl_rl.runners import OnPolicyRunner

            print(f"Training: creating {cfg.train.num_envs} environments on {contract.device}…", flush=True)
            env = NavVecEnv(contract)
            if not cfg.train.show:
                env.viewer_trigger = run / "live_view"
                env.viewer_trigger.unlink(missing_ok=True)
            if cfg.train.deployment_ready:
                deploy_spec(contract, env)  # fail before PPO starts
            runner = OnPolicyRunner(env, OmegaConf.to_container(contract.agent, resolve=True),
                                    log_dir=str(run / "checkpoints"), device=contract.device)
            adapter = RunnerAdapter.for_runner(runner)
            adapter.after_save.append(lambda path: complete_checkpoint(run, path, runner.current_learning_iteration))
            checks = attach_validation(runner, run, cfg)
            attach(runner, env, checks.state_dict)
            from dwarl.training.diagnostics import attach_diagnostics

            attach_diagnostics(runner, env, cfg, checks)
            if checkpoint:
                infos = runner.load(str(checkpoint.path), map_location=contract.device)
                runner.current_learning_iteration += 1
                if infos and "dwarl" in infos:
                    restore(env, infos["dwarl"])
                checks.restore(checkpoint, infos)
            print(f"Training run: {run}", flush=True)
            print(f"TensorBoard: tensorboard --logdir {run / 'checkpoints'}", flush=True)
            if not cfg.train.show:
                print(f"Live view: dwarl watch {run}  (toggle; slows training while open)", flush=True)
            reason = "iteration_budget"
            try:
                if cfg.train.early_stop.enabled and checks.state["stop_reason"]:
                    raise EarlyStop(checks.state["stop_reason"])
                runner.learn(max(0, cfg.train.max_iterations - runner.current_learning_iteration))
                state = "finished"
            except EarlyStop as exc:
                state, reason = "early_stopped", str(exc)
                print(f"Early stop: {reason}", flush=True)
            except KeyboardInterrupt:
                state, reason = "interrupted", "user_interrupt"
            finally:
                writer = getattr(runner.logger, "writer", None)
                if writer is not None:
                    writer.flush()
                    writer.close()
            write_json(run / "status.json", {"state": state, "pid": None, "reason": reason,
                                             "iteration": runner.current_learning_iteration})
        except BaseException as exc:
            write_json(run / "status.json", {"state": "failed", "pid": None, "error": str(exc)})
            raise
        finally:
            if env is not None:
                env.close()
    return run
