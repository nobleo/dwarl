"""Resume bookkeeping in the existing checkpoint's infos, with explicit fresh episodes.

Model/optimizer/normalizers are RSL-RL's responsibility. We save RNG and sampler
state, but do not claim bitwise uninterrupted continuation of MuJoCo contacts.
"""

import random
from copy import deepcopy

import numpy as np
import torch

from dwarl.training.environment.dataset import CurriculumState


def capture(env):
    from dwarl.runs.store import sha256
    sampler = env.sampler
    curriculum = sampler.curriculum_state()
    return {"version": 2, "resume_mode": "fresh_episodes", "python": random.getstate(),
            "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "env_rng": deepcopy(env.rng.bit_generator.state),
            "sampler_rng": deepcopy(sampler.rng.bit_generator.state), "level": curriculum.level,
            "outcomes": curriculum.window, "last_window_success": curriculum.last_window_success,
            "lidar_rng": deepcopy(env.lidar._rng.bit_generator.state),
             "episode_visits": dict(env.episode_visits),
            "reward_weights": env.reward_manager.weights,
            "dataset_index_sha256": sha256(env.dataset.root / "index.json")}


def restore(env, state):
    if state.get("version") != 2:
        raise ValueError("unsupported checkpoint bookkeeping version")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    env.rng.bit_generator.state = state["env_rng"]
    env.lidar._rng.bit_generator.state = state["lidar_rng"]
    if "reward_weights" in state:
        env.reward_manager.set_weights(state["reward_weights"])
    from dwarl.runs.store import sha256
    if state["dataset_index_sha256"] == sha256(env.dataset.root / "index.json"):
        env.sampler.rng.bit_generator.state = state["sampler_rng"]
        env.sampler.restore_curriculum(CurriculumState(
            state["level"], list(state["outcomes"]), state.get("last_window_success")))
        env.episode_visits = dict(state["episode_visits"])
    env.map_idx[:] = -1
    env.reset()


def attach(runner, env, training_state=None):
    from dwarl.training.runner_adapter import RunnerAdapter

    def state():
        result = {"dwarl": capture(env)}
        if training_state is not None:
            result["dwarl_training"] = training_state()
        return result

    RunnerAdapter.for_runner(runner).checkpoint_state.append(state)


def warm_start(runner, path, device):
    """Copy actor and critic weights whose names and shapes match from checkpoint ``path``.

    Layers whose input changed (for example a critic with more privileged observations) keep their
    fresh initialization, as do the optimizer and the iteration counter, so the new recipe trains from
    the old policy's behaviour rather than resuming its run. The learned exploration noise is kept:
    resetting it to ``init_std`` perturbed a converged policy and its steering grew jerkier.
    """
    saved = torch.load(path, weights_only=False, map_location=device)
    summary = []
    for name, model in (("actor", runner.alg._raw_actor), ("critic", runner.alg._raw_critic)):
        own = model.state_dict()
        source = saved.get(f"{name}_state_dict") or {}
        copied = {key: value for key, value in source.items()
                  if key in own and own[key].shape == value.shape}
        model.load_state_dict({**own, **copied})
        summary.append(f"{name} {len(copied)}/{len(own)} tensors")
    return ", ".join(summary)
