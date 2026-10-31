"""Small, observational TensorBoard diagnostics; no training behavior changes."""

import numpy as np

from dwarl.training.runner_adapter import RunnerAdapter


def attach_diagnostics(runner, env, cfg, checks):
    def log(iteration):
        values = {"Progress/iteration": iteration, "Progress/target_iterations": cfg.train.max_iterations,
                  "Progress/fraction": min(1., (iteration + 1) / cfg.train.max_iterations),
                  "EarlyStop/enabled": cfg.train.early_stop.enabled,
                  "Validation/enabled": bool(cfg.train.validation_interval),
                  "Benchmark/enabled": bool(cfg.train.benchmark_interval)}
        sampler = env.sampler
        if sampler.enabled:
            values.update({"Curriculum/level": sampler.level, "Curriculum/window_episodes": sampler.window_episodes,
                           "Curriculum/window_success": sampler.last_window_success,
                           "Curriculum/promote_threshold": sampler.promote,
                           "Curriculum/demote_threshold": sampler.demote,
                           "Curriculum/available_pairs": np.count_nonzero(env.dataset.rank <= sampler.level)})
        for tag, value in values.items():
            checks.scalar(tag, value, iteration)

    RunnerAdapter.for_runner(runner).after_log.append(log)
