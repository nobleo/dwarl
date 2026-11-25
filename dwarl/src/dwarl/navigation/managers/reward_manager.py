"""Weighted MDP reward terms; weights are rates, integrated over the control timestep."""

import numpy as np

from .base import FunctionTerm, check_batch, config_dict


class RewardManager:
    def __init__(self, cfg, env):
        self.env, self.cfg = env, config_dict(cfg)
        self.terms = {name: FunctionTerm(term, env) for name, term in self.cfg.items() if term is not None}
        self.episode_sums = {name: np.zeros(env.num_envs) for name in self.terms}
        self.term_values = {name: np.zeros(env.num_envs) for name in self.terms}

    @property
    def weights(self):
        return {name: float(term.cfg["weight"]) for name, term in self.terms.items()}

    def set_weights(self, weights):
        unknown = set(weights) - self.terms.keys()
        if unknown:
            raise KeyError(f"unknown reward terms: {sorted(unknown)}")
        for name, weight in weights.items():
            if not np.isfinite(weight):
                raise ValueError(f"nonfinite reward weight for {name}")
            self.terms[name].cfg["weight"] = float(weight)
            self.env.cfg.rewards[name].weight = float(weight)

    def compute(self, dt):
        total = np.zeros(self.env.num_envs)
        for name, term in self.terms.items():
            weight = float(term.cfg["weight"])
            value = term(self.env) if weight != 0 else np.zeros_like(total)
            check_batch(value, self.env.num_envs, name, scalar=True)
            self.term_values[name] = value * weight * dt
            total += self.term_values[name]
            self.episode_sums[name] += self.term_values[name]
        return total

    def reset(self, env_ids):
        log = {f"Episode_Reward/{name}": float(values[env_ids].mean())
               for name, values in self.episode_sums.items()}
        for name, term in self.terms.items():
            self.episode_sums[name][env_ids] = 0.
            self.term_values[name][env_ids] = 0.
            term.reset(env_ids)
        return log
