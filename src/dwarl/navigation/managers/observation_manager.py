"""Compute named observation groups from ordered, configurable MDP terms."""

import numpy as np
import torch
from tensordict import TensorDict

from .base import FunctionTerm, check_batch, config_dict


class ObservationManager:
    def __init__(self, cfg, env):
        self.env, self.cfg = env, config_dict(cfg)
        self.groups = {
            name: {key: FunctionTerm(term, env) for key, term in group["terms"].items() if term is not None}
            for name, group in self.cfg.items() if group is not None
        }
        if any(not terms for terms in self.groups.values()):
            raise ValueError("observation groups must contain at least one term")

    def get_term(self, group, name):
        return self.groups[group][name].func

    def compute(self):
        groups = {}
        for name, terms in self.groups.items():
            values = []
            for key, term in terms.items():
                value = term(self.env)
                check_batch(value, self.env.num_envs, f"{name}.{key}")
                if term.cfg.get("clip") is not None:
                    value = np.clip(value, *term.cfg["clip"])
                value = value * term.cfg.get("scale", 1.)
                values.append(value)
            value = np.concatenate(values, axis=int(self.cfg[name].get("concatenate_dim", -1)))
            groups[name] = torch.as_tensor(value, dtype=torch.float32, device=self.env.device)
        return TensorDict(groups, batch_size=[self.env.num_envs], device=self.env.device)

    def reset(self, env_ids):
        for terms in self.groups.values():
            for term in terms.values():
                term.reset(env_ids)
