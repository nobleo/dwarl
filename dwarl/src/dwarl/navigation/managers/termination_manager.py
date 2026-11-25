"""Evaluate termination terms and distinguish task endings from time-limit truncation."""

import numpy as np

from .base import FunctionTerm, check_batch, config_dict


class TerminationManager:
    def __init__(self, cfg, env):
        self.env, self.cfg = env, config_dict(cfg)
        self.terms = {name: FunctionTerm(term, env) for name, term in self.cfg.items() if term is not None}
        self.term_dones = {name: np.zeros(env.num_envs, bool) for name in self.terms}
        self.terminated = np.zeros(env.num_envs, bool)
        self.time_outs = np.zeros(env.num_envs, bool)

    def get_term(self, name):
        return self.term_dones.get(name, np.zeros(self.env.num_envs, bool))

    def get_outcome(self, role):
        """Union of terms with this semantic role, independent of their configured names."""
        if role == "timeout":
            return self.time_outs.copy()
        if role not in ("success", "collision", "stuck"):
            raise ValueError(f"unknown termination outcome {role!r}")
        value = np.zeros(self.env.num_envs, bool)
        for name, term in self.terms.items():
            if term.cfg.get("role") == role:
                value |= self.term_dones[name]
        return value

    @property
    def outcomes(self):
        return {role: self.get_outcome(role) for role in ("success", "collision", "timeout", "stuck")}

    @property
    def episode_limits(self):
        """Earliest declared per-environment time limit, or an unbounded sentinel."""
        limits = np.full(self.env.num_envs, np.iinfo(np.int32).max)
        for term in self.terms.values():
            if term.cfg.get("time_out", False) and hasattr(term.func, "max_steps"):
                limits = np.minimum(limits, term.func.max_steps)
        return limits

    def compute(self):
        self.terminated[:] = False
        self.time_outs[:] = False
        for name, term in self.terms.items():
            value = term(self.env)
            check_batch(value, self.env.num_envs, name, scalar=True)
            if value.dtype != bool:
                raise TypeError(f"termination {name!r} must return booleans")
            self.term_dones[name] = value.copy()
            if term.cfg.get("time_out", False):
                self.time_outs |= value
            else:
                self.terminated |= value
        # Actual task termination takes precedence over a simultaneous time limit.
        self.time_outs &= ~self.terminated
        for name, term in self.terms.items():
            if term.cfg.get("time_out", False):
                self.term_dones[name] &= ~self.terminated
        return self.terminated | self.time_outs

    def reset(self, env_ids):
        log = {f"Termination_Terms/{name}": float(value[env_ids].mean())
               for name, value in self.term_dones.items()}
        log.update({f"Episode_Termination/{role}": float(value[env_ids].mean())
                    for role, value in self.outcomes.items()})
        for name, term in self.terms.items():
            self.term_dones[name][env_ids] = False
            term.reset(env_ids)
        self.terminated[env_ids] = False
        self.time_outs[env_ids] = False
        return log
