"""Split policy actions into configurable stateful actuator terms."""

from abc import ABC, abstractmethod

import numpy as np

from .base import ManagerTermBase, build_class_term, config_dict


class ActionTerm(ManagerTermBase, ABC):
    """Stateful action component. Process once per control tick; apply the resulting actuator targets."""

    @property
    @abstractmethod
    def action_dim(self):
        """Number of policy outputs consumed by this term."""

    @abstractmethod
    def process_actions(self, actions):
        """Transform clipped policy actions, updating persistent actuator-command state."""

    @abstractmethod
    def apply_actions(self):
        """Write the processed targets into the simulator."""


class ActionManager:
    def __init__(self, cfg, env):
        self.env = env
        self.cfg = config_dict(cfg)
        self.terms = {name: build_class_term(term, env, ActionTerm)
                      for name, term in self.cfg.items() if term is not None}
        self.total_action_dim = sum(term.action_dim for term in self.terms.values())
        if self.total_action_dim == 0:
            raise ValueError("at least one action term is required")
        self.action = np.zeros((env.num_envs, self.total_action_dim))
        self.prev_action = np.zeros_like(self.action)

    def get_term(self, name):
        return self.terms[name]

    def process_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action.shape or not np.isfinite(action).all():
            raise ValueError(f"expected finite actions of shape {self.action.shape}, got {action.shape}")
        self.prev_action[:] = self.action
        offset = 0
        for name, term in self.terms.items():
            end = offset + term.action_dim
            clip = self.cfg[name].get("clip", [-1., 1.])
            self.action[:, offset:end] = np.clip(action[:, offset:end], *clip) if clip else action[:, offset:end]
            term.process_actions(self.action[:, offset:end])
            offset = end

    def apply_action(self):
        for term in self.terms.values():
            term.apply_actions()

    def reset(self, env_ids):
        self.action[env_ids] = 0.
        self.prev_action[env_ids] = 0.
        for term in self.terms.values():
            term.reset(env_ids)
