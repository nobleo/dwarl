"""Manage task guidance (for navigation: path-relative local targets)."""

from abc import ABC, abstractmethod

from .base import ManagerTermBase, build_class_term, config_dict


class CommandTerm(ManagerTermBase, ABC):
    """Task guidance with a public ``command`` batch and per-environment reset state."""

    @abstractmethod
    def compute(self, dt):
        """Update the command from the scene; dt=0 refreshes it without advancing time."""


class CommandManager:
    def __init__(self, cfg, env):
        self.terms = {name: build_class_term(term, env, CommandTerm)
                      for name, term in config_dict(cfg).items() if term is not None}

    def get_term(self, name):
        return self.terms[name]

    def get_command(self, name):
        return self.terms[name].command

    def compute(self, dt):
        for term in self.terms.values():
            term.compute(dt)

    def reset(self, env_ids):
        for term in self.terms.values():
            term.reset(env_ids)
