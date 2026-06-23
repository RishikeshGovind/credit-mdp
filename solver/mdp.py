"""A tiny, readable generic interface for multi-objective sequential decisions.

No RL framework. The contract is deliberately small so the lending problem (and any
other problem) can plug in:

* ``reset(rng)``       -> initial observation
* ``legal_actions()``  -> actions available in the current state
* ``step(action, rng)``-> (observation, reward_vector, done, info)
* ``done``             -> episode finished?

Environments are *stateful* (gym-like): they hold the current state internally,
which keeps the lending dynamics far more readable than threading an immutable
state object that carries a whole loan ledger. ``reward_vector`` is a 1-D numpy
array of length ``n_objectives`` and is always expressed so that **larger is
better** for every component (minimisation objectives are negated inside the env).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


class MultiObjectiveMDP(ABC):
    """Contract for a multi-objective sequential decision environment."""

    n_objectives: int
    objective_names: list[str]

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> Any:
        """Start a new episode; return the first observation."""

    @abstractmethod
    def legal_actions(self) -> list[Any]:
        """Actions available in the current state."""

    @abstractmethod
    def step(self, action: Any, rng: np.random.Generator):
        """Apply ``action``; return (observation, reward_vector, done, info)."""

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether the current episode has terminated."""

    def summary(self) -> dict:
        """Optional end-of-episode bookkeeping (objectives beyond the reward sum)."""
        return {}


# A policy maps the current observation (and the env, for legal actions) to an action.
Policy = Callable[[Any, MultiObjectiveMDP], Any]


@dataclass
class EpisodeResult:
    reward_vector: np.ndarray            # summed per-step reward vector
    summary: dict = field(default_factory=dict)  # env-reported aggregate metrics


def rollout(env: MultiObjectiveMDP, policy: Policy,
            rng: np.random.Generator) -> EpisodeResult:
    """Run one episode of ``env`` under ``policy``; accumulate the reward vector."""
    obs = env.reset(rng)
    total = np.zeros(env.n_objectives, dtype=float)
    while not env.done:
        action = policy(obs, env)
        obs, reward_vec, done, _info = env.step(action, rng)
        total += reward_vec
    return EpisodeResult(reward_vector=total, summary=env.summary())
