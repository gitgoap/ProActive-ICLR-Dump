"""Acquisition baselines with explicit oracle/non-oracle boundaries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from proactive.policy.controller import select_action
from proactive.teacher.offline import FIXED_BASELINES
from proactive.train.state_data import ACTION_ORDER, PROBE_ORDER


class AcquisitionPolicy:
    name: str

    def select(
        self,
        *,
        instance_id: str,
        step: int,
        acquired: Sequence[str],
        legal_mask: Sequence[bool],
        predicted_values: Sequence[float] | None = None,
    ) -> str:
        raise NotImplementedError


@dataclass
class LearnedVOIPolicy(AcquisitionPolicy):
    name: str = "learned_voi"

    def select(self, **kwargs: object) -> str:
        values = kwargs.get("predicted_values")
        if values is None:
            raise ValueError("Learned VOI policy requires predicted values")
        return select_action(values, kwargs["legal_mask"])


@dataclass
class RandomPolicy(AcquisitionPolicy):
    seed: int = 42
    name: str = "random"

    def select(self, **kwargs: object) -> str:
        legal = [
            action
            for action, allowed in zip(ACTION_ORDER[:-1], kwargs["legal_mask"][:-1])
            if allowed
        ]
        if not legal:
            return "stop"
        key = f"{self.seed}|{kwargs['instance_id']}|{kwargs['step']}"
        index = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % len(legal)
        return legal[index]


@dataclass
class FixedSchedulePolicy(AcquisitionPolicy):
    schedule_name: str

    def __post_init__(self) -> None:
        if self.schedule_name not in FIXED_BASELINES:
            raise ValueError(f"Unknown fixed schedule: {self.schedule_name}")
        self.name = self.schedule_name

    def select(self, **kwargs: object) -> str:
        legal = dict(zip(ACTION_ORDER, kwargs["legal_mask"]))
        for action in FIXED_BASELINES[self.schedule_name]:
            if legal.get(action, False):
                return action
        return "stop"


@dataclass
class UncertaintyGreedyPolicy(AcquisitionPolicy):
    """Non-leaking baseline requiring scores predicted before acquisition.

    The supplied scorer may use the current state and action identity, but its
    contract must not accept or inspect the candidate action's realized cached
    observation.  Supplying realized next-state scores is reserved for oracle
    policies and is rejected by the evaluation manifest.
    """

    score_fn: Callable[[str], float]
    name: str = "uncertainty_greedy"

    def select(self, **kwargs: object) -> str:
        values = [self.score_fn(action) for action in PROBE_ORDER] + [0.0]
        return select_action(values, kwargs["legal_mask"])


@dataclass
class OracleNextPolicy(AcquisitionPolicy):
    realized_values: Mapping[str, float]
    name: str = "oracle_next"

    def select(self, **kwargs: object) -> str:
        values = [float(self.realized_values.get(action, float("-inf"))) for action in PROBE_ORDER] + [0.0]
        return select_action(values, kwargs["legal_mask"])

