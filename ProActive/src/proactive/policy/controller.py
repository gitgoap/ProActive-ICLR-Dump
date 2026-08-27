"""Legal-action masking and the plan-fixed STOP rule."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from proactive.train.state_data import ACTION_ORDER, PROBE_ORDER


def masked_action_values(
    values: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    if values.shape != legal_mask.shape or values.shape[-1] != len(ACTION_ORDER):
        raise ValueError("Action values/mask must share shape [..., 8]")
    legal = legal_mask.bool()
    if not torch.all(legal[..., -1]):
        raise ValueError("STOP must be legal in every state")
    return values.masked_fill(~legal, float("-inf"))


def select_action(values: Sequence[float], legal_mask: Sequence[bool]) -> str:
    """Select STOP iff no legal non-stop value is strictly positive."""

    value = np.asarray(values, dtype=np.float64)
    legal = np.asarray(legal_mask, dtype=bool)
    if value.shape != (len(ACTION_ORDER),) or legal.shape != value.shape:
        raise ValueError("Expected one value/mask for every action")
    if not np.isfinite(value).all():
        raise ValueError("Action values must be finite")
    if not legal[-1]:
        raise ValueError("STOP must be legal")
    legal_non_stop = np.flatnonzero(legal[:-1])
    if legal_non_stop.size == 0:
        return "stop"
    best = int(legal_non_stop[np.argmax(value[legal_non_stop])])
    if value[best] <= 0.0:
        return "stop"
    return PROBE_ORDER[best]


def validate_trace(actions: Sequence[str], max_budget: int, relation_legal: bool) -> None:
    probes = [action for action in actions if action != "stop"]
    if len(probes) != len(set(probes)):
        raise ValueError("Policy trace repeats a probe")
    if len(probes) > max_budget:
        raise ValueError("Policy trace exceeds its forward-pass budget")
    if "relation" in probes and not relation_legal:
        raise ValueError("Policy trace acquired an inapplicable relation probe")
    if "stop" in actions and actions[-1] != "stop":
        raise ValueError("No action may follow STOP")

