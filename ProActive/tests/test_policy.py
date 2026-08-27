from __future__ import annotations

import pytest
import torch

from proactive.networks.losses import policy_loss
from proactive.policy.controller import masked_action_values, select_action, validate_trace
from proactive.train.state_data import ACTION_ORDER


def test_policy_mask_and_stop_rule() -> None:
    values = [0.0] * len(ACTION_ORDER)
    mask = [True] * len(ACTION_ORDER)
    assert select_action(values, mask) == "stop"
    values[1] = 0.01
    assert select_action(values, mask) == ACTION_ORDER[1]
    mask[1] = False
    assert select_action(values, mask) == "stop"


def test_tensor_mask_requires_legal_stop() -> None:
    values = torch.zeros(2, len(ACTION_ORDER))
    legal = torch.ones_like(values, dtype=torch.bool)
    legal[0, 0] = False
    masked = masked_action_values(values, legal)
    assert torch.isneginf(masked[0, 0])
    legal[1, -1] = False
    with pytest.raises(ValueError, match="STOP"):
        masked_action_values(values, legal)


def test_policy_loss_uses_stop_on_zero_tie_and_is_finite() -> None:
    target = torch.zeros(1, len(ACTION_ORDER))
    legal = torch.ones_like(target, dtype=torch.bool)
    good = torch.zeros_like(target)
    good[0, -1] = 5.0
    bad = torch.zeros_like(target)
    bad[0, 0] = 5.0
    assert policy_loss(good, target, legal).action_ce < policy_loss(bad, target, legal).action_ce
    assert torch.isfinite(policy_loss(good, target, legal).total)


def test_trace_rejects_repeats_budget_and_relation() -> None:
    with pytest.raises(ValueError, match="repeats"):
        validate_trace(["blur", "blur", "stop"], 2, False)
    with pytest.raises(ValueError, match="budget"):
        validate_trace(["blur", "crop", "stop"], 1, False)
    with pytest.raises(ValueError, match="inapplicable"):
        validate_trace(["relation", "stop"], 1, False)
