from __future__ import annotations

from proactive.eval.baselines import FixedSchedulePolicy, RandomPolicy, UncertaintyGreedyPolicy
from proactive.train.state_data import ACTION_ORDER


def test_random_policy_is_deterministic_per_instance_step() -> None:
    mask = [True] * len(ACTION_ORDER)
    policy = RandomPolicy(seed=42)
    first = policy.select(instance_id="x", step=0, acquired=[], legal_mask=mask)
    second = policy.select(instance_id="x", step=0, acquired=[], legal_mask=mask)
    assert first == second


def test_fixed_policy_respects_mask_and_never_repeats() -> None:
    policy = FixedSchedulePolicy("blank_first")
    mask = [True] * len(ACTION_ORDER)
    mask[0] = False
    assert policy.select(instance_id="x", step=1, acquired=["blank"], legal_mask=mask) == "grounding"


def test_uncertainty_scorer_receives_identity_only() -> None:
    observed = []

    def score(action: str) -> float:
        observed.append(action)
        return 1.0 if action == "crop" else -1.0

    policy = UncertaintyGreedyPolicy(score)
    mask = [True] * len(ACTION_ORDER)
    assert policy.select(instance_id="x", step=0, acquired=[], legal_mask=mask) == "crop"
    assert observed == list(ACTION_ORDER[:-1])
