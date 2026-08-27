from __future__ import annotations

import torch

from proactive.networks.diagnostic import build_diagnostic_model
from proactive.train.state_data import collate_vectorized_states, vectorize_state
from tests._week5_fixtures import make_state


ARCHITECTURE = {
    "state_dim": 32,
    "probe_token_dim": 16,
    "action_embedding_dim": 8,
    "budget_embedding_dim": 8,
    "phi_hidden_dim": 32,
    "pooled_dim": 32,
    "dropout": 0.0,
    "max_budget": 7,
}


def _inputs() -> tuple[dict, dict]:
    vector = vectorize_state(make_state(acquired=("blank", "blur", "crop")), max_budget=4)
    first = collate_vectorized_states([vector])["model_input"]
    second = {key: value.clone() for key, value in first.items()}
    second["sequence_indices"][0, :3] = second["sequence_indices"][0, :3].flip(0)
    return first, second


def test_invariant_encoders_ignore_arrival_order() -> None:
    first, second = _inputs()
    for name in ("deep_sets", "masked_slot_mlp"):
        model = build_diagnostic_model(name, ARCHITECTURE).eval()
        with torch.no_grad():
            left = model(first)
            right = model(second)
        assert torch.allclose(left.hidden, right.hidden, atol=1e-7, rtol=0)
        assert torch.allclose(left.six_way_logits, right.six_way_logits, atol=1e-7, rtol=0)


def test_gru_is_order_sensitive_and_clean_control_is_not() -> None:
    torch.manual_seed(7)
    first, second = _inputs()
    gru = build_diagnostic_model("gru", ARCHITECTURE).eval()
    clean = build_diagnostic_model("clean_mlp", ARCHITECTURE).eval()
    with torch.no_grad():
        assert not torch.allclose(gru(first).hidden, gru(second).hidden, atol=1e-8, rtol=0)
        assert torch.equal(clean(first).hidden, clean(second).hidden)


def test_all_encoders_share_output_schema() -> None:
    first, _ = _inputs()
    for name in ("clean_mlp", "gru", "masked_slot_mlp", "deep_sets"):
        output = build_diagnostic_model(name, ARCHITECTURE).eval()(first)
        assert output.hidden.shape == (1, 32)
        assert output.bit_logits.shape == (1, 3)
        assert output.six_way_logits.shape == (1, 6)
        assert output.signature.shape == (1, 3)
