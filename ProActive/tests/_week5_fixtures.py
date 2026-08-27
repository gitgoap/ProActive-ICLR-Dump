from __future__ import annotations

from typing import Iterable


PROBES = ("blank", "blur", "crop", "brightness", "noise", "grounding", "relation")


def make_state(
    *,
    acquired: Iterable[str] = ("blank", "blur"),
    split: str = "train",
    instance_id: str = "instance-1",
    model_id: str = "model/test",
    relation_applicable: bool = False,
) -> dict:
    names = list(acquired)
    if "relation" in names and not relation_applicable:
        raise ValueError("relation acquisition requires relation_applicable")
    observations = []
    for index, name in enumerate(names):
        observations.append(
            {
                "probe_id": name,
                "flip": index % 2,
                "conf_shift": 0.1 + index * 0.01,
                "entropy_shift": -0.2 + index * 0.01,
                "margin_shift": 0.3 + index * 0.01,
                "exact_match": float(index % 2 == 0),
                "semantic_match": 0.8 - index * 0.01,
                "applicable": 1,
            }
        )
    mask = {
        name: int(name not in names and (name != "relation" or relation_applicable))
        for name in PROBES
    }
    mask["stop"] = 1
    return {
        "record_type": "partial_state_v1",
        "state_id": f"state:{model_id}:{instance_id}:{'-'.join(names) or 'empty'}",
        "metadata": {
            "instance_id": instance_id,
            "group_id": f"group:{instance_id}",
            "dataset": "pope",
            "split": split,
            "model_id": model_id,
            "model_revision": "a" * 40,
        },
        "learner_input": {
            "clean_features": {
                "answer_prob": 0.8,
                "token_entropy_mean": 0.2,
                "token_margin_mean": 0.6,
                "answer_len_tokens": 2.0,
            },
            "acquired_probe_names": names,
            "acquired_observations": observations,
            "remaining_budget": 7 - len(names),
            "action_mask": mask,
        },
        "targets": {
            "clean_correct": 1,
            "teacher_signature": {"V": 0.2, "L": 0.3, "A": 0.4},
            "teacher_bits": {"visual": 1, "language": 0, "alignment": 1},
            "teacher_label6": "mixed",
        },
        "sampling": {"sources": ["test"]},
    }


def make_teacher(state: dict) -> dict:
    probes = {}
    for index, name in enumerate(PROBES):
        if name == "relation" and not state["learner_input"]["action_mask"]["relation"]:
            continue
        probes[name] = {
            "probe_id": name,
            "flip": bool(index % 2),
            "conf_shift": 0.2,
            "entropy_shift": -0.1,
            "margin_shift": 0.1,
            "exact_match": 1.0,
            "semantic_match": 0.9,
            "applicable": True,
            "valid": True,
        }
    return {"probes": probes}
