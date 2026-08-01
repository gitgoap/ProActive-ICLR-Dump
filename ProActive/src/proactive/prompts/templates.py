"""
Prompt templates and output parsing for probes and MLLM queries.

Each dataset type has a standard prompt. The grounding probe uses
a structured 'describe-then-answer' prompt requiring machine-readable
final answer syntax (FINAL_ANSWER: <answer>) (Plan §13.1, §14.5).
The relation probe re-asks the question with the swapped relation text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from proactive.features.normalization import normalize_answer


# ---------------------------------------------------------------------------
# Grounding parse result
# ---------------------------------------------------------------------------

@dataclass
class GroundingParsedResult:
    """Parsed output from a grounding (describe-then-answer) probe."""
    is_valid: bool
    raw_final_answer: str
    norm_final_answer: str
    description: str
    parse_status: str  # "ok", "regex_fallback", "malformed", "empty"
    invalid_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Standard dataset prompts
# ---------------------------------------------------------------------------

def make_binary_prompt(question: str, dataset: str) -> str:
    """Standard prompt for binary (yes/no or true/false) datasets.

    Args:
        question: The question or statement text.
        dataset: Dataset name for answer-format hint.
    """
    dataset_lower = dataset.lower()
    if dataset_lower in ("vsr", "gqa_relation"):
        return (
            f"Is the following statement true or false?\n"
            f"Statement: {question}\n"
            f"Answer with exactly one word: true or false."
        )
    else:
        # POPE, HallusionBench, PRE-HAL, IllusionBench
        return (
            f"{question}\n"
            f"Answer with exactly one word: yes or no."
        )


def make_freeform_prompt(question: str) -> str:
    """Standard prompt for free-form VQA datasets (VizWiz)."""
    return (
        f"{question}\n"
        f"Answer the question concisely."
    )


def make_dataset_prompt(
    question: str,
    dataset: str,
) -> str:
    """Dispatch to the correct prompt template based on dataset.

    Args:
        question: The question or statement text.
        dataset: Dataset name (e.g., 'pope', 'vizwiz', 'vsr').
    """
    dataset_lower = dataset.lower().replace("-", "").replace(" ", "_")
    if dataset_lower in ("vizwiz", "vizwiz_vqa"):
        return make_freeform_prompt(question)
    else:
        return make_binary_prompt(question, dataset)


# ---------------------------------------------------------------------------
# Grounding probe prompt & parser (Plan §13.1, §14.5)
# ---------------------------------------------------------------------------

def make_grounding_prompt(question: str, dataset: str) -> str:
    """Grounding probe: 'Describe what you see, then answer the question.'

    Forces visual grounding followed by machine-readable FINAL_ANSWER: tag.
    """
    dataset_lower = dataset.lower().replace("-", "").replace(" ", "_")
    if dataset_lower in ("vsr", "gqa_relation"):
        return (
            f"First, describe what you see in the image in 1-2 sentences.\n"
            f"Then, determine if the following statement is true or false.\n"
            f"Statement: {question}\n"
            f"At the very end of your response, format your final answer strictly as:\n"
            f"FINAL_ANSWER: <true or false>"
        )
    elif dataset_lower in ("vizwiz", "vizwiz_vqa"):
        return (
            f"First, describe what you see in the image in 1-2 sentences.\n"
            f"Then, answer the following question concisely.\n"
            f"Question: {question}\n"
            f"At the very end of your response, format your final answer strictly as:\n"
            f"FINAL_ANSWER: <answer>"
        )
    else:
        return (
            f"First, describe what you see in the image in 1-2 sentences.\n"
            f"Then, answer the following question.\n"
            f"Question: {question}\n"
            f"At the very end of your response, format your final answer strictly as:\n"
            f"FINAL_ANSWER: <yes or no>"
        )


def parse_grounding_output(raw_text: str, dataset: str) -> GroundingParsedResult:
    """Parse grounding probe output into description and normalized final answer.

    Enforces fail-closed validation: malformed answers are marked is_valid=False.
    """
    if not raw_text or not raw_text.strip():
        return GroundingParsedResult(
            is_valid=False,
            raw_final_answer="",
            norm_final_answer="",
            description="",
            parse_status="empty",
            invalid_reason="Empty generation output",
        )

    clean_text = raw_text.strip()
    dataset_lower = dataset.lower().replace("-", "").replace(" ", "_")
    is_binary = dataset_lower in ("pope", "vsr", "hallusionbench", "gqa_relation", "prehal", "illusionbench")

    # Match FINAL_ANSWER: <content>
    pattern = r"(?:FINAL_ANSWER|FINAL ANSWER|ANSWER)\s*:\s*(.*)$"
    match = re.search(pattern, clean_text, re.IGNORECASE | re.DOTALL)

    if match:
        desc_part = clean_text[:match.start()].strip()
        ans_part = match.group(1).strip()
        # Clean answer part (take first line or first token if binary)
        first_line = ans_part.split("\n")[0].strip()
        norm_ans = normalize_answer(first_line, dataset)

        if is_binary:
            if norm_ans in ("yes", "no", "true", "false"):
                return GroundingParsedResult(
                    is_valid=True,
                    raw_final_answer=first_line,
                    norm_final_answer=norm_ans,
                    description=desc_part,
                    parse_status="ok",
                )
            else:
                return GroundingParsedResult(
                    is_valid=False,
                    raw_final_answer=first_line,
                    norm_final_answer=norm_ans,
                    description=desc_part,
                    parse_status="malformed",
                    invalid_reason=f"Binary dataset answer not in valid domain: '{first_line}' -> '{norm_ans}'",
                )
        else:
            # Free-form
            if norm_ans and norm_ans != "unknown":
                return GroundingParsedResult(
                    is_valid=True,
                    raw_final_answer=first_line,
                    norm_final_answer=norm_ans,
                    description=desc_part,
                    parse_status="ok",
                )
            else:
                return GroundingParsedResult(
                    is_valid=False,
                    raw_final_answer=first_line,
                    norm_final_answer=norm_ans,
                    description=desc_part,
                    parse_status="malformed",
                    invalid_reason="Empty or unanswerable free-form output",
                )

    # Fallback search if tag was omitted: look at the last line
    lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
    if lines:
        last_line = lines[-1]
        norm_ans = normalize_answer(last_line, dataset)
        if is_binary and norm_ans in ("yes", "no", "true", "false"):
            desc_part = "\n".join(lines[:-1]).strip()
            return GroundingParsedResult(
                is_valid=True,
                raw_final_answer=last_line,
                norm_final_answer=norm_ans,
                description=desc_part,
                parse_status="regex_fallback",
            )

    return GroundingParsedResult(
        is_valid=False,
        raw_final_answer="",
        norm_final_answer="",
        description=clean_text,
        parse_status="malformed",
        invalid_reason="Missing FINAL_ANSWER tag and unable to resolve valid final answer",
    )


# ---------------------------------------------------------------------------
# Relation probe prompt  (Plan §13.1)
# ---------------------------------------------------------------------------

def make_relation_prompt(
    swapped_question: str,
    dataset: str,
) -> str:
    """Relation probe: re-ask with the swapped relation text."""
    return make_dataset_prompt(swapped_question, dataset)
