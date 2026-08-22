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
from proactive.data.hallusion_contract import HALLUSION_OPEN_ENDED


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
    # "ok", "regex_fallback", "terminal_answer_fallback", "malformed", "empty"
    parse_status: str
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
    answer_type: Optional[str] = None,
) -> str:
    """Dispatch to the correct prompt template based on dataset.

    Args:
        question: The question or statement text.
        dataset: Dataset name (e.g., 'pope', 'vizwiz', 'vsr').
    """
    dataset_lower = dataset.lower().replace("-", "").replace(" ", "_")
    if answer_type == HALLUSION_OPEN_ENDED or dataset_lower in ("vizwiz", "vizwiz_vqa"):
        return make_freeform_prompt(question)
    else:
        return make_binary_prompt(question, dataset)


# ---------------------------------------------------------------------------
# Grounding probe prompt & parser (Plan §13.1, §14.5)
# ---------------------------------------------------------------------------

def make_grounding_prompt(
    question: str, dataset: str, answer_type: Optional[str] = None
) -> str:
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
    elif answer_type == HALLUSION_OPEN_ENDED or dataset_lower in ("vizwiz", "vizwiz_vqa"):
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


_EXPLICIT_TERMINAL_ANSWER_RE = re.compile(
    r"^(?:[-*]\s*)?(?:\*\*)?"
    r"(?:(?:the|my)\s+)?(?:final\s+)?answer(?:\*\*)?\s+"
    r"(?:is|would\s+be)\s+(?P<answer>.+?)\s*$",
    re.IGNORECASE,
)
_EMBEDDED_BINARY_ANSWER_RE = re.compile(
    r"\b(?:the\s+)?(?:final\s+)?answer\s+(?:is|would\s+be)\s+"
    r"(?P<answer>yes|no|true|false|uncertain)\b",
    re.IGNORECASE,
)


def _strip_answer_markup(value: str) -> str:
    """Remove only surrounding presentation markup, never answer content."""
    answer = value.strip()
    while len(answer) >= 2:
        pairs = (("**", "**"), ("__", "__"), ("`", "`"), ('"', '"'), ("'", "'"))
        stripped = False
        for left, right in pairs:
            if answer.startswith(left) and answer.endswith(right):
                answer = answer[len(left) : len(answer) - len(right)].strip()
                stripped = True
                break
        if not stripped:
            break
    return answer


def _binary_candidate_is_unambiguous(
    candidate: str,
    dataset: str,
    norm_answer: str,
    normalizer_type: Optional[str] = None,
) -> bool:
    """Reject explicit answer phrases containing conflicting binary indicators."""
    outcomes = {
        normalize_answer(token, dataset, normalizer_type=normalizer_type)
        for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", candidate)
    }
    outcomes.intersection_update({"yes", "no", "true", "false", "uncertain"})
    return bool(outcomes) and outcomes == {norm_answer}


def parse_grounding_output(
    raw_text: str,
    dataset: str,
    answer_type: Optional[str] = None,
) -> GroundingParsedResult:
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
    normalizer_type = "freeform" if answer_type == HALLUSION_OPEN_ENDED else None
    is_binary = (
        answer_type != HALLUSION_OPEN_ENDED
        and dataset_lower
        in ("pope", "vsr", "hallusionbench", "gqa_relation", "prehal", "illusionbench")
    )
    binary_domain = {"yes", "no", "true", "false"}
    if dataset_lower == "hallusionbench":
        binary_domain.add("uncertain")

    # Match FINAL_ANSWER: <content>
    pattern = r"(?:FINAL_ANSWER|FINAL ANSWER|ANSWER)\s*:\s*(.*)$"
    match = re.search(pattern, clean_text, re.IGNORECASE | re.DOTALL)

    if match:
        desc_part = clean_text[:match.start()].strip()
        ans_part = match.group(1).strip()
        # Clean answer part (take first line or first token if binary)
        first_line = ans_part.split("\n")[0].strip()
        if not first_line:
            return GroundingParsedResult(
                is_valid=False,
                raw_final_answer="",
                norm_final_answer="",
                description=desc_part,
                parse_status="malformed",
                invalid_reason="FINAL_ANSWER tag has no answer content",
            )
        norm_ans = normalize_answer(
            first_line, dataset, normalizer_type=normalizer_type
        )

        if is_binary:
            if norm_ans in binary_domain:
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
            # VizWiz uses ``unanswerable`` as a legitimate answer class.  An
            # explicit structured abstention of exactly ``Unknown`` carries
            # that behavior; it is not a missing parser value.
            if norm_ans == "unknown" and first_line.lower().rstrip(".") == "unknown":
                norm_ans = "unanswerable"
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
                    invalid_reason="Empty or unknown free-form output",
                )

    # Fallback search if tag was omitted: look at the last line
    lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
    if lines:
        last_line = lines[-1]
        norm_ans = normalize_answer(
            last_line, dataset, normalizer_type=normalizer_type
        )
        if is_binary and norm_ans in binary_domain:
            desc_part = "\n".join(lines[:-1]).strip()
            return GroundingParsedResult(
                is_valid=True,
                raw_final_answer=last_line,
                norm_final_answer=norm_ans,
                description=desc_part,
                parse_status="regex_fallback",
            )

        # Some deterministic model outputs follow the requested reasoning but
        # omit the literal tag and finish with "The answer is ...".  Recover
        # only that terminal, explicitly answer-bearing construction.  A bare
        # free-form last line is intentionally not accepted because it cannot
        # be distinguished from an image description.
        terminal_match = _EXPLICIT_TERMINAL_ANSWER_RE.fullmatch(last_line)
        if terminal_match:
            candidate = _strip_answer_markup(terminal_match.group("answer"))
            if candidate:
                norm_candidate = normalize_answer(
                    candidate, dataset, normalizer_type=normalizer_type
                )
                binary_valid = (
                    norm_candidate in binary_domain
                    and _binary_candidate_is_unambiguous(
                        candidate,
                        dataset,
                        norm_candidate,
                        normalizer_type=normalizer_type,
                    )
                )
                freeform_valid = not is_binary and norm_candidate != "unknown"
                if binary_valid or freeform_valid:
                    return GroundingParsedResult(
                        is_valid=True,
                        raw_final_answer=candidate,
                        norm_final_answer=norm_candidate,
                        description="\n".join(lines[:-1]).strip(),
                        parse_status="terminal_answer_fallback",
                    )

        if is_binary and terminal_match is None:
            embedded_matches = list(_EMBEDDED_BINARY_ANSWER_RE.finditer(last_line))
            embedded_answers = {
                normalize_answer(
                    match.group("answer"),
                    dataset,
                    normalizer_type=normalizer_type,
                )
                for match in embedded_matches
            }
            if len(embedded_answers) == 1:
                norm_candidate = embedded_answers.pop()
                if norm_candidate in binary_domain:
                    raw_candidate = embedded_matches[-1].group("answer")
                    return GroundingParsedResult(
                        is_valid=True,
                        raw_final_answer=raw_candidate,
                        norm_final_answer=norm_candidate,
                        description="\n".join(lines[:-1]).strip(),
                        parse_status="terminal_answer_fallback",
                    )

        # A short isolated final paragraph is an answer-bearing construction
        # for free-form VQA (for example ``MVG`` or ``$1.00``).  Keep the gate
        # deliberately narrow so a prose description is never reinterpreted.
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", clean_text) if part.strip()]
        if (
            not is_binary
            and len(paragraphs) >= 2
            and paragraphs[-1] == last_line
        ):
            candidate = _strip_answer_markup(last_line)
            candidate_words = candidate.split()
            if (
                candidate
                and len(candidate) <= 80
                and len(candidate_words) <= 8
                and not candidate.endswith(("?", ":", ";", ","))
            ):
                norm_candidate = normalize_answer(
                    candidate, dataset, normalizer_type=normalizer_type
                )
                if norm_candidate != "unknown":
                    return GroundingParsedResult(
                        is_valid=True,
                        raw_final_answer=candidate,
                        norm_final_answer=norm_candidate,
                        description="\n".join(lines[:-1]).strip(),
                        parse_status="terminal_answer_fallback",
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
    answer_type: Optional[str] = None,
) -> str:
    """Relation probe: re-ask with the swapped relation text."""
    return make_dataset_prompt(swapped_question, dataset, answer_type=answer_type)
