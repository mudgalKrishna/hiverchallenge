"""
evaluator.py
============

LLM-as-a-Judge evaluator for the generated suggested replies.

For each (incoming_email, generated_reply, ground_truth_reply) triple, this
module builds an evaluation prompt that explicitly injects the dataset's
`gold_requirements`, `must_include`, and `must_not_include` fields, asks the
judge LLM to score the reply, and parses a structured result:

    {
        "intent_fulfillment_score": 1-5,
        "intent_fulfillment_reasoning": "...",
        "constraint_adherence_score": 1-5,
        "constraint_adherence_reasoning": "...",
        "missing_must_include": [...],       # which must_include items were absent
        "present_must_not_include": [...],   # which must_not_include items leaked in
        "overall_score": average of the two scores,
    }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from llm_provider import LLMProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are a meticulous, impartial QA evaluator for a customer support "
    "AI system at a company called CloudSync. You grade AI-generated email "
    "replies against strict rubrics. You always respond with valid JSON and "
    "nothing else -- no markdown code fences, no commentary outside the JSON."
)

JUDGE_PROMPT_TEMPLATE = """\
Evaluate the AI-GENERATED REPLY below against the rubric and reference material.

CUSTOMER'S INCOMING EMAIL:
{incoming_email}

GROUND-TRUTH (HUMAN AGENT) REPLY, for reference only:
{ground_truth_reply}

GOLD REQUIREMENTS (what a correct reply must accomplish):
{gold_requirements}

MUST INCLUDE (facts/phrases/actions that MUST appear in the reply):
{must_include}

MUST NOT INCLUDE (facts/phrases/actions that must NOT appear in the reply):
{must_not_include}

AI-GENERATED REPLY TO EVALUATE:
{generated_reply}

Score the AI-GENERATED REPLY on a 1-5 integer scale (5 = best) for each of the
following two dimensions:

1. intent_fulfillment_score: Does the reply resolve the customer's problem as
   effectively as the ground-truth reply, and satisfy the gold requirements?
2. constraint_adherence_score: Does the reply include everything in
   MUST INCLUDE and avoid everything in MUST NOT INCLUDE?

Respond with ONLY a single JSON object with exactly this schema (no extra text):
{{
  "intent_fulfillment_score": <integer 1-5>,
  "intent_fulfillment_reasoning": "<one or two sentences>",
  "constraint_adherence_score": <integer 1-5>,
  "constraint_adherence_reasoning": "<one or two sentences>",
  "missing_must_include": ["<any MUST INCLUDE items that are absent from the reply>"],
  "present_must_not_include": ["<any MUST NOT INCLUDE items that leaked into the reply>"]
}}
"""


def _format_list_field(items: List[str]) -> str:
    if not items:
        return "(none specified)"
    return "\n".join(f"- {item}" for item in items)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from an LLM response, tolerating stray
    markdown code fences or extra commentary the model may have added
    despite instructions.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first {...} block via brace matching.
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in judge response: {text[:300]!r}")

    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                return json.loads(candidate)

    raise ValueError(f"Could not parse a complete JSON object from judge response: {text[:300]!r}")


@dataclass
class JudgeResult:
    intent_fulfillment_score: int
    intent_fulfillment_reasoning: str
    constraint_adherence_score: int
    constraint_adherence_reasoning: str
    missing_must_include: List[str]
    present_must_not_include: List[str]

    @property
    def overall_score(self) -> float:
        return (self.intent_fulfillment_score + self.constraint_adherence_score) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_fulfillment_score": self.intent_fulfillment_score,
            "intent_fulfillment_reasoning": self.intent_fulfillment_reasoning,
            "constraint_adherence_score": self.constraint_adherence_score,
            "constraint_adherence_reasoning": self.constraint_adherence_reasoning,
            "missing_must_include": self.missing_must_include,
            "present_must_not_include": self.present_must_not_include,
            "overall_score": self.overall_score,
        }


class LLMJudge:
    """LLM-as-a-Judge evaluator, driven by any LLMProvider implementation."""

    def __init__(self, llm_provider: LLMProvider, max_parse_retries: int = 2):
        self.llm_provider = llm_provider
        self.max_parse_retries = max_parse_retries

    def _clamp_score(self, value: Any, default: int = 1) -> int:
        try:
            score = int(round(float(value)))
        except (TypeError, ValueError):
            logger.warning("Judge returned a non-numeric score (%r); defaulting to %d", value, default)
            return default
        return max(1, min(5, score))

    def evaluate(
        self,
        incoming_email: str,
        generated_reply: str,
        ground_truth_reply: str,
        gold_requirements: List[str],
        must_include: List[str],
        must_not_include: List[str],
    ) -> JudgeResult:
        """Run the judge on a single (email, generated_reply) pair."""
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            incoming_email=incoming_email,
            ground_truth_reply=ground_truth_reply,
            gold_requirements=_format_list_field(gold_requirements),
            must_include=_format_list_field(must_include),
            must_not_include=_format_list_field(must_not_include),
            generated_reply=generated_reply or "(the system failed to generate a reply)",
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_parse_retries + 2):
            try:
                raw_response = self.llm_provider.generate(
                    prompt=prompt,
                    system=JUDGE_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=500,
                )
                parsed = _extract_json_object(raw_response)
                return JudgeResult(
                    intent_fulfillment_score=self._clamp_score(parsed.get("intent_fulfillment_score")),
                    intent_fulfillment_reasoning=str(parsed.get("intent_fulfillment_reasoning", "")).strip(),
                    constraint_adherence_score=self._clamp_score(parsed.get("constraint_adherence_score")),
                    constraint_adherence_reasoning=str(parsed.get("constraint_adherence_reasoning", "")).strip(),
                    missing_must_include=list(parsed.get("missing_must_include", []) or []),
                    present_must_not_include=list(parsed.get("present_must_not_include", []) or []),
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Judge parse attempt %d/%d failed: %s",
                    attempt, self.max_parse_retries + 1, exc,
                )

        logger.error("Judge evaluation failed after retries: %s. Returning a default low score.", last_error)
        return JudgeResult(
            intent_fulfillment_score=1,
            intent_fulfillment_reasoning=f"Evaluation failed: {last_error}",
            constraint_adherence_score=1,
            constraint_adherence_reasoning=f"Evaluation failed: {last_error}",
            missing_must_include=must_include,
            present_must_not_include=[],
        )

    def evaluate_dataframe(self, generated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the judge over every row of a DataFrame that already contains a
        'generated_reply' column (i.e. the output of generator.py), plus the
        original gold_requirements / must_include / must_not_include columns.
        """
        required_cols = {
            "incoming_email", "generated_reply", "ground_truth_reply",
            "gold_requirements", "must_include", "must_not_include",
        }
        missing = required_cols - set(generated_df.columns)
        if missing:
            raise ValueError(f"generated_df is missing required columns: {missing}")

        judged_rows = []
        for i, row in generated_df.reset_index(drop=True).iterrows():
            logger.info("Judging response %d/%d (id=%s)...", i + 1, len(generated_df), row.get("id", i))
            result = self.evaluate(
                incoming_email=row["incoming_email"],
                generated_reply=row["generated_reply"],
                ground_truth_reply=row["ground_truth_reply"],
                gold_requirements=row["gold_requirements"],
                must_include=row["must_include"],
                must_not_include=row["must_not_include"],
            )
            merged = row.to_dict()
            merged.update(result.to_dict())
            judged_rows.append(merged)

        return pd.DataFrame(judged_rows)
