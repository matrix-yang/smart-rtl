"""Deterministic EDA ordering used before any LLM judgment."""

from __future__ import annotations

import re
from typing import Iterable, Tuple

from .node import EDAEvidence, EDAStage


STAGE_ALIASES = {
    "INFRA_SETUP": EDAStage.INFRASTRUCTURE,
    "INFRASTRUCTURE": EDAStage.INFRASTRUCTURE,
    "LOCAL_PREFLIGHT": EDAStage.INCOMPLETE,
    "INCOMPLETE": EDAStage.INCOMPLETE,
    "CANDIDATE_COMPILE": EDAStage.COMPILE,
    "COMPILE": EDAStage.COMPILE,
    "ELABORATE": EDAStage.ELABORATE,
    "ELABORATION": EDAStage.ELABORATE,
    "CANDIDATE_FUNCTIONAL": EDAStage.SIMULATION,
    "SIMULATION": EDAStage.SIMULATION,
    "PASS": EDAStage.PASS,
}


def parse_eda_stage(raw_stage: str) -> EDAStage:
    key = str(raw_stage or "").strip().upper()
    try:
        return STAGE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unknown EDA stage: {raw_stage!r}") from exc


def normalize_failure_signature(text: str) -> str:
    signature = str(text or "").strip()
    signature = re.sub(r"/workspace/[^\s:]+", "<workspace>", signature)
    signature = re.sub(r":\d+:", ":N:", signature)
    signature = re.sub(r"\b\d+(?:\.\d+)?ns\b", "<time>", signature, flags=re.IGNORECASE)
    signature = re.sub(r"\s+", " ", signature)
    return signature[:240]


def evidence_rank(evidence: EDAEvidence) -> Tuple[int, int, int, int, int]:
    """Return an ordinal rank whose first element is always the hard EDA stage."""

    mismatch_rank = -evidence.mismatch_count if evidence.mismatch_count is not None else -10**9
    cycle_rank = evidence.first_mismatch_cycle if evidence.first_mismatch_cycle is not None else -1
    return (
        int(evidence.stage),
        evidence.passed_subtests,
        -max(0, evidence.total_subtests - evidence.passed_subtests),
        mismatch_rank,
        cycle_rank,
    )


def compare_evidence(left: EDAEvidence, right: EDAEvidence) -> int:
    left_rank = evidence_rank(left)
    right_rank = evidence_rank(right)
    return (left_rank > right_rank) - (left_rank < right_rank)


def is_structural_regression(parent: EDAEvidence, child: EDAEvidence) -> bool:
    return parent.stage >= EDAStage.SIMULATION and child.stage < EDAStage.SIMULATION


def has_subtest_regression(parent: EDAEvidence, child: EDAEvidence) -> bool:
    if not parent.total_subtests or not child.total_subtests:
        return False
    return child.passed_subtests < parent.passed_subtests


def is_abab_oscillation(signatures: Iterable[str]) -> bool:
    values = [normalize_failure_signature(value) for value in signatures]
    if len(values) < 4:
        return False
    a, b, c, d = values[-4:]
    return bool(a and b and a != b and a == c and b == d)
