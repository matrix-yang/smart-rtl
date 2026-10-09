"""Immutable state records for the RTL search tree."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional


class NodeKind(str, Enum):
    ROOT = "root"
    PLAN = "plan"
    CANDIDATE = "candidate"


class NodeStatus(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    PRUNED = "pruned"
    PASSED = "passed"
    BACKTRACKED = "backtracked"


class ActionKind(str, Enum):
    EXPAND = "expand"
    REFINE = "refine"
    BACKTRACK = "backtrack"
    STOP = "stop"


class EDAStage(IntEnum):
    """Hard evaluator ordering.

    Infrastructure failure is deliberately outside normal candidate progress.
    A policy must not interpret it as evidence that the RTL is worse.
    """

    INFRASTRUCTURE = -1
    INCOMPLETE = 0
    COMPILE = 10
    ELABORATE = 20
    SIMULATION = 30
    PASS = 40


@dataclass(frozen=True, slots=True)
class EDAEvidence:
    stage: EDAStage
    original_stage: str
    failure_class: str
    failure_signature: str
    complete_bundle: bool
    passed_subtests: int = 0
    total_subtests: int = 0
    mismatch_count: Optional[int] = None
    first_mismatch_cycle: Optional[int] = None
    public_feedback: str = ""

    def __post_init__(self) -> None:
        if self.passed_subtests < 0 or self.total_subtests < 0:
            raise ValueError("subtest counts must be non-negative")
        if self.passed_subtests > self.total_subtests:
            raise ValueError("passed_subtests cannot exceed total_subtests")
        if self.stage is EDAStage.PASS and self.total_subtests:
            if self.passed_subtests != self.total_subtests:
                raise ValueError("PASS evidence must pass every recorded subtest")

    @property
    def passed(self) -> bool:
        return self.stage is EDAStage.PASS


@dataclass(frozen=True, slots=True)
class SearchNode:
    """An immutable plan or executable candidate state.

    Mutable search bookkeeping (active/pruned/backtracked and tried actions) is
    owned by :class:`SearchTree`, so node lineage and evidence cannot silently
    change after a decision has been recorded.
    """

    node_id: str
    parent_id: Optional[str]
    kind: NodeKind
    depth: int
    plan_id: Optional[str] = None
    hypothesis: Optional[str] = None
    candidate_hash: Optional[str] = None
    evidence: Optional[EDAEvidence] = None
    action_from_parent: Optional[ActionKind] = None
    source_iteration: Optional[int] = None
    source_ref: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if self.depth < 0:
            raise ValueError("depth must be non-negative")
        if self.kind is NodeKind.ROOT:
            if self.parent_id is not None or self.depth != 0:
                raise ValueError("root nodes cannot have a parent or non-zero depth")
        elif self.parent_id is None:
            raise ValueError("non-root nodes require a parent")
        if self.kind is NodeKind.CANDIDATE:
            if not self.candidate_hash:
                raise ValueError("candidate nodes require candidate_hash")
            if self.evidence is None:
                raise ValueError("candidate nodes require EDA evidence")
