"""Structured contracts for the skill-defined adaptive BW-RTL loop.

These types describe model decisions.  They intentionally contain no scoring
thresholds or fixed-width selection policy; deterministic code validates only
referential integrity and legal state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .contracts import CandidateArtifact, PlanHypothesis
from .node import EDAEvidence


class AdaptiveActionKind(str, Enum):
    SEED_PLAN = "seed_plan"
    REFINE = "refine"
    FORK = "fork"
    REVIEW_SPEC = "review_spec"
    REPLAN = "replan"
    PROBE = "probe"
    DISCUSS = "discuss"
    FUSE = "fuse"
    BACKTRACK = "backtrack"
    PARK = "park"
    PRUNE = "prune"
    STOP = "stop"

    @property
    def produces_candidate(self) -> bool:
        return self in {
            AdaptiveActionKind.SEED_PLAN,
            AdaptiveActionKind.REFINE,
            AdaptiveActionKind.FORK,
            AdaptiveActionKind.FUSE,
        }

    @property
    def invokes_planner(self) -> bool:
        return self in {AdaptiveActionKind.REVIEW_SPEC, AdaptiveActionKind.REPLAN}


class BranchStatus(str, Enum):
    ACTIVE = "active"
    PARKED = "parked"
    PRUNED = "pruned"
    BACKTRACKED = "backtracked"
    PASSED = "passed"
    FAILED_ACTION = "failed_action"


@dataclass(frozen=True, slots=True)
class ActionCard:
    """One Director-selected semantic action.

    ``source_node_ids`` may name candidates or, for backtrack, a source and a
    destination.  ``plan_id`` names the plan used by candidate-producing work.
    Empty hypothesis fields are rejected so that concurrency never degenerates
    into unlabelled duplicate sampling.
    """

    action_id: str
    kind: AdaptiveActionKind
    source_node_ids: Tuple[str, ...] = ()
    plan_id: Optional[str] = None
    hypothesis: str = ""
    obligation: str = ""
    semantic_delta: str = ""
    expected_observation: str = ""
    falsifier: str = ""
    evidence_refs: Tuple[str, ...] = ()
    knowledge_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise ValueError("action_id must be non-empty")
        if self.kind is not AdaptiveActionKind.STOP:
            required = {
                "hypothesis": self.hypothesis,
                "obligation": self.obligation,
                "expected_observation": self.expected_observation,
                "falsifier": self.falsifier,
            }
            missing = [name for name, value in required.items() if not value.strip()]
            if missing:
                raise ValueError(
                    f"action {self.action_id} omitted decision fields: {', '.join(missing)}"
                )
        if self.kind in {
            AdaptiveActionKind.REFINE,
            AdaptiveActionKind.FORK,
            AdaptiveActionKind.FUSE,
        } and not self.semantic_delta.strip():
            raise ValueError(
                f"action {self.action_id} must state a semantic_delta from its source state"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "source_node_ids": list(self.source_node_ids),
            "plan_id": self.plan_id,
            "hypothesis": self.hypothesis,
            "obligation": self.obligation,
            "semantic_delta": self.semantic_delta,
            "expected_observation": self.expected_observation,
            "falsifier": self.falsifier,
            "evidence_refs": list(self.evidence_refs),
            "knowledge_refs": list(self.knowledge_refs),
        }


@dataclass(frozen=True, slots=True)
class TaskAssessment:
    spec_certainty: str
    plan_concentration: str
    implementation_risk: str
    verification_observability: str
    rationale: str

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.spec_certainty,
                self.plan_concentration,
                self.implementation_risk,
                self.verification_observability,
                self.rationale,
            )
        ):
            raise ValueError("task assessment fields must be non-empty")

    def to_dict(self) -> Dict[str, str]:
        return {
            "spec_certainty": self.spec_certainty,
            "plan_concentration": self.plan_concentration,
            "implementation_risk": self.implementation_risk,
            "verification_observability": self.verification_observability,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeProposal:
    proposal_id: str
    item_type: str
    scope: str
    scope_ref: Optional[str]
    claim: str
    evidence_refs: Tuple[str, ...]
    confidence: str
    conflicts_with: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.proposal_id.strip() or not self.claim.strip():
            raise ValueError("knowledge proposals require an id and claim")
        if not self.evidence_refs:
            raise ValueError("knowledge proposals require at least one evidence reference")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "item_type": self.item_type,
            "scope": self.scope,
            "scope_ref": self.scope_ref,
            "claim": self.claim,
            "evidence_refs": list(self.evidence_refs),
            "confidence": self.confidence,
            "conflicts_with": list(self.conflicts_with),
        }


@dataclass(frozen=True, slots=True)
class BranchReview:
    candidate_node_id: str
    evidence_id: str
    strengths: Tuple[str, ...]
    weaknesses: Tuple[str, ...]
    observations: Tuple[str, ...]
    root_causes: Tuple[str, ...]
    invalidated_assumptions: Tuple[str, ...]
    open_questions: Tuple[str, ...]
    recommended_actions: Tuple[str, ...]
    knowledge_proposals: Tuple[KnowledgeProposal, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_node_id": self.candidate_node_id,
            "evidence_id": self.evidence_id,
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
            "observations": list(self.observations),
            "root_causes": list(self.root_causes),
            "invalidated_assumptions": list(self.invalidated_assumptions),
            "open_questions": list(self.open_questions),
            "recommended_actions": list(self.recommended_actions),
            "knowledge_proposals": [item.to_dict() for item in self.knowledge_proposals],
        }


@dataclass(frozen=True, slots=True)
class KnowledgeDecision:
    item_id: str
    status: str
    rationale: str

    def __post_init__(self) -> None:
        if not self.item_id.strip() or not self.rationale.strip():
            raise ValueError("knowledge decisions require item_id and rationale")


@dataclass(frozen=True, slots=True)
class DirectorDecision:
    assessment: TaskAssessment
    rationale: str
    dispatch: Tuple[ActionCard, ...]
    knowledge_decisions: Tuple[KnowledgeDecision, ...] = ()

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("Director rationale must be non-empty")
        action_ids = [action.action_id for action in self.dispatch]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("Director action_id values must be unique within a round")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment": self.assessment.to_dict(),
            "rationale": self.rationale,
            "dispatch": [action.to_dict() for action in self.dispatch],
            "knowledge_decisions": [
                {
                    "item_id": item.item_id,
                    "status": item.status,
                    "rationale": item.rationale,
                }
                for item in self.knowledge_decisions
            ],
        }


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    node_id: str
    plan_id: str
    parent_node_ids: Tuple[str, ...]
    producing_action_id: str
    artifact: CandidateArtifact
    status: BranchStatus
    round_created: int
    evidence_id: Optional[str] = None
    review: Optional[BranchReview] = None
    error: Optional[str] = None
    candidate_judgement: Optional["CandidateJudgement"] = None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "plan_id": self.plan_id,
            "parent_node_ids": list(self.parent_node_ids),
            "producing_action_id": self.producing_action_id,
            "candidate_hash": self.artifact.candidate_hash,
            "status": self.status.value,
            "round_created": self.round_created,
            "evidence_id": self.evidence_id,
            "review": self.review.to_dict() if self.review else None,
            "error": self.error,
            "candidate_judgement": (
                self.candidate_judgement.to_dict()
                if self.candidate_judgement is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    candidate_node_id: str
    evidence: EDAEvidence

    def to_dict(self) -> Dict[str, Any]:
        feedback = self.evidence.public_feedback
        if len(feedback) > 6000:
            feedback = feedback[:3000] + "\n... [BOUNDED] ...\n" + feedback[-2500:]
        return {
            "evidence_id": self.evidence_id,
            "candidate_node_id": self.candidate_node_id,
            "stage": self.evidence.stage.name,
            "original_stage": self.evidence.original_stage,
            "failure_class": self.evidence.failure_class,
            "failure_signature": self.evidence.failure_signature,
            "complete_bundle": self.evidence.complete_bundle,
            "passed_subtests": self.evidence.passed_subtests,
            "total_subtests": self.evidence.total_subtests,
            "public_feedback": feedback,
        }


@dataclass(frozen=True, slots=True)
class RoundEvent:
    action_id: str
    kind: AdaptiveActionKind
    status: str
    produced_ids: Tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "status": self.status,
            "produced_ids": list(self.produced_ids),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class SearchSnapshot:
    round_index: int
    plans: Tuple[PlanHypothesis, ...]
    candidates: Tuple[CandidateRecord, ...]
    evidence: Tuple[EvidenceRecord, ...]
    knowledge: Tuple[Mapping[str, Any], ...]
    prior_round_events: Tuple[RoundEvent, ...]
    physical_worker_limit: int
    available_actions: Tuple[str, ...]
    max_rounds: Optional[int] = None
    resource_budget: Mapping[str, int] = field(default_factory=dict)
    resource_usage: Mapping[str, int] = field(default_factory=dict)
    resource_remaining: Mapping[str, int] = field(default_factory=dict)
    resource_cost_history: Mapping[str, Any] = field(default_factory=dict)
    lineage_diagnostics: Tuple[Mapping[str, Any], ...] = ()

    def to_prompt_dict(self) -> Dict[str, Any]:
        rounds_remaining = (
            max(self.max_rounds - self.round_index + 1, 0)
            if self.max_rounds is not None
            else None
        )
        return {
            "round_index": self.round_index,
            "max_rounds": self.max_rounds,
            "rounds_remaining_including_current": rounds_remaining,
            "plans": [
                {
                    "plan_id": plan.plan_id,
                    "summary": plan.summary,
                    "distinguishing_assumption": plan.distinguishing_assumption,
                    "invariants": list(plan.invariants),
                }
                for plan in self.plans
            ],
            "candidates": [candidate.to_summary() for candidate in self.candidates],
            "evidence": [item.to_dict() for item in self.evidence],
            "knowledge": [dict(item) for item in self.knowledge],
            "prior_round_events": [event.to_dict() for event in self.prior_round_events],
            "physical_worker_limit": self.physical_worker_limit,
            "available_actions": list(self.available_actions),
            "resource_budget": dict(self.resource_budget),
            "resource_usage": dict(self.resource_usage),
            "resource_remaining": dict(self.resource_remaining),
            "resource_cost_history": dict(self.resource_cost_history),
            "lineage_diagnostics": [dict(item) for item in self.lineage_diagnostics],
            "note": (
                "physical_worker_limit affects execution latency only; it is not a target "
                "dispatch width or branch-selection constraint. max_rounds is a safety "
                "horizon, not a target search depth. Resource remaining is a hard runtime "
                "feasibility constraint, not a branch-count hyperparameter. Lineage "
                "diagnostics report facts; the Director owns the semantic response."
            ),
        }


@dataclass(frozen=True, slots=True)
class BranchExecution:
    action: ActionCard
    node_id: str
    evidence_id: str
    artifact: Optional[CandidateArtifact]
    evidence: Optional[EDAEvidence]
    review: Optional[BranchReview]
    error: Optional[str] = None
    note: str = ""
    candidate_judgement: Optional["CandidateJudgement"] = None


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    summary: str
    evidence_refs: Tuple[str, ...]
    knowledge_proposals: Tuple[KnowledgeProposal, ...] = ()

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("probe outcome summary must be non-empty")


@dataclass(frozen=True, slots=True)
class DebateOutcome:
    summary: str
    branch_advantages: Tuple[str, ...]
    branch_risks: Tuple[str, ...]
    disagreements: Tuple[str, ...]
    recommended_discriminators: Tuple[str, ...]
    knowledge_proposals: Tuple[KnowledgeProposal, ...] = ()

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("debate outcome summary must be non-empty")


@dataclass(frozen=True, slots=True)
class CandidateJudgement:
    accepted: bool
    rationale: str
    action_fidelity: str
    semantic_novelty: str
    state_preservation: str
    rtl_risk: str
    revision_requirements: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.rationale,
                self.action_fidelity,
                self.semantic_novelty,
                self.state_preservation,
                self.rtl_risk,
            )
        ):
            raise ValueError("candidate judgement fields must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rationale": self.rationale,
            "action_fidelity": self.action_fidelity,
            "semantic_novelty": self.semantic_novelty,
            "state_preservation": self.state_preservation,
            "rtl_risk": self.rtl_risk,
            "revision_requirements": list(self.revision_requirements),
        }


@dataclass(frozen=True, slots=True)
class AdaptiveLoopResult:
    status: str
    rounds: int
    selected_candidate_node_id: Optional[str]
    selected_candidate_hash: Optional[str]
    plans: Tuple[PlanHypothesis, ...]
    candidates: Tuple[CandidateRecord, ...]
    evidence: Tuple[EvidenceRecord, ...]
    knowledge: Tuple[Mapping[str, Any], ...]
    decisions: Tuple[DirectorDecision, ...]
    round_events: Tuple[Tuple[RoundEvent, ...], ...]
    role_errors: Tuple[Mapping[str, Any], ...]
    resource_budget: Mapping[str, int] = field(default_factory=dict)
    resource_usage: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "rounds": self.rounds,
            "selected_candidate_node_id": self.selected_candidate_node_id,
            "selected_candidate_hash": self.selected_candidate_hash,
            "plans": [
                {
                    "plan_id": plan.plan_id,
                    "summary": plan.summary,
                    "distinguishing_assumption": plan.distinguishing_assumption,
                    "invariants": list(plan.invariants),
                }
                for plan in self.plans
            ],
            "candidates": [candidate.to_summary() for candidate in self.candidates],
            "evidence": [item.to_dict() for item in self.evidence],
            "knowledge": [dict(item) for item in self.knowledge],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "round_events": [
                [event.to_dict() for event in events] for events in self.round_events
            ],
            "role_errors": [dict(item) for item in self.role_errors],
            "resource_budget": dict(self.resource_budget),
            "resource_usage": dict(self.resource_usage),
        }
