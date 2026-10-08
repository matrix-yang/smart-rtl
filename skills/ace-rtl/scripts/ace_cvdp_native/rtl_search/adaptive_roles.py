"""Role protocols for the skill-defined BW-RTL round controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from .adaptive_contracts import (
    ActionCard,
    BranchReview,
    CandidateJudgement,
    DebateOutcome,
    DirectorDecision,
    ProbeOutcome,
    SearchSnapshot,
)
from .contracts import CandidateArtifact, PlanHypothesis, TaskContract
from .node import EDAEvidence


class AdaptivePlannerRole(Protocol):
    def propose(
        self,
        task: TaskContract,
        instruction: str,
        existing_plans: Sequence[PlanHypothesis],
        knowledge: Sequence[dict],
    ) -> Sequence[PlanHypothesis]: ...

    def revise(
        self,
        task: TaskContract,
        instruction: str,
        action: ActionCard,
        snapshot: SearchSnapshot,
    ) -> PlanHypothesis: ...


class AdaptiveBuilderRole(Protocol):
    def build(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        instruction: str,
        action: ActionCard,
        parents: Sequence[CandidateArtifact],
        snapshot: SearchSnapshot,
    ) -> CandidateArtifact: ...


class AdaptiveVerifierRole(Protocol):
    def verify(self, task: TaskContract, candidate: CandidateArtifact) -> EDAEvidence: ...


class AdaptiveAnalystRole(Protocol):
    def review(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        action: ActionCard,
        candidate: CandidateArtifact,
        evidence: EDAEvidence,
        evidence_id: str,
        candidate_node_id: str,
        instruction: str,
        snapshot: SearchSnapshot,
    ) -> BranchReview: ...


class AdaptiveDirectorRole(Protocol):
    def decide(
        self,
        task: TaskContract,
        instruction: str,
        snapshot: SearchSnapshot,
    ) -> DirectorDecision: ...


class AdaptiveCandidateJudgeRole(Protocol):
    def review_candidate(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        action: ActionCard,
        parents: Sequence[CandidateArtifact],
        candidate: CandidateArtifact,
        instruction: str,
        snapshot: SearchSnapshot,
    ) -> CandidateJudgement: ...


class AdaptiveProbeRole(Protocol):
    def probe(
        self,
        task: TaskContract,
        instruction: str,
        action: ActionCard,
        snapshot: SearchSnapshot,
    ) -> ProbeOutcome: ...


class AdaptiveDebaterRole(Protocol):
    def discuss(
        self,
        task: TaskContract,
        instruction: str,
        action: ActionCard,
        snapshot: SearchSnapshot,
    ) -> DebateOutcome: ...


@dataclass(frozen=True, slots=True)
class AdaptiveRoleSet:
    planner: AdaptivePlannerRole
    builder: AdaptiveBuilderRole
    verifier: AdaptiveVerifierRole
    analyst: AdaptiveAnalystRole
    director: AdaptiveDirectorRole
    debater: Optional[AdaptiveDebaterRole] = None
    probe: Optional[AdaptiveProbeRole] = None
    judge: Optional[AdaptiveCandidateJudgeRole] = None
    api_call_counter: Optional[Callable[[], int]] = None
    resource_usage_callback: Optional[Callable[[], Mapping[str, int]]] = None
    resource_cost_callback: Optional[Callable[[], Mapping[str, Any]]] = None

    @property
    def api_calls(self) -> int:
        return self.api_call_counter() if self.api_call_counter is not None else 0

    @property
    def resource_usage(self) -> Mapping[str, int]:
        if self.resource_usage_callback is None:
            return {"llm_calls": self.api_calls}
        return {
            str(key): int(value)
            for key, value in self.resource_usage_callback().items()
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        }

    @property
    def resource_cost_history(self) -> Mapping[str, Any]:
        return self.resource_cost_callback() if self.resource_cost_callback is not None else {}
