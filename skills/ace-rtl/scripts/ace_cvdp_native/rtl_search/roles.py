"""Role protocols used by the mechanism loop and future ACE/LLM adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

from .contracts import CandidateArtifact, Diagnosis, PlanHypothesis, TaskContract
from .node import ActionKind, EDAEvidence


class PlannerRole(Protocol):
    def propose(
        self,
        task: TaskContract,
        instruction: str,
        requested_count: int,
        existing_plans: Sequence[PlanHypothesis],
    ) -> Sequence[PlanHypothesis]: ...


class BuilderRole(Protocol):
    def build(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        instruction: str,
        action: ActionKind,
        parent: Optional[CandidateArtifact],
        diagnosis: Optional[Diagnosis],
    ) -> CandidateArtifact: ...


class VerifierRole(Protocol):
    def verify(self, task: TaskContract, candidate: CandidateArtifact) -> EDAEvidence: ...


class AnalystRole(Protocol):
    def analyze(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        candidate: CandidateArtifact,
        evidence: EDAEvidence,
        instruction: str,
    ) -> Diagnosis: ...


@dataclass(frozen=True, slots=True)
class RoleSet:
    planner: PlannerRole
    builder: BuilderRole
    verifier: VerifierRole
    analyst: AnalystRole
    api_call_counter: Optional[Callable[[], int]] = None

    @property
    def api_calls(self) -> int:
        return self.api_call_counter() if self.api_call_counter is not None else 0
