"""Deterministic no-API roles that exercise the complete mechanism loop."""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

from .contracts import CandidateArtifact, Diagnosis, PlanHypothesis, TaskContract
from .node import ActionKind, EDAEvidence, EDAStage
from .roles import RoleSet


class ScriptedPlanner:
    def __init__(self) -> None:
        self.instructions: list[str] = []

    def propose(
        self,
        task: TaskContract,
        instruction: str,
        requested_count: int,
        existing_plans: Sequence[PlanHypothesis],
    ) -> Sequence[PlanHypothesis]:
        self.instructions.append(instruction)
        available = [
            PlanHypothesis(
                "streaming",
                "Use a single synchronous streaming datapath with explicit registered state.",
                "Outputs follow the specified cycle contract through one registered path.",
                ("preserve interface", "honor reset", "drive every output"),
            ),
            PlanHypothesis(
                "staged",
                "Separate state update and output selection into explicit stages.",
                "Control and datapath staging are separated while preserving latency.",
                ("preserve interface", "honor reset", "complete assignments"),
            ),
            PlanHypothesis(
                "minimal",
                "Use the smallest direct implementation consistent with the public contract.",
                "No additional state is introduced beyond required behavior.",
                ("preserve interface", "honor reset"),
            ),
        ]
        existing_ids = {plan.plan_id for plan in existing_plans}
        return tuple(plan for plan in available if plan.plan_id not in existing_ids)[:requested_count]


class ScriptedBuilder:
    def __init__(self) -> None:
        self.plan_calls: Counter[str] = Counter()
        self.instructions: list[str] = []

    def build(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        instruction: str,
        action: ActionKind,
        parent: Optional[CandidateArtifact],
        diagnosis: Optional[Diagnosis],
    ) -> CandidateArtifact:
        self.instructions.append(instruction)
        self.plan_calls[plan.plan_id] += 1
        call = self.plan_calls[plan.plan_id]
        target = task.target_files[0]
        if plan.plan_id == "streaming":
            if call == 1:
                return CandidateArtifact.from_mapping({})
            if call == 2:
                return CandidateArtifact.from_mapping(
                    {target: "module top; // SIM_FAIL_LOCALIZED\nendmodule\n"}
                )
            if call == 3:
                return CandidateArtifact.from_mapping({target: ""})
            return CandidateArtifact.from_mapping(
                {target: "module top; // SIM_FAIL_UNLOCALIZED\nendmodule\n"}
            )
        return CandidateArtifact.from_mapping(
            {target: "module top; // SCRIPTED_PASS\nendmodule\n"}
        )


class ScriptedVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def verify(self, task: TaskContract, candidate: CandidateArtifact) -> EDAEvidence:
        self.calls += 1
        content = "\n".join(candidate.as_dict().values())
        if "SCRIPTED_PASS" in content:
            return EDAEvidence(
                stage=EDAStage.PASS,
                original_stage="PASS",
                failure_class="none",
                failure_signature="no failure detected",
                complete_bundle=True,
                passed_subtests=1,
                total_subtests=1,
            )
        marker = "unlocalized functional mismatch" if "UNLOCALIZED" in content else "localized functional mismatch"
        return EDAEvidence(
            stage=EDAStage.SIMULATION,
            original_stage="CANDIDATE_FUNCTIONAL",
            failure_class="functional",
            failure_signature=marker,
            complete_bundle=True,
            passed_subtests=0,
            total_subtests=1,
        )


class ScriptedAnalyst:
    def __init__(self) -> None:
        self.instructions: list[str] = []

    def analyze(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        candidate: CandidateArtifact,
        evidence: EDAEvidence,
        instruction: str,
    ) -> Diagnosis:
        self.instructions.append(instruction)
        content = "\n".join(candidate.as_dict().values())
        if "UNLOCALIZED" in content:
            return Diagnosis(
                failure_class="functional_unlocalized",
                rationale="The public failure signal does not isolate a local repair.",
                repair_obligation="try a distinct public-spec-consistent plan",
                localized_failure=False,
                distinct_hypotheses_available=True,
            )
        return Diagnosis(
            failure_class="functional_localized",
            rationale="The public failure signal supports a bounded local correction.",
            repair_obligation="repair the localized output behavior without changing the interface",
            localized_failure=True,
        )


def make_scripted_roles() -> RoleSet:
    return RoleSet(
        planner=ScriptedPlanner(),
        builder=ScriptedBuilder(),
        verifier=ScriptedVerifier(),
        analyst=ScriptedAnalyst(),
    )
