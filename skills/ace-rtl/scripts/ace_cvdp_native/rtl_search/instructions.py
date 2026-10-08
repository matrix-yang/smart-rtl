"""Replaceable, benchmark-agnostic role instructions for the first live loop.

These prompts intentionally contain no trace-derived repair knowledge.  A later
skill may implement the same interface and replace only this module's output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from .contracts import CandidateArtifact, Diagnosis, PlanHypothesis, TaskContract
from .node import ActionKind, EDAEvidence


class RoleInstructionProvider(Protocol):
    name: str
    version: str

    def planner(
        self,
        task: TaskContract,
        existing_plans: Sequence[PlanHypothesis],
        requested_count: int,
    ) -> str: ...

    def builder(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        action: ActionKind,
        parent: Optional[CandidateArtifact],
        diagnosis: Optional[Diagnosis],
    ) -> str: ...

    def analyst(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        candidate: CandidateArtifact,
        evidence: EDAEvidence,
    ) -> str: ...

    def director(self, task: TaskContract) -> str: ...


@dataclass(frozen=True, slots=True)
class GeneralACEInstructions:
    """General ACE-compatible role boundaries, before trace-derived skills."""

    name: str = "general_ace_role_contract"
    version: str = "v0.1"

    def planner(
        self,
        task: TaskContract,
        existing_plans: Sequence[PlanHypothesis],
        requested_count: int,
    ) -> str:
        existing = ", ".join(plan.plan_id for plan in existing_plans) or "none"
        targets = ", ".join(task.target_files)
        return f"""ROLE: RTL Planner
GOAL: derive {requested_count} materially distinct implementation plan hypothesis/hypotheses from the public RTL task.
PUBLIC SPECIFICATION:
{task.specification}
GENERATED TARGETS: {targets}
EXISTING PLAN IDS: {existing}
RULES:
- Preserve every public interface, parameter, reset, clocking, latency, and protocol requirement.
- Treat context, harnesses, tests, and verification assets as immutable evidence, never as generated targets.
- State the architectural assumption that distinguishes each plan and list invariants the implementation must preserve.
- Do not emit RTL, do not claim verification success, and do not use private tests or golden implementations.
- Return structured plan hypotheses only; avoid cosmetic variants of an existing plan.
"""

    def builder(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        action: ActionKind,
        parent: Optional[CandidateArtifact],
        diagnosis: Optional[Diagnosis],
    ) -> str:
        targets = ", ".join(task.target_files)
        parent_state = "none; implement the plan from the public task" if parent is None else parent.candidate_hash
        obligation = diagnosis.repair_obligation if diagnosis else "implement all public obligations"
        return f"""ROLE: RTL Builder
GOAL: execute action {action.value.upper()} for plan {plan.plan_id} and produce a complete synthesizable RTL candidate.
PLAN: {plan.summary}
DISTINGUISHING ASSUMPTION: {plan.distinguishing_assumption}
PARENT CANDIDATE: {parent_state}
CURRENT OBLIGATION: {obligation}
EXACT GENERATED TARGETS: {targets}
RULES:
- Return the complete content of every declared target file on every call; a patch or omitted target is invalid.
- Do not create a testbench, checker, assertion harness, golden model, or undeclared file.
- Do not edit immutable context, benchmark harnesses, tests, build scripts, or evaluator behavior.
- Preserve the public module names and interfaces and keep the result synthesizable.
- For REFINE, make the smallest evidence-grounded change that addresses the obligation without regressing satisfied behavior.
- For EXPAND, implement a genuinely different hypothesis or repair route rather than paraphrasing the parent.
- Emit RTL artifacts only; do not self-report PASS.
"""

    def analyst(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        candidate: CandidateArtifact,
        evidence: EDAEvidence,
    ) -> str:
        return f"""ROLE: RTL Failure Analyst
GOAL: convert public EDA evidence into a falsifiable diagnosis for the Director and Builder.
PLAN: {plan.plan_id} — {plan.summary}
CANDIDATE HASH: {candidate.candidate_hash}
EDA STAGE: {evidence.original_stage}
FAILURE SIGNATURE: {evidence.failure_signature}
PUBLIC EVALUATOR FEEDBACK:
{evidence.public_feedback or "No additional public evaluator feedback was recorded."}
RULES:
- Use only the public specification, immutable public context, candidate RTL, and recorded EDA evidence.
- Distinguish infrastructure, incomplete bundle, compile/elaboration, localized functional, and plan/spec failures.
- State one bounded repair obligation and whether the failure is localized.
- Mark plan conflict or alternative hypotheses only when the evidence supports it.
- Do not write RTL, modify evidence, inspect private/golden artifacts, or declare correctness.
"""

    def director(self, task: TaskContract) -> str:
        return f"""ROLE: RTL Director
GOAL: manage the global executable search tree for task {task.task_id}.
RULES:
- Choose only EXPAND, REFINE, BACKTRACK, or STOP from immutable tree state, structured diagnosis, and EDA evidence.
- Hard EDA stage ordering overrides language-model confidence.
- Preserve promising ancestors; never overwrite a candidate in place.
- Backtrack on structural regression, exact repetition, or oscillation; stop only on fail-closed pass, infrastructure failure, or exhausted legal work.
- Do not write RTL, reinterpret private tests, or modify the verifier.
"""
