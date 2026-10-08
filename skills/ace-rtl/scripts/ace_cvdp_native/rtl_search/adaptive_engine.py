"""Variable-width, round-based search controlled by a skill-defined Director."""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..processes import write_json
from .adaptive_contracts import (
    ActionCard,
    AdaptiveActionKind,
    AdaptiveLoopResult,
    BranchExecution,
    BranchStatus,
    CandidateRecord,
    DebateOutcome,
    DirectorDecision,
    EvidenceRecord,
    ProbeOutcome,
    RoundEvent,
    SearchSnapshot,
)
from .adaptive_roles import AdaptiveRoleSet
from .contracts import CandidateArtifact, PlanHypothesis, TaskContract
from .knowledge import KnowledgeLedger
from .node import EDAEvidence, EDAStage
from .skill_instructions import BWRTLSkillInstructions


@dataclass(frozen=True, slots=True)
class _WorkOutcome:
    action: ActionCard
    payload: Any = None
    error: Optional[str] = None
    cancelled_reason: Optional[str] = None


class AdaptiveRTLDirectorEngine:
    """Execute LLM-selected actions while enforcing deterministic invariants.

    ``physical_workers`` limits only how many independent role pipelines run at
    once.  It never truncates, ranks, or selects the Director's dispatch.
    ``max_rounds`` is a fail-safe against an unresponsive controller, not a
    search-width or candidate-budget policy.
    """

    def __init__(
        self,
        *,
        roles: AdaptiveRoleSet,
        instructions: BWRTLSkillInstructions,
        physical_workers: int = 4,
        max_rounds: int = 20,
        checkpoint_dir: Optional[Path] = None,
        resource_budget: Optional[Mapping[str, int]] = None,
    ) -> None:
        if physical_workers < 1:
            raise ValueError("physical_workers must be positive")
        if max_rounds < 1:
            raise ValueError("max_rounds safety guard must be positive")
        self.roles = roles
        self.instructions = instructions
        self.physical_workers = physical_workers
        self.max_rounds = max_rounds
        self.checkpoint_dir = checkpoint_dir
        supported_resources = {"total_tokens", "llm_calls", "verifier_calls"}
        raw_budget = resource_budget or {}
        unknown_resources = set(raw_budget) - supported_resources
        if unknown_resources:
            raise ValueError(
                "unsupported resource budget keys: " + ", ".join(sorted(unknown_resources))
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in raw_budget.values()
        ):
            raise ValueError("resource budget values must be positive integers")
        self.resource_budget = dict(raw_budget)

    def run(self, task: TaskContract) -> AdaptiveLoopResult:
        plans: Dict[str, PlanHypothesis] = {}
        candidates: Dict[str, CandidateRecord] = {}
        evidence: Dict[str, EvidenceRecord] = {}
        ledger = KnowledgeLedger()
        decisions: List[DirectorDecision] = []
        all_round_events: List[Tuple[RoundEvent, ...]] = []
        role_errors: List[Mapping[str, Any]] = []
        prior_events: Tuple[RoundEvent, ...] = ()
        candidate_counter = 0
        evidence_counter = 0

        try:
            proposed = self.roles.planner.propose(
                task,
                self.instructions.for_role("planner"),
                (),
                (),
            )
            self._insert_plans(plans, proposed)
        except Exception as exc:
            role_errors.append({"round": 0, "role": "planner", "error": str(exc)})
            return self._result(
                "no_plan",
                0,
                None,
                plans,
                candidates,
                evidence,
                ledger,
                decisions,
                all_round_events,
                role_errors,
            )
        if not plans:
            return self._result(
                "no_plan",
                0,
                None,
                plans,
                candidates,
                evidence,
                ledger,
                decisions,
                all_round_events,
                role_errors,
            )

        budget_status = self._budget_terminal_status()
        if budget_status is not None:
            result = self._result(
                budget_status,
                0,
                None,
                plans,
                candidates,
                evidence,
                ledger,
                decisions,
                all_round_events,
                role_errors,
            )
            self._checkpoint(result)
            return result

        for round_index in range(1, self.max_rounds + 1):
            snapshot = self._snapshot(
                round_index,
                plans,
                candidates,
                evidence,
                ledger,
                prior_events,
            )
            try:
                decision = self.roles.director.decide(
                    task,
                    self.instructions.for_role("director"),
                    snapshot,
                )
                self._validate_decision(
                    decision,
                    plans,
                    candidates,
                    evidence,
                    ledger,
                    round_index=round_index,
                )
                ledger.apply_decisions(decision.knowledge_decisions)
            except Exception as exc:
                role_errors.append(
                    {"round": round_index, "role": "director", "error": str(exc)}
                )
                terminal_status = self._budget_terminal_status() or "director_error"
                result = self._result(
                    terminal_status,
                    round_index,
                    None,
                    plans,
                    candidates,
                    evidence,
                    ledger,
                    decisions,
                    all_round_events,
                    role_errors,
                )
                self._checkpoint(result)
                return result

            decisions.append(decision)
            budget_status = self._budget_terminal_status()
            if budget_status == "resource_overrun":
                result = self._result(
                    budget_status,
                    round_index,
                    None,
                    plans,
                    candidates,
                    evidence,
                    ledger,
                    decisions,
                    all_round_events,
                    role_errors,
                )
                self._checkpoint(result)
                return result
            required_tokens = self._decision_token_reservation(decision)
            remaining_tokens = self._resource_remaining().get("total_tokens")
            if (
                remaining_tokens is not None
                and required_tokens > remaining_tokens
                and decision.dispatch[0].kind is not AdaptiveActionKind.STOP
            ):
                events = (
                    RoundEvent(
                        f"automatic-reservation-stop-r{round_index}",
                        AdaptiveActionKind.STOP,
                        "resource_limit",
                        (),
                        "Selected work was not started because its minimum completion "
                        f"reservation ({required_tokens} tokens) exceeds remaining "
                        f"capacity ({remaining_tokens} tokens).",
                    ),
                )
                all_round_events.append(events)
                result = self._result(
                    "resource_limit",
                    round_index,
                    None,
                    plans,
                    candidates,
                    evidence,
                    ledger,
                    decisions,
                    all_round_events,
                    role_errors,
                )
                self._checkpoint(result)
                return result
            if decision.dispatch[0].kind is AdaptiveActionKind.STOP:
                source_ids = decision.dispatch[0].source_node_ids
                selected = source_ids[0] if source_ids else None
                status = "passed" if selected else "stopped_without_correctness"
                events = (
                    RoundEvent(
                        decision.dispatch[0].action_id,
                        AdaptiveActionKind.STOP,
                        status,
                        (selected,) if selected else (),
                        decision.rationale,
                    ),
                )
                all_round_events.append(events)
                result = self._result(
                    status,
                    round_index,
                    selected,
                    plans,
                    candidates,
                    evidence,
                    ledger,
                    decisions,
                    all_round_events,
                    role_errors,
                )
                self._checkpoint(result)
                return result

            if budget_status is not None:
                result = self._result(
                    budget_status,
                    round_index,
                    None,
                    plans,
                    candidates,
                    evidence,
                    ledger,
                    decisions,
                    all_round_events,
                    role_errors,
                )
                self._checkpoint(result)
                return result

            immediate_events: List[RoundEvent] = []
            executable: List[Tuple[ActionCard, Optional[str], Optional[str]]] = []
            for action in decision.dispatch:
                if action.kind in {
                    AdaptiveActionKind.BACKTRACK,
                    AdaptiveActionKind.PARK,
                    AdaptiveActionKind.PRUNE,
                }:
                    immediate_events.append(self._apply_state_action(action, candidates))
                    continue
                node_id = None
                evidence_id = None
                if action.kind.produces_candidate:
                    candidate_counter += 1
                    evidence_counter += 1
                    node_id = f"c{candidate_counter:04d}"
                    evidence_id = f"e{evidence_counter:04d}"
                executable.append((action, node_id, evidence_id))

            work_outcomes = self._execute_round(
                task,
                snapshot,
                executable,
                plans,
                candidates,
            )
            round_events = list(immediate_events)
            for outcome in work_outcomes:
                action = outcome.action
                if outcome.cancelled_reason is not None:
                    round_events.append(
                        RoundEvent(
                            action.action_id,
                            action.kind,
                            "cancelled_after_sibling_pass",
                            (),
                            outcome.cancelled_reason,
                        )
                    )
                    continue
                if outcome.error is not None:
                    role_errors.append(
                        {
                            "round": round_index,
                            "action_id": action.action_id,
                            "kind": action.kind.value,
                            "error": outcome.error,
                        }
                    )
                    round_events.append(
                        RoundEvent(action.action_id, action.kind, "role_error", (), outcome.error)
                    )
                    continue
                if action.kind.produces_candidate:
                    execution = outcome.payload
                    if not isinstance(execution, BranchExecution):
                        raise RuntimeError("candidate action returned invalid execution payload")
                    event = self._integrate_branch(
                        execution,
                        round_index,
                        candidates,
                        evidence,
                        ledger,
                        role_errors,
                    )
                    round_events.append(event)
                elif action.kind.invokes_planner:
                    plan = outcome.payload
                    try:
                        self._insert_plans(plans, (plan,))
                        round_events.append(
                            RoundEvent(
                                action.action_id,
                                action.kind,
                                "completed",
                                (plan.plan_id,),
                                plan.summary,
                            )
                        )
                    except Exception as exc:
                        role_errors.append(
                            {
                                "round": round_index,
                                "action_id": action.action_id,
                                "role": "planner",
                                "error": str(exc),
                            }
                        )
                        round_events.append(
                            RoundEvent(
                                action.action_id,
                                action.kind,
                                "rejected",
                                (),
                                str(exc),
                            )
                        )
                elif action.kind is AdaptiveActionKind.PROBE:
                    probe = outcome.payload
                    if not isinstance(probe, ProbeOutcome):
                        raise RuntimeError("probe action returned invalid payload")
                    try:
                        created = ledger.add_proposals(
                            probe.knowledge_proposals,
                            source_role="probe",
                            source_id=action.action_id,
                            valid_evidence_refs=evidence,
                        )
                        round_events.append(
                            RoundEvent(
                                action.action_id,
                                action.kind,
                                "completed",
                                created,
                                probe.summary,
                            )
                        )
                    except Exception as exc:
                        role_errors.append(
                            {
                                "round": round_index,
                                "action_id": action.action_id,
                                "role": "probe",
                                "error": str(exc),
                            }
                        )
                        round_events.append(
                            RoundEvent(
                                action.action_id,
                                action.kind,
                                "knowledge_rejected",
                                (),
                                str(exc),
                            )
                        )
                elif action.kind is AdaptiveActionKind.DISCUSS:
                    debate = outcome.payload
                    if not isinstance(debate, DebateOutcome):
                        raise RuntimeError("discuss action returned invalid payload")
                    try:
                        created = ledger.add_proposals(
                            debate.knowledge_proposals,
                            source_role="debater",
                            source_id=action.action_id,
                            valid_evidence_refs=evidence,
                        )
                        round_events.append(
                            RoundEvent(
                                action.action_id,
                                action.kind,
                                "completed",
                                created,
                                debate.summary,
                            )
                        )
                    except Exception as exc:
                        role_errors.append(
                            {
                                "round": round_index,
                                "action_id": action.action_id,
                                "role": "debater",
                                "error": str(exc),
                            }
                        )
                        round_events.append(
                            RoundEvent(
                                action.action_id,
                                action.kind,
                                "knowledge_rejected",
                                (),
                                str(exc),
                            )
                        )

            passed_node = self._passed_candidate_id(candidates, evidence)
            budget_status = self._budget_terminal_status()
            if passed_node is not None and budget_status != "resource_overrun":
                round_events.append(
                    RoundEvent(
                        f"automatic-pass-stop-r{round_index}",
                        AdaptiveActionKind.STOP,
                        "passed",
                        (passed_node,),
                        "Fail-closed passing evidence is terminal; no additional LLM "
                        "scheduling call was spent.",
                    )
                )
                prior_events = tuple(round_events)
                all_round_events.append(prior_events)
                result = self._result(
                    "passed",
                    round_index,
                    passed_node,
                    plans,
                    candidates,
                    evidence,
                    ledger,
                    decisions,
                    all_round_events,
                    role_errors,
                )
                self._checkpoint(result)
                return result
            if budget_status is not None:
                round_events.append(
                    RoundEvent(
                        f"automatic-resource-stop-r{round_index}",
                        AdaptiveActionKind.STOP,
                        budget_status,
                        (),
                        "The configured heterogeneous resource envelope cannot fund "
                        "another controller step.",
                    )
                )
                prior_events = tuple(round_events)
                all_round_events.append(prior_events)
                result = self._result(
                    budget_status,
                    round_index,
                    None,
                    plans,
                    candidates,
                    evidence,
                    ledger,
                    decisions,
                    all_round_events,
                    role_errors,
                )
                self._checkpoint(result)
                return result
            prior_events = tuple(round_events)
            all_round_events.append(prior_events)
            partial = self._result(
                "running",
                round_index,
                None,
                plans,
                candidates,
                evidence,
                ledger,
                decisions,
                all_round_events,
                role_errors,
            )
            self._checkpoint(partial)

        result = self._result(
            "safety_limit",
            self.max_rounds,
            None,
            plans,
            candidates,
            evidence,
            ledger,
            decisions,
            all_round_events,
            role_errors,
        )
        self._checkpoint(result)
        return result

    def _execute_round(
        self,
        task: TaskContract,
        snapshot: SearchSnapshot,
        work: Sequence[Tuple[ActionCard, Optional[str], Optional[str]]],
        plans: Mapping[str, PlanHypothesis],
        candidates: Mapping[str, CandidateRecord],
    ) -> Tuple[_WorkOutcome, ...]:
        if not work:
            return ()
        sibling_passed = threading.Event()
        futures: List[Tuple[int, Future[_WorkOutcome]]] = []
        with ThreadPoolExecutor(
            max_workers=self.physical_workers,
            thread_name_prefix="bw-rtl-round",
        ) as executor:
            for index, (action, node_id, evidence_id) in enumerate(work):
                future = executor.submit(
                    self._execute_action,
                    task,
                    snapshot,
                    action,
                    node_id,
                    evidence_id,
                    plans,
                    candidates,
                    sibling_passed,
                )
                futures.append((index, future))
            ordered: List[Optional[_WorkOutcome]] = [None] * len(futures)
            for index, future in futures:
                try:
                    ordered[index] = future.result()
                except Exception as exc:  # fail branch-locally
                    action = work[index][0]
                    ordered[index] = _WorkOutcome(action, error=str(exc))
        return tuple(item for item in ordered if item is not None)

    def _execute_action(
        self,
        task: TaskContract,
        snapshot: SearchSnapshot,
        action: ActionCard,
        node_id: Optional[str],
        evidence_id: Optional[str],
        plans: Mapping[str, PlanHypothesis],
        candidates: Mapping[str, CandidateRecord],
        sibling_passed: threading.Event,
    ) -> _WorkOutcome:
        if sibling_passed.is_set():
            return _WorkOutcome(
                action,
                cancelled_reason=(
                    "A sibling produced fail-closed passing evidence before this action "
                    "started; return control to Director."
                ),
            )
        if action.kind.produces_candidate:
            assert node_id is not None and evidence_id is not None and action.plan_id is not None
            plan = plans[action.plan_id]
            parents = tuple(candidates[node].artifact for node in action.source_node_ids)
            try:
                artifact = self.roles.builder.build(
                    task,
                    plan,
                    self.instructions.for_role("builder"),
                    action,
                    parents,
                    snapshot,
                )
            except Exception as exc:
                return _WorkOutcome(action, error=f"builder: {exc}")
            duplicate_sources = tuple(
                source_node_id
                for source_node_id, parent in zip(action.source_node_ids, parents)
                if artifact.candidate_hash == parent.candidate_hash
            )
            if duplicate_sources:
                source_text = ", ".join(duplicate_sources)
                signature = (
                    f"candidate hash {artifact.candidate_hash} exactly matches source "
                    f"candidate(s): {source_text}"
                )
                eda = EDAEvidence(
                    stage=EDAStage.INCOMPLETE,
                    original_stage="LOCAL_NO_PROGRESS",
                    failure_class="candidate_no_change",
                    failure_signature=signature,
                    complete_bundle=False,
                    public_feedback=(
                        f"The normalized Builder artifact made no textual change from "
                        f"{source_text}. The candidate was rejected before EDA verification. "
                        "Inspect Builder output extraction/normalization or choose a distinct "
                        "repair action."
                    ),
                )
                return _WorkOutcome(
                    action,
                    BranchExecution(action, node_id, evidence_id, artifact, eda, None),
                )
            issues = artifact.contract_issues(task)
            if issues:
                eda = EDAEvidence(
                    stage=EDAStage.INCOMPLETE,
                    original_stage="LOCAL_CONTRACT",
                    failure_class="candidate_contract",
                    failure_signature=issues[0],
                    complete_bundle=False,
                )
                return _WorkOutcome(
                    action,
                    BranchExecution(action, node_id, evidence_id, artifact, eda, None),
                )
            judge_note = ""
            judgement = None
            if (
                self.roles.judge is not None
                and parents
                and action.kind
                in {
                    AdaptiveActionKind.REFINE,
                    AdaptiveActionKind.FORK,
                    AdaptiveActionKind.FUSE,
                }
            ):
                try:
                    judgement = self.roles.judge.review_candidate(
                        task,
                        plan,
                        action,
                        parents,
                        artifact,
                        self.instructions.for_role("judge"),
                        snapshot,
                    )
                except Exception as exc:
                    eda = EDAEvidence(
                        stage=EDAStage.INCOMPLETE,
                        original_stage="LOCAL_CANDIDATE_JUDGE_ERROR",
                        failure_class="candidate_judge_error",
                        failure_signature=str(exc)[:500],
                        complete_bundle=False,
                        public_feedback=(
                            "Candidate Judge failed closed before EDA: " + str(exc)[:2000]
                        ),
                    )
                    return _WorkOutcome(
                        action,
                        BranchExecution(
                            action,
                            node_id,
                            evidence_id,
                            artifact,
                            eda,
                            None,
                            f"judge: {exc}",
                        ),
                    )
                if not judgement.accepted:
                    rendered = json.dumps(judgement.to_dict(), indent=2)
                    eda = EDAEvidence(
                        stage=EDAStage.INCOMPLETE,
                        original_stage="LOCAL_ACTION_CONTRACT",
                        failure_class="candidate_action_contract",
                        failure_signature=judgement.rationale[:500],
                        complete_bundle=False,
                        public_feedback=(
                            "Candidate Judge rejected this source transform before EDA.\n"
                            + rendered[:6000]
                        ),
                    )
                    return _WorkOutcome(
                        action,
                        BranchExecution(
                            action,
                            node_id,
                            evidence_id,
                            artifact,
                            eda,
                            None,
                            None,
                            "Candidate Judge rejected the delta before evaluator spend.",
                            candidate_judgement=judgement,
                        ),
                    )
                judge_note = "Candidate Judge accepted the bounded source transform."
                if self._budget_terminal_status() is not None:
                    eda = EDAEvidence(
                        stage=EDAStage.INCOMPLETE,
                        original_stage="LOCAL_RESOURCE_LIMIT",
                        failure_class="resource_limit_before_verifier",
                        failure_signature="resource envelope exhausted after Candidate Judge",
                        complete_bundle=False,
                        public_feedback=(
                            "Candidate was not sent to EDA because the resource envelope "
                            "was exhausted by the pre-verification review."
                        ),
                    )
                    return _WorkOutcome(
                        action,
                        BranchExecution(action, node_id, evidence_id, artifact, eda, None),
                    )
            try:
                eda = self.roles.verifier.verify(task, artifact)
            except Exception as exc:
                return _WorkOutcome(action, error=f"verifier: {exc}")
            if eda.passed:
                sibling_passed.set()
            review = None
            analyst_error = None
            analysis_note = judge_note
            if not eda.passed and eda.stage is not EDAStage.INFRASTRUCTURE:
                if sibling_passed.is_set():
                    analysis_note = (
                        (analysis_note + " ") if analysis_note else ""
                    ) + (
                        "Analyst skipped because a sibling already produced fail-closed "
                        "passing evidence in this round."
                    )
                else:
                    try:
                        review = self.roles.analyst.review(
                            task,
                            plan,
                            action,
                            artifact,
                            eda,
                            evidence_id,
                            node_id,
                            self.instructions.for_role("analyst"),
                            snapshot,
                        )
                        if review.candidate_node_id != node_id or review.evidence_id != evidence_id:
                            raise ValueError("Analyst review references the wrong branch or evidence")
                    except Exception as exc:
                        analyst_error = f"analyst: {exc}"
            return _WorkOutcome(
                action,
                BranchExecution(
                    action,
                    node_id,
                    evidence_id,
                    artifact,
                    eda,
                    review,
                    analyst_error,
                    analysis_note,
                    candidate_judgement=judgement,
                ),
            )
        if action.kind.invokes_planner:
            try:
                plan = self.roles.planner.revise(
                    task,
                    self.instructions.for_role("planner"),
                    action,
                    snapshot,
                )
                return _WorkOutcome(action, plan)
            except Exception as exc:
                return _WorkOutcome(action, error=f"planner: {exc}")
        if action.kind is AdaptiveActionKind.PROBE:
            if self.roles.probe is None:
                return _WorkOutcome(action, error="probe role is not configured")
            try:
                probe = self.roles.probe.probe(
                    task,
                    self.instructions.for_role("verifier"),
                    action,
                    snapshot,
                )
                return _WorkOutcome(action, probe)
            except Exception as exc:
                return _WorkOutcome(action, error=f"probe: {exc}")
        if action.kind is AdaptiveActionKind.DISCUSS:
            if self.roles.debater is None:
                return _WorkOutcome(action, error="debater role is not configured")
            try:
                debate = self.roles.debater.discuss(
                    task,
                    self.instructions.for_role("debater"),
                    action,
                    snapshot,
                )
                return _WorkOutcome(action, debate)
            except Exception as exc:
                return _WorkOutcome(action, error=f"debater: {exc}")
        return _WorkOutcome(action, error=f"unhandled executable action {action.kind.value}")

    def _integrate_branch(
        self,
        execution: BranchExecution,
        round_index: int,
        candidates: Dict[str, CandidateRecord],
        evidence: Dict[str, EvidenceRecord],
        ledger: KnowledgeLedger,
        role_errors: List[Mapping[str, Any]],
    ) -> RoundEvent:
        if execution.artifact is None or execution.evidence is None:
            return RoundEvent(
                execution.action.action_id,
                execution.action.kind,
                "role_error",
                (),
                execution.error or "branch did not produce an artifact and evidence",
            )
        if execution.evidence.passed:
            status = BranchStatus.PASSED
        elif execution.evidence.failure_class in {
            "candidate_no_change",
            "candidate_action_contract",
            "candidate_judge_error",
            "resource_limit_before_verifier",
        }:
            status = BranchStatus.PRUNED
        else:
            status = BranchStatus.ACTIVE
        record = CandidateRecord(
            node_id=execution.node_id,
            plan_id=execution.action.plan_id or "",
            parent_node_ids=execution.action.source_node_ids,
            producing_action_id=execution.action.action_id,
            artifact=execution.artifact,
            status=status,
            round_created=round_index,
            evidence_id=execution.evidence_id,
            review=execution.review,
            error=execution.error,
            candidate_judgement=execution.candidate_judgement,
        )
        candidates[record.node_id] = record
        evidence[execution.evidence_id] = EvidenceRecord(
            execution.evidence_id,
            execution.node_id,
            execution.evidence,
        )
        detail = execution.evidence.original_stage
        if execution.note:
            detail += f"; {execution.note}"
        if execution.error:
            role_errors.append(
                {
                    "round": round_index,
                    "action_id": execution.action.action_id,
                    "candidate_node_id": execution.node_id,
                    "error": execution.error,
                }
            )
            detail += f"; {execution.error}"
        created: Tuple[str, ...] = ()
        if execution.review is not None:
            try:
                created = ledger.add_proposals(
                    execution.review.knowledge_proposals,
                    source_role="analyst",
                    source_id=execution.node_id,
                    valid_evidence_refs=evidence,
                )
            except Exception as exc:
                role_errors.append(
                    {
                        "round": round_index,
                        "action_id": execution.action.action_id,
                        "candidate_node_id": execution.node_id,
                        "role": "knowledge_validator",
                        "error": str(exc),
                    }
                )
                detail += f"; knowledge rejected: {exc}"
        return RoundEvent(
            execution.action.action_id,
            execution.action.kind,
            "completed",
            (execution.node_id, execution.evidence_id, *created),
            detail,
        )

    @staticmethod
    def _insert_plans(
        plans: Dict[str, PlanHypothesis], proposed: Sequence[PlanHypothesis]
    ) -> None:
        summaries = {plan.summary.strip().lower() for plan in plans.values()}
        for plan in proposed:
            if plan.plan_id in plans:
                raise ValueError(f"duplicate plan id: {plan.plan_id}")
            normalized = plan.summary.strip().lower()
            if normalized in summaries:
                raise ValueError(f"cosmetic duplicate plan summary: {plan.summary}")
            plans[plan.plan_id] = plan
            summaries.add(normalized)

    def _validate_decision(
        self,
        decision: DirectorDecision,
        plans: Mapping[str, PlanHypothesis],
        candidates: Mapping[str, CandidateRecord],
        evidence: Mapping[str, EvidenceRecord],
        ledger: KnowledgeLedger,
        *,
        round_index: int,
    ) -> None:
        if not decision.dispatch:
            raise ValueError("Director dispatch cannot be empty; use an explicit stop action")
        if any(action.kind is AdaptiveActionKind.STOP for action in decision.dispatch):
            if len(decision.dispatch) != 1:
                raise ValueError("stop must be the only action in its round")
        if round_index == self.max_rounds and not any(
            action.kind.produces_candidate or action.kind is AdaptiveActionKind.STOP
            for action in decision.dispatch
        ):
            raise ValueError(
                "final-round dispatch must produce verifier evidence or stop; "
                "review/replan/state-only work cannot be consumed after the safety horizon"
            )
        verifier_remaining = self._resource_remaining().get("verifier_calls")
        candidate_actions = sum(action.kind.produces_candidate for action in decision.dispatch)
        if verifier_remaining is not None and candidate_actions > verifier_remaining:
            raise ValueError(
                "dispatch requires more verifier calls than remain in the resource "
                f"envelope: requested={candidate_actions}, remaining={verifier_remaining}"
            )
        semantic_keys: set[Tuple[Any, ...]] = set()
        for action in decision.dispatch:
            key = (
                action.kind.value,
                action.plan_id,
                action.source_node_ids,
                action.hypothesis.strip().lower(),
                action.obligation.strip().lower(),
            )
            if key in semantic_keys:
                raise ValueError(f"duplicate semantic action in dispatch: {action.action_id}")
            semantic_keys.add(key)
            for reference in action.evidence_refs:
                if reference not in evidence:
                    raise ValueError(
                        f"action {action.action_id} references unknown evidence {reference}"
                    )
            for reference in action.knowledge_refs:
                if not ledger.contains(reference):
                    raise ValueError(
                        f"action {action.action_id} references unknown knowledge {reference}"
                    )
            self._validate_action_refs(action, plans, candidates, evidence)

    def _validate_action_refs(
        self,
        action: ActionCard,
        plans: Mapping[str, PlanHypothesis],
        candidates: Mapping[str, CandidateRecord],
        evidence: Mapping[str, EvidenceRecord],
    ) -> None:
        kind = action.kind
        if kind is AdaptiveActionKind.PROBE and self.roles.probe is None:
            raise ValueError("probe is not available in this runtime")
        if kind is AdaptiveActionKind.DISCUSS and self.roles.debater is None:
            raise ValueError("discuss is not available in this runtime")
        if kind.produces_candidate:
            if not action.plan_id or action.plan_id not in plans:
                raise ValueError(f"action {action.action_id} requires a known plan_id")
        if kind is AdaptiveActionKind.SEED_PLAN and action.source_node_ids:
            raise ValueError("seed_plan cannot have source candidates")
        if kind is AdaptiveActionKind.REFINE and len(action.source_node_ids) != 1:
            raise ValueError("refine requires exactly one source candidate")
        if kind is AdaptiveActionKind.FORK and len(action.source_node_ids) > 1:
            raise ValueError("fork accepts at most one source candidate")
        if kind is AdaptiveActionKind.FUSE and len(action.source_node_ids) < 2:
            raise ValueError("fuse requires at least two source candidates")
        if kind is AdaptiveActionKind.PROBE and len(action.source_node_ids) != 1:
            raise ValueError("probe requires exactly one source candidate")
        if kind is AdaptiveActionKind.DISCUSS and len(action.source_node_ids) < 2:
            raise ValueError("discuss requires at least two source candidates")
        if kind is AdaptiveActionKind.BACKTRACK and len(action.source_node_ids) != 2:
            raise ValueError("backtrack requires source and destination ids")
        if kind in {AdaptiveActionKind.PARK, AdaptiveActionKind.PRUNE} and not action.source_node_ids:
            raise ValueError(f"{kind.value} requires at least one candidate")
        if kind is AdaptiveActionKind.STOP:
            if len(action.source_node_ids) > 1:
                raise ValueError("stop accepts at most one candidate")
            if action.source_node_ids:
                node_id = action.source_node_ids[0]
                if node_id not in candidates:
                    raise ValueError(f"stop references unknown candidate {node_id}")
                record = candidates[node_id]
                if not record.evidence_id or record.evidence_id not in evidence:
                    raise ValueError("stop candidate has no verifier evidence")
                if not evidence[record.evidence_id].evidence.passed:
                    raise ValueError("a correctness stop requires fail-closed passing evidence")
            elif any(item.evidence.passed for item in evidence.values()):
                raise ValueError(
                    "source-free stop cannot ignore fail-closed passing evidence; "
                    "cite exactly one passed candidate"
                )

        for node_id in action.source_node_ids:
            if kind is AdaptiveActionKind.BACKTRACK and node_id in plans:
                continue
            if node_id not in candidates:
                raise ValueError(f"action {action.action_id} references unknown node {node_id}")
        if kind is AdaptiveActionKind.BACKTRACK:
            source, destination = action.source_node_ids
            if not self._is_ancestor(destination, source, plans, candidates):
                raise ValueError(
                    f"backtrack destination {destination} is not an ancestor of {source}"
                )

    @staticmethod
    def _is_ancestor(
        destination: str,
        source: str,
        plans: Mapping[str, PlanHypothesis],
        candidates: Mapping[str, CandidateRecord],
    ) -> bool:
        if source not in candidates:
            return False
        if destination in plans:
            return candidates[source].plan_id == destination
        seen: set[str] = set()
        frontier = list(candidates[source].parent_node_ids)
        while frontier:
            current = frontier.pop()
            if current == destination:
                return True
            if current in seen or current not in candidates:
                continue
            seen.add(current)
            frontier.extend(candidates[current].parent_node_ids)
        return False

    @staticmethod
    def _apply_state_action(
        action: ActionCard,
        candidates: Dict[str, CandidateRecord],
    ) -> RoundEvent:
        if action.kind is AdaptiveActionKind.BACKTRACK:
            source, destination = action.source_node_ids
            candidates[source] = replace(candidates[source], status=BranchStatus.BACKTRACKED)
            if destination in candidates:
                candidates[destination] = replace(
                    candidates[destination], status=BranchStatus.ACTIVE
                )
            return RoundEvent(
                action.action_id,
                action.kind,
                "completed",
                (source, destination),
                action.obligation,
            )
        status = (
            BranchStatus.PARKED
            if action.kind is AdaptiveActionKind.PARK
            else BranchStatus.PRUNED
        )
        for node_id in action.source_node_ids:
            candidates[node_id] = replace(candidates[node_id], status=status)
        return RoundEvent(
            action.action_id,
            action.kind,
            "completed",
            action.source_node_ids,
            action.obligation,
        )

    def _snapshot(
        self,
        round_index: int,
        plans: Mapping[str, PlanHypothesis],
        candidates: Mapping[str, CandidateRecord],
        evidence: Mapping[str, EvidenceRecord],
        ledger: KnowledgeLedger,
        prior_events: Tuple[RoundEvent, ...],
    ) -> SearchSnapshot:
        available_actions = tuple(
            action.value
            for action in AdaptiveActionKind
            if (action is not AdaptiveActionKind.PROBE or self.roles.probe is not None)
            and (action is not AdaptiveActionKind.DISCUSS or self.roles.debater is not None)
        )
        return SearchSnapshot(
            round_index=round_index,
            plans=tuple(plans.values()),
            candidates=tuple(candidates.values()),
            evidence=tuple(evidence.values()),
            knowledge=ledger.snapshots(),
            prior_round_events=prior_events,
            physical_worker_limit=self.physical_workers,
            available_actions=available_actions,
            max_rounds=self.max_rounds,
            resource_budget=self.resource_budget,
            resource_usage=self._resource_usage(),
            resource_remaining=self._resource_remaining(),
            resource_cost_history=self.roles.resource_cost_history,
            lineage_diagnostics=self._lineage_diagnostics(candidates, evidence),
        )

    def _result(
        self,
        status: str,
        rounds: int,
        selected: Optional[str],
        plans: Mapping[str, PlanHypothesis],
        candidates: Mapping[str, CandidateRecord],
        evidence: Mapping[str, EvidenceRecord],
        ledger: KnowledgeLedger,
        decisions: Sequence[DirectorDecision],
        round_events: Sequence[Tuple[RoundEvent, ...]],
        role_errors: Sequence[Mapping[str, Any]],
    ) -> AdaptiveLoopResult:
        record = candidates.get(selected) if selected else None
        return AdaptiveLoopResult(
            status=status,
            rounds=rounds,
            selected_candidate_node_id=selected,
            selected_candidate_hash=record.artifact.candidate_hash if record else None,
            plans=tuple(plans.values()),
            candidates=tuple(candidates.values()),
            evidence=tuple(evidence.values()),
            knowledge=ledger.snapshots(),
            decisions=tuple(decisions),
            round_events=tuple(round_events),
            role_errors=tuple(role_errors),
            resource_budget=self.resource_budget,
            resource_usage=self._resource_usage(),
        )

    def _resource_usage(self) -> Dict[str, int]:
        observed = self.roles.resource_usage
        return {
            key: max(int(observed.get(key, 0)), 0)
            for key in {"total_tokens", "llm_calls", "verifier_calls"}
            if key in self.resource_budget or key in observed
        }

    def _resource_remaining(self) -> Dict[str, int]:
        usage = self._resource_usage()
        return {
            key: max(limit - usage.get(key, 0), 0)
            for key, limit in self.resource_budget.items()
        }

    def _budget_terminal_status(self) -> Optional[str]:
        usage = self._resource_usage()
        if any(usage.get(key, 0) > limit for key, limit in self.resource_budget.items()):
            return "resource_overrun"
        if any(usage.get(key, 0) >= limit for key, limit in self.resource_budget.items()):
            return "resource_limit"
        return None

    def _decision_token_reservation(self, decision: DirectorDecision) -> int:
        history = self.roles.resource_cost_history
        raw = history.get("minimum_completion_reservation_total_tokens", {})
        if not isinstance(raw, Mapping):
            return 0

        def cost(name: str) -> int:
            value = raw.get(name, 0)
            return int(value) if isinstance(value, (int, float)) and value > 0 else 0

        total = 0
        for action in decision.dispatch:
            if action.kind.produces_candidate:
                total += cost("candidate_action")
                if action.source_node_ids and action.kind in {
                    AdaptiveActionKind.REFINE,
                    AdaptiveActionKind.FORK,
                    AdaptiveActionKind.FUSE,
                }:
                    total += cost("candidate_judge_action")
            elif action.kind.invokes_planner:
                total += cost("planner_action")
            elif action.kind is AdaptiveActionKind.DISCUSS:
                total += cost("discuss_action")
        return total

    @staticmethod
    def _passed_candidate_id(
        candidates: Mapping[str, CandidateRecord],
        evidence: Mapping[str, EvidenceRecord],
    ) -> Optional[str]:
        passed = sorted(
            item.candidate_node_id for item in evidence.values() if item.evidence.passed
        )
        return passed[0] if passed and passed[0] in candidates else None

    @staticmethod
    def _lineage_diagnostics(
        candidates: Mapping[str, CandidateRecord],
        evidence: Mapping[str, EvidenceRecord],
    ) -> Tuple[Mapping[str, Any], ...]:
        """Summarize recurrence without encoding a scripted stagnation threshold."""

        def lineage(node_id: str) -> List[CandidateRecord]:
            seen: set[str] = set()
            frontier = [node_id]
            records: List[CandidateRecord] = []
            while frontier:
                current = frontier.pop()
                if current in seen or current not in candidates:
                    continue
                seen.add(current)
                record = candidates[current]
                records.append(record)
                frontier.extend(record.parent_node_ids)
            return sorted(records, key=lambda item: (item.round_created, item.node_id))

        diagnostics: List[Mapping[str, Any]] = []
        for node_id in sorted(candidates):
            records = lineage(node_id)
            trajectory = []
            normalized_signatures = []
            for record in records:
                if not record.evidence_id or record.evidence_id not in evidence:
                    continue
                item = evidence[record.evidence_id].evidence
                normalized = re.sub(r"\s+", " ", item.failure_signature).strip().lower()
                trajectory.append(
                    {
                        "candidate_node_id": record.node_id,
                        "eda_stage": item.stage.name,
                        "failure_class": item.failure_class,
                        "failure_signature": item.failure_signature,
                    }
                )
                normalized_signatures.append(normalized)
            current_signature = normalized_signatures[-1] if normalized_signatures else ""
            consecutive = 0
            for signature in reversed(normalized_signatures):
                if signature != current_signature:
                    break
                consecutive += 1
            diagnostics.append(
                {
                    "candidate_node_id": node_id,
                    "lineage_length": len(records),
                    "evidence_trajectory": trajectory,
                    "current_failure_signature_occurrences": (
                        normalized_signatures.count(current_signature)
                        if current_signature
                        else 0
                    ),
                    "consecutive_current_failure_signature": consecutive,
                    "interpretation": (
                        "Recurrence is diagnostic evidence only; decide whether the next "
                        "obligation is genuinely discriminating or whether to fork, review, "
                        "replan, discuss, or backtrack."
                    ),
                }
            )
        return tuple(diagnostics)

    def _checkpoint(self, result: AdaptiveLoopResult) -> None:
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.checkpoint_dir / "adaptive_state.json", result.to_dict())
