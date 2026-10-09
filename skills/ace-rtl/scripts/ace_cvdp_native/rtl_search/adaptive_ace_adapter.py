"""ACE language-model adapters for the skill-defined adaptive controller.

All semantic choices remain model outputs.  This module performs only prompt
assembly, strict schema parsing, target-file normalization, and durable call
bookkeeping.
"""

from __future__ import annotations

import difflib
import json
import re
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from extract_solution import extract_code, split_target_files

from ..config import classify_task, system_prompt_for_task
from ..processes import write_json
from ..sv import normalize_generated_sv
from .ace_adapter import (
    CallLedger,
    ResourceBudgetExceeded,
    _artifact_text,
    _bounded,
    _context_text,
    _json_object,
)
from .adaptive_contracts import (
    ActionCard,
    AdaptiveActionKind,
    BranchReview,
    CandidateJudgement,
    DebateOutcome,
    DirectorDecision,
    KnowledgeDecision,
    KnowledgeProposal,
    SearchSnapshot,
    TaskAssessment,
)
from .contracts import CandidateArtifact, PlanHypothesis, TaskContract
from .node import EDAEvidence


_PLAN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")
_KNOWLEDGE_TYPES = (
    "spec_constraint",
    "observed_fact",
    "root_cause",
    "fix_pattern",
    "rejected_hypothesis",
    "conflict",
    "reusable_artifact",
    "open_question",
)
_KNOWLEDGE_SCOPES = ("global", "plan_family", "module", "candidate_lineage")


def _strings(value: Any, field: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    return result


def _parse_plan(item: Any, *, fallback_id: str) -> PlanHypothesis:
    if not isinstance(item, dict):
        raise ValueError("plan must be a JSON object")
    raw_id = str(item.get("plan_id", "")).strip()
    plan_id = raw_id if _PLAN_ID_RE.fullmatch(raw_id) else fallback_id
    summary = str(item.get("summary", "")).strip()
    assumption = str(item.get("distinguishing_assumption", "")).strip()
    invariants = _strings(item.get("invariants", []), "invariants")
    if not summary or not assumption:
        raise ValueError("plan requires summary and distinguishing_assumption")
    return PlanHypothesis(plan_id, summary, assumption, invariants)


def _plan_dict(plan: PlanHypothesis) -> Dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "summary": plan.summary,
        "distinguishing_assumption": plan.distinguishing_assumption,
        "invariants": list(plan.invariants),
    }


def _relevant_knowledge(snapshot: SearchSnapshot, action: ActionCard) -> Tuple[dict, ...]:
    selected = []
    requested = set(action.knowledge_refs)
    for raw in snapshot.knowledge:
        item = dict(raw)
        if requested and item.get("item_id") in requested:
            selected.append(item)
        elif not requested and item.get("status") in {"supported", "conflicting"}:
            selected.append(item)
    return tuple(selected)


def _candidate_delta(
    candidate: CandidateArtifact,
    parents: Sequence[CandidateArtifact],
) -> str:
    if not parents:
        return _bounded(_artifact_text(candidate), 50_000)
    baseline = parents[0].as_dict()
    current = candidate.as_dict()
    chunks = []
    for name in sorted(set(baseline) | set(current)):
        before = baseline.get(name, "").splitlines(keepends=True)
        after = current.get(name, "").splitlines(keepends=True)
        chunks.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"parent/{name}",
                tofile=f"candidate/{name}",
                n=4,
            )
        )
    delta = "".join(chunks) or "No textual delta from the primary parent."
    if len(parents) > 1:
        hashes = ", ".join(parent.candidate_hash for parent in parents[1:])
        delta = f"Additional fusion parent hashes: {hashes}\n\n{delta}"
    return _bounded(delta, 50_000)


_MODULE_START_RE = re.compile(r"(?m)^\s*module\b")
_MACRO_DEFINE_RE = re.compile(r"(?m)^\s*`define\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_MACRO_REFERENCE_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)\b")


def _preserve_refine_compilation_unit_scaffold(
    action: ActionCard,
    parents: Sequence[CandidateArtifact],
    artifact: CandidateArtifact,
) -> Tuple[CandidateArtifact, Tuple[Mapping[str, Any], ...]]:
    """Restore accidentally dropped parent macro context on bounded refinement.

    This is a state-preservation invariant, not an RTL repair heuristic.  It
    activates only when a refine child still references a macro defined in the
    primary parent's pre-module compilation-unit prefix but omits that
    definition.  Seed/fork/fuse candidates and self-contained children are not
    changed.
    """

    if action.kind is not AdaptiveActionKind.REFINE or len(parents) != 1:
        return artifact, ()
    parent_files = parents[0].as_dict()
    child_files = artifact.as_dict()
    updated = dict(child_files)
    repairs = []
    for name, child_source in child_files.items():
        parent_source = parent_files.get(name)
        if parent_source is None:
            continue
        parent_module = _MODULE_START_RE.search(parent_source)
        child_module = _MODULE_START_RE.search(child_source)
        if parent_module is None or child_module is None:
            continue
        parent_prefix = parent_source[: parent_module.start()]
        if not parent_prefix.strip():
            continue
        parent_definitions = set(_MACRO_DEFINE_RE.findall(parent_prefix))
        if not parent_definitions:
            continue
        child_definitions = set(_MACRO_DEFINE_RE.findall(child_source[: child_module.start()]))
        child_references = set(_MACRO_REFERENCE_RE.findall(child_source))
        missing = sorted((parent_definitions & child_references) - child_definitions)
        if not missing:
            continue
        updated[name] = (
            parent_prefix.rstrip() + "\n\n" + child_source[child_module.start() :].lstrip()
        )
        repairs.append(
            {
                "kind": "restore_parent_compilation_unit_scaffold",
                "target_file": name,
                "restored_required_macros": missing,
                "parent_hash": parents[0].candidate_hash,
            }
        )
    if not repairs:
        return artifact, ()
    return CandidateArtifact.from_mapping(updated), tuple(repairs)


class AceAdaptivePlanner:
    def __init__(
        self,
        *,
        workspace: Path,
        model: str,
        timeout: int,
        ledger: CallLedger,
        max_tokens: int = 8192,
        reasoning_effort: str = "low",
    ) -> None:
        self.workspace = workspace
        self.model = model
        self.timeout = timeout
        self.ledger = ledger
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def propose(
        self,
        task: TaskContract,
        instruction: str,
        existing_plans: Sequence[PlanHypothesis],
        knowledge: Sequence[dict],
    ) -> Sequence[PlanHypothesis]:
        schema = {
            "adequacy_rationale": "why this number of plans matches current uncertainty",
            "plans": [
                {
                    "plan_id": "short_unique_id",
                    "summary": "implementation plan",
                    "distinguishing_assumption": "materially different assumption",
                    "invariants": ["externally observable property"],
                }
            ],
        }
        prompt = f"""{instruction}

PUBLIC SPECIFICATION:
{task.specification}

IMMUTABLE PUBLIC CONTEXT:
{_context_text(task)}

EXISTING PLANS:
{json.dumps([_plan_dict(plan) for plan in existing_plans], indent=2)}

SHARED KNOWLEDGE:
{json.dumps(list(knowledge), indent=2)}

Return strict JSON matching:
{json.dumps(schema, indent=2)}
Choose the number of plans from semantic uncertainty.  One is valid when one
interpretation clearly dominates; add plans only for materially incompatible
implementations.  Return at least one plan and no RTL or Markdown.
"""
        try:
            content = self.ledger.invoke(
                workspace=self.workspace,
                role="adaptive_planner",
                component="coordinator",
                model=self.model,
                system="You are the BW-RTL Planner. Return strict JSON and never write RTL.",
                prompt=prompt,
                timeout=self.timeout,
                temperature=0.3,
                max_tokens=self.max_tokens,
                thinking="enabled",
                reasoning_effort=self.reasoning_effort,
                response_format="json_object",
            )
            return self._parse_plan_set(content, len(existing_plans))
        except ResourceBudgetExceeded:
            raise
        except Exception as primary_error:
            self.ledger.mark_last_invalid(primary_error)
            primary_error_text = str(primary_error)

        retry_prompt = f"""{prompt}

YOUR PREVIOUS RESPONSE DID NOT PRODUCE A VALID FINAL PLAN OBJECT:
{primary_error_text}

Return the compact final JSON now.  Do not reason aloud, add Markdown, or omit
the plans list.  Preserve semantic diversity; do not add cosmetic duplicates.
"""
        content = self.ledger.invoke(
            workspace=self.workspace,
            role="adaptive_planner_retry",
            component="coordinator",
            model=self.model,
            system="Return one compact valid BW-RTL Planner JSON object only.",
            prompt=retry_prompt,
            timeout=self.timeout,
            temperature=0.0,
            max_tokens=self.max_tokens,
            thinking="disabled",
            reasoning_effort=None,
            response_format="json_object",
        )
        try:
            return self._parse_plan_set(content, len(existing_plans))
        except Exception as retry_error:
            self.ledger.mark_last_invalid(retry_error)
            raise

    @staticmethod
    def _parse_plan_set(content: str, existing_count: int) -> Tuple[PlanHypothesis, ...]:
        payload = _json_object(content)
        raw_plans = payload.get("plans")
        if not isinstance(raw_plans, list) or not raw_plans:
            raise ValueError("adaptive Planner must return at least one plan")
        parsed = tuple(
            _parse_plan(item, fallback_id=f"p{existing_count + index:03d}")
            for index, item in enumerate(raw_plans, start=1)
        )
        ids = [plan.plan_id for plan in parsed]
        summaries = [plan.summary.strip().lower() for plan in parsed]
        if len(set(ids)) != len(ids) or len(set(summaries)) != len(summaries):
            raise ValueError("adaptive Planner returned duplicate plans")
        return parsed

    def revise(
        self,
        task: TaskContract,
        instruction: str,
        action: ActionCard,
        snapshot: SearchSnapshot,
    ) -> PlanHypothesis:
        prompt = f"""{instruction}

PUBLIC SPECIFICATION:
{task.specification}

IMMUTABLE PUBLIC CONTEXT:
{_context_text(task)}

DIRECTOR ACTION:
{json.dumps(action.to_dict(), indent=2)}

CURRENT PLANS AND EVIDENCE-BASED STATE:
{json.dumps(snapshot.to_prompt_dict(), indent=2)}

Return one strict JSON object with keys plan_id, summary,
distinguishing_assumption, and invariants.  The revision must be semantic and
must name a new unique plan_id.  Return no RTL or Markdown.
"""
        try:
            content = self.ledger.invoke(
                workspace=self.workspace,
                role="adaptive_planner_revision",
                component="coordinator",
                model=self.model,
                system="You are the BW-RTL Planner revising a plan. Return strict JSON only.",
                prompt=prompt,
                timeout=self.timeout,
                temperature=0.2,
                max_tokens=self.max_tokens,
                thinking="enabled",
                reasoning_effort=self.reasoning_effort,
                response_format="json_object",
            )
            return _parse_plan(
                _json_object(content), fallback_id=f"p_revision_{action.action_id}"
            )
        except ResourceBudgetExceeded:
            raise
        except Exception as primary_error:
            self.ledger.mark_last_invalid(primary_error)
            primary_error_text = str(primary_error)

        retry_prompt = f"""{prompt}

YOUR PREVIOUS RESPONSE DID NOT PRODUCE A VALID FINAL REVISED PLAN:
{primary_error_text}

Return the compact revised-plan JSON now, with no reasoning or Markdown.
"""
        content = self.ledger.invoke(
            workspace=self.workspace,
            role="adaptive_planner_revision_retry",
            component="coordinator",
            model=self.model,
            system="Return one compact valid revised-plan JSON object only.",
            prompt=retry_prompt,
            timeout=self.timeout,
            temperature=0.0,
            max_tokens=self.max_tokens,
            thinking="disabled",
            reasoning_effort=None,
            response_format="json_object",
        )
        try:
            return _parse_plan(
                _json_object(content), fallback_id=f"p_revision_{action.action_id}"
            )
        except Exception as retry_error:
            self.ledger.mark_last_invalid(retry_error)
            raise


class AceAdaptiveBuilder:
    def __init__(
        self,
        *,
        workspace: Path,
        cid: str,
        model: str,
        timeout: int,
        ledger: CallLedger,
        run_dir: Path,
        max_tokens: int = 32000,
        reasoning_effort: str = "low",
    ) -> None:
        self.workspace = workspace
        self.cid = cid
        self.model = model
        self.timeout = timeout
        self.ledger = ledger
        self.run_dir = run_dir
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.candidates: Dict[str, CandidateArtifact] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def build(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        instruction: str,
        action: ActionCard,
        parents: Sequence[CandidateArtifact],
        snapshot: SearchSnapshot,
    ) -> CandidateArtifact:
        with self._lock:
            self._calls += 1
            call_index = self._calls
        targets = "\n".join(f"- {path}" for path in task.target_files)
        output_contract = (
            "Return one complete section per target headed exactly by "
            "`// TARGET_FILE: <path>`; include every target and no other file."
            if len(task.target_files) > 1
            else f"Return the complete raw source for {task.target_files[0]} only."
        )
        parent_text = "\n\n".join(
            f"# SOURCE {index} hash={parent.candidate_hash}\n{_artifact_text(parent)}"
            for index, parent in enumerate(parents, start=1)
        ) or "No parent candidate exists."
        prompt = f"""{instruction}

PUBLIC SPECIFICATION:
{task.specification}

IMMUTABLE PUBLIC CONTEXT:
{_context_text(task)}

SELECTED PLAN:
{json.dumps(_plan_dict(plan), indent=2)}

DIRECTOR ACTION:
{json.dumps(action.to_dict(), indent=2)}

RELEVANT SHARED KNOWLEDGE:
{json.dumps(_relevant_knowledge(snapshot, action), indent=2)}

SOURCE CANDIDATES:
{_bounded(parent_text)}

DECLARED TARGETS:
{targets}

OUTPUT CONTRACT:
{output_contract}
Do not include analysis, JSON, a patch, tests, or Markdown fences.
"""
        primary = task.target_files[0]
        task_kind = classify_task(primary, task.specification)
        content = self.ledger.invoke(
            workspace=self.workspace,
            role="adaptive_builder",
            component="generator",
            model=self.model,
            system=system_prompt_for_task(task_kind, multi_file=len(task.target_files) > 1),
            prompt=prompt,
            timeout=self.timeout,
            temperature=0.2,
            max_tokens=self.max_tokens,
            thinking="enabled",
            reasoning_effort=self.reasoning_effort,
            response_format="text",
        )
        extracted = extract_code(content)
        if len(task.target_files) == 1:
            raw_files: Mapping[str, str] = {primary: extracted}
        else:
            sections = split_target_files(extracted)
            raw_files = sections if sections else {primary: extracted}
        normalized: Dict[str, str] = {}
        for name, raw_code in raw_files.items():
            normalized_name = str(PurePosixPath(str(name).strip()))
            path = PurePosixPath(normalized_name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Builder returned unsafe target path: {name}")
            code, _actions = normalize_generated_sv(str(raw_code))
            normalized[normalized_name] = code
        artifact = CandidateArtifact.from_mapping(normalized)
        artifact, scaffold_repairs = _preserve_refine_compilation_unit_scaffold(
            action, parents, artifact
        )
        candidate_dir = (
            self.run_dir
            / "adaptive_candidates"
            / f"builder_{call_index:03d}_{artifact.candidate_hash}"
        )
        for name, code in artifact.files:
            output_path = candidate_dir.joinpath(*PurePosixPath(name).parts)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code, encoding="utf-8")
        write_json(
            candidate_dir / "candidate.json",
            {
                "candidate_hash": artifact.candidate_hash,
                "action": action.to_dict(),
                "plan_id": plan.plan_id,
                "parent_hashes": [parent.candidate_hash for parent in parents],
                "files": [name for name, _ in artifact.files],
                "contract_issues": list(artifact.contract_issues(task)),
                "deterministic_state_preservation": list(scaffold_repairs),
            },
        )
        with self._lock:
            self.candidates[artifact.candidate_hash] = artifact
        return artifact


class AceAdaptiveCandidateJudge:
    """Fail-closed action-fidelity review before spending an evaluator call."""

    def __init__(
        self,
        *,
        workspace: Path,
        model: str,
        timeout: int,
        ledger: CallLedger,
        max_tokens: int = 4096,
        reasoning_effort: str = "low",
    ) -> None:
        self.workspace = workspace
        self.model = model
        self.timeout = timeout
        self.ledger = ledger
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def review_candidate(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        action: ActionCard,
        parents: Sequence[CandidateArtifact],
        candidate: CandidateArtifact,
        instruction: str,
        snapshot: SearchSnapshot,
    ) -> CandidateJudgement:
        parent_ids = set(action.source_node_ids)
        parent_evidence_ids = {
            record.evidence_id
            for record in snapshot.candidates
            if record.node_id in parent_ids and record.evidence_id
        }
        parent_evidence = [
            item.to_dict()
            for item in snapshot.evidence
            if item.evidence_id in parent_evidence_ids
        ]
        schema = {
            "accepted": "boolean",
            "rationale": "evidence-grounded accept/reject rationale",
            "action_fidelity": "does actual delta implement the obligation",
            "semantic_novelty": "which behavior changes versus coding form only",
            "state_preservation": "working parent state preserved or regressed",
            "rtl_risk": "obvious declaration/driver/assignment/syntax risks",
            "revision_requirements": ["concrete requirement; no replacement RTL"],
        }
        prompt = f"""{instruction}

TASK REFERENCE:
- task_id: {task.task_id}
- target_files: {json.dumps(task.target_files)}

SELECTED PLAN:
{json.dumps(_plan_dict(plan), indent=2)}

DIRECTOR ACTION:
{json.dumps(action.to_dict(), indent=2)}

PARENT EDA EVIDENCE:
{json.dumps(parent_evidence, indent=2)}

CANDIDATE DELTA FROM PRIMARY PARENT:
{_candidate_delta(candidate, parents)}

RELEVANT SHARED KNOWLEDGE:
{json.dumps(_relevant_knowledge(snapshot, action), indent=2)}

Return strict JSON matching:
{json.dumps(schema, indent=2)}
Judge action fidelity and evaluator worth, not hidden correctness.  Reject if
the delta is broader than the claimed discriminator or introduces an obvious
structural regression.  Return no RTL or Markdown.
"""
        try:
            content = self.ledger.invoke(
                workspace=self.workspace,
                role="adaptive_candidate_judge",
                component="reflector",
                model=self.model,
                system=(
                    "You are the BW-RTL Candidate Judge. Return strict JSON and "
                    "never write replacement RTL."
                ),
                prompt=prompt,
                timeout=self.timeout,
                temperature=0.0,
                max_tokens=self.max_tokens,
                thinking="enabled",
                reasoning_effort=self.reasoning_effort,
                response_format="json_object",
            )
            return self._parse(content)
        except ResourceBudgetExceeded:
            raise
        except Exception as primary_error:
            self.ledger.mark_last_invalid(primary_error)
            primary_error_text = str(primary_error)

        repair_prompt = f"""{prompt}

YOUR PREVIOUS RESPONSE WAS INVALID:
{primary_error_text}

Return one compact valid judgement JSON now.  Do not add reasoning or RTL.
"""
        content = self.ledger.invoke(
            workspace=self.workspace,
            role="adaptive_candidate_judge_retry",
            component="reflector",
            model=self.model,
            system="Return one compact valid Candidate Judge JSON object only.",
            prompt=repair_prompt,
            timeout=self.timeout,
            temperature=0.0,
            max_tokens=self.max_tokens,
            thinking="disabled",
            reasoning_effort=None,
            response_format="json_object",
        )
        try:
            return self._parse(content)
        except Exception as retry_error:
            self.ledger.mark_last_invalid(retry_error)
            raise

    @staticmethod
    def _parse(content: str) -> CandidateJudgement:
        payload = _json_object(content)
        accepted = payload.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError("Candidate Judge accepted must be a JSON boolean")
        requirements = _strings(
            payload.get("revision_requirements", []), "revision_requirements"
        )
        if not accepted and not requirements:
            raise ValueError("rejected candidate judgement requires revision_requirements")
        return CandidateJudgement(
            accepted=accepted,
            rationale=str(payload.get("rationale", "")).strip(),
            action_fidelity=str(payload.get("action_fidelity", "")).strip(),
            semantic_novelty=str(payload.get("semantic_novelty", "")).strip(),
            state_preservation=str(payload.get("state_preservation", "")).strip(),
            rtl_risk=str(payload.get("rtl_risk", "")).strip(),
            revision_requirements=requirements,
        )


class AceAdaptiveAnalyst:
    def __init__(
        self,
        *,
        workspace: Path,
        model: str,
        timeout: int,
        ledger: CallLedger,
        max_tokens: int = 6144,
        reasoning_effort: str = "low",
    ) -> None:
        self.workspace = workspace
        self.model = model
        self.timeout = timeout
        self.ledger = ledger
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

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
    ) -> BranchReview:
        parents = tuple(
            record.artifact
            for record in snapshot.candidates
            if record.node_id in action.source_node_ids
        )
        schema = {
            "strengths": ["evidence-supported strength"],
            "weaknesses": ["evidence-supported weakness"],
            "observations": ["direct EDA observation"],
            "root_causes": ["inferred cause"],
            "invalidated_assumptions": ["falsified assumption"],
            "open_questions": ["unresolved question"],
            "recommended_actions": ["discriminating next action"],
            "knowledge_proposals": [
                {
                    "proposal_id": "local id",
                    "item_type": "one of: " + "|".join(_KNOWLEDGE_TYPES),
                    "scope": "one of: " + "|".join(_KNOWLEDGE_SCOPES),
                    "scope_ref": candidate_node_id,
                    "claim": "concise reusable claim",
                    "evidence_refs": [evidence_id],
                    "confidence": "low|medium|high",
                    "conflicts_with": [],
                }
            ],
        }
        evidence_payload = {
            "evidence_id": evidence_id,
            "stage": evidence.original_stage,
            "failure_class": evidence.failure_class,
            "failure_signature": evidence.failure_signature,
            "complete_bundle": evidence.complete_bundle,
            "passed_subtests": evidence.passed_subtests,
            "total_subtests": evidence.total_subtests,
            "public_feedback": _bounded(evidence.public_feedback, 6000),
        }
        prompt = f"""{instruction}

TASK REFERENCE:
- task_id: {task.task_id}
- target_files: {json.dumps(task.target_files)}
The Builder was bound to the immutable public specification.  Diagnose this
branch from the selected plan invariants, action obligation, code delta, and EDA
evidence; request `review_spec` if those facts are insufficient.

SELECTED PLAN:
{json.dumps(_plan_dict(plan), indent=2)}

DIRECTOR ACTION:
{json.dumps(action.to_dict(), indent=2)}

CANDIDATE DELTA FROM PRIMARY PARENT (full candidate only for a root seed):
{_candidate_delta(candidate, parents)}

EDA EVIDENCE:
{json.dumps(evidence_payload, indent=2)}

RELEVANT SHARED KNOWLEDGE:
{json.dumps(_relevant_knowledge(snapshot, action), indent=2)}

Return strict JSON matching:
{json.dumps(schema, indent=2)}
Use only supplied evidence IDs.  item_type must be exactly one of
{json.dumps(_KNOWLEDGE_TYPES)} and scope exactly one of
{json.dumps(_KNOWLEDGE_SCOPES)}.  Do not invent taxonomy labels.  Do not emit
RTL or Markdown.
"""
        try:
            content = self.ledger.invoke(
                workspace=self.workspace,
                role="adaptive_analyst",
                component="reflector",
                model=self.model,
                system=(
                    "You are the BW-RTL branch Analyst. Diagnose evidence and "
                    "return strict JSON."
                ),
                prompt=prompt,
                timeout=self.timeout,
                temperature=0.1,
                max_tokens=self.max_tokens,
                thinking="enabled",
                reasoning_effort=self.reasoning_effort,
                response_format="json_object",
            )
            return self._parse_review(content, candidate_node_id, evidence_id)
        except ResourceBudgetExceeded:
            raise
        except Exception as primary_error:
            self.ledger.mark_last_invalid(primary_error)
            primary_error_text = str(primary_error)

        retry_prompt = f"""The previous Analyst response was rejected:
{primary_error_text}

Return one compact final diagnosis JSON. Use only these bounded public facts:
- candidate_node_id: {candidate_node_id}
- evidence_id: {evidence_id}
- plan: {plan.plan_id} — {plan.summary}
- action: {json.dumps(action.to_dict())}
- EDA evidence: {json.dumps(evidence_payload)}
- candidate delta: {_bounded(_candidate_delta(candidate, parents), 12000)}

Required list keys: strengths, weaknesses, observations, root_causes,
invalidated_assumptions, open_questions, recommended_actions,
knowledge_proposals. Each knowledge proposal must use item_type from
{json.dumps(_KNOWLEDGE_TYPES)}, scope from {json.dumps(_KNOWLEDGE_SCOPES)}, and
only evidence id {evidence_id}. Return JSON only, with no RTL or Markdown.
"""
        content = self.ledger.invoke(
            workspace=self.workspace,
            role="adaptive_analyst_retry",
            component="reflector",
            model=self.model,
            system="Return one compact evidence-grounded BW-RTL Analyst JSON object only.",
            prompt=retry_prompt,
            timeout=self.timeout,
            temperature=0.0,
            max_tokens=min(self.max_tokens, 2048),
            thinking="disabled",
            reasoning_effort=None,
            response_format="json_object",
        )
        try:
            return self._parse_review(content, candidate_node_id, evidence_id)
        except Exception as retry_error:
            self.ledger.mark_last_invalid(retry_error)
            raise

    @staticmethod
    def _parse_review(
        content: str,
        candidate_node_id: str,
        evidence_id: str,
    ) -> BranchReview:
        payload = _json_object(content)
        raw_proposals = payload.get("knowledge_proposals", [])
        if not isinstance(raw_proposals, list):
            raise ValueError("knowledge_proposals must be a JSON list")
        proposals = []
        for index, raw in enumerate(raw_proposals, start=1):
            if not isinstance(raw, dict):
                raise ValueError("knowledge proposal must be a JSON object")
            proposal = KnowledgeProposal(
                proposal_id=str(raw.get("proposal_id") or f"{candidate_node_id}-{index}"),
                item_type=str(raw.get("item_type", "")).strip(),
                scope=str(raw.get("scope", "")).strip(),
                scope_ref=(str(raw["scope_ref"]).strip() if raw.get("scope_ref") else None),
                claim=str(raw.get("claim", "")).strip(),
                evidence_refs=_strings(raw.get("evidence_refs", []), "evidence_refs"),
                confidence=str(raw.get("confidence", "unspecified")).strip(),
                conflicts_with=_strings(raw.get("conflicts_with", []), "conflicts_with"),
            )
            proposals.append(proposal)
        return BranchReview(
            candidate_node_id=candidate_node_id,
            evidence_id=evidence_id,
            strengths=_strings(payload.get("strengths", []), "strengths"),
            weaknesses=_strings(payload.get("weaknesses", []), "weaknesses"),
            observations=_strings(payload.get("observations", []), "observations"),
            root_causes=_strings(payload.get("root_causes", []), "root_causes"),
            invalidated_assumptions=_strings(
                payload.get("invalidated_assumptions", []), "invalidated_assumptions"
            ),
            open_questions=_strings(payload.get("open_questions", []), "open_questions"),
            recommended_actions=_strings(
                payload.get("recommended_actions", []), "recommended_actions"
            ),
            knowledge_proposals=tuple(proposals),
        )


class AceAdaptiveDebater:
    """Compare selected branches without editing code or owning scheduling."""

    def __init__(
        self,
        *,
        workspace: Path,
        model: str,
        timeout: int,
        ledger: CallLedger,
        max_tokens: int = 6144,
        reasoning_effort: str = "low",
    ) -> None:
        self.workspace = workspace
        self.model = model
        self.timeout = timeout
        self.ledger = ledger
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def discuss(
        self,
        task: TaskContract,
        instruction: str,
        action: ActionCard,
        snapshot: SearchSnapshot,
    ) -> DebateOutcome:
        selected_ids = set(action.source_node_ids)
        branch_payload = [
            candidate.to_summary()
            for candidate in snapshot.candidates
            if candidate.node_id in selected_ids
        ]
        evidence_ids = {
            candidate.evidence_id
            for candidate in snapshot.candidates
            if candidate.node_id in selected_ids and candidate.evidence_id
        }
        evidence_payload = [
            item.to_dict() for item in snapshot.evidence if item.evidence_id in evidence_ids
        ]
        schema = {
            "summary": "fair comparison",
            "branch_advantages": ["branch id: evidence-grounded advantage"],
            "branch_risks": ["branch id: evidence-grounded risk"],
            "disagreements": ["unresolved incompatible claim"],
            "recommended_discriminators": ["observation that separates the claims"],
            "knowledge_proposals": [
                {
                    "proposal_id": "local id",
                    "item_type": "one of: " + "|".join(_KNOWLEDGE_TYPES),
                    "scope": "one of: " + "|".join(_KNOWLEDGE_SCOPES),
                    "scope_ref": None,
                    "claim": "concise comparison claim",
                    "evidence_refs": list(sorted(evidence_ids)),
                    "confidence": "low|medium|high",
                    "conflicts_with": [],
                }
            ],
        }
        prompt = f"""{instruction}

TASK REFERENCE:
- task_id: {task.task_id}
- target_files: {json.dumps(task.target_files)}

DIRECTOR DISCUSS ACTION:
{json.dumps(action.to_dict(), indent=2)}

SELECTED BRANCH SUMMARIES:
{json.dumps(branch_payload, indent=2)}

SELECTED PUBLIC EDA EVIDENCE:
{json.dumps(evidence_payload, indent=2)}

CURRENT SHARED KNOWLEDGE:
{json.dumps(snapshot.knowledge, indent=2)}

Return strict JSON matching:
{json.dumps(schema, indent=2)}
Represent competing branches fairly.  Do not decide scheduling, edit RTL, or
invent evidence.  Use only supplied evidence and knowledge IDs. item_type must
be exactly one of {json.dumps(_KNOWLEDGE_TYPES)} and scope exactly one of
{json.dumps(_KNOWLEDGE_SCOPES)}; do not invent taxonomy labels.
"""
        content = self.ledger.invoke(
            workspace=self.workspace,
            role="adaptive_debater",
            component="reflector",
            model=self.model,
            system="You are the BW-RTL Debater. Compare named branches and return strict JSON.",
            prompt=prompt,
            timeout=self.timeout,
            temperature=0.2,
            max_tokens=self.max_tokens,
            thinking="enabled",
            reasoning_effort=self.reasoning_effort,
            response_format="json_object",
        )
        payload = _json_object(content)
        raw_proposals = payload.get("knowledge_proposals", [])
        if not isinstance(raw_proposals, list):
            raise ValueError("knowledge_proposals must be a JSON list")
        proposals = []
        for index, raw in enumerate(raw_proposals, start=1):
            if not isinstance(raw, dict):
                raise ValueError("knowledge proposal must be a JSON object")
            proposals.append(
                KnowledgeProposal(
                    proposal_id=str(raw.get("proposal_id") or f"{action.action_id}-{index}"),
                    item_type=str(raw.get("item_type", "")).strip(),
                    scope=str(raw.get("scope", "")).strip(),
                    scope_ref=(str(raw["scope_ref"]).strip() if raw.get("scope_ref") else None),
                    claim=str(raw.get("claim", "")).strip(),
                    evidence_refs=_strings(raw.get("evidence_refs", []), "evidence_refs"),
                    confidence=str(raw.get("confidence", "unspecified")).strip(),
                    conflicts_with=_strings(raw.get("conflicts_with", []), "conflicts_with"),
                )
            )
        return DebateOutcome(
            summary=str(payload.get("summary", "")).strip(),
            branch_advantages=_strings(
                payload.get("branch_advantages", []), "branch_advantages"
            ),
            branch_risks=_strings(payload.get("branch_risks", []), "branch_risks"),
            disagreements=_strings(payload.get("disagreements", []), "disagreements"),
            recommended_discriminators=_strings(
                payload.get("recommended_discriminators", []),
                "recommended_discriminators",
            ),
            knowledge_proposals=tuple(proposals),
        )


def _director_decision_from_content(content: str) -> DirectorDecision:
    payload = _json_object(content)
    raw_assessment = payload.get("assessment")
    if not isinstance(raw_assessment, dict):
        raise ValueError("Director assessment must be an object")
    assessment = TaskAssessment(
        spec_certainty=str(raw_assessment.get("spec_certainty", "")).strip(),
        plan_concentration=str(raw_assessment.get("plan_concentration", "")).strip(),
        implementation_risk=str(raw_assessment.get("implementation_risk", "")).strip(),
        verification_observability=str(
            raw_assessment.get("verification_observability", "")
        ).strip(),
        rationale=str(raw_assessment.get("rationale", "")).strip(),
    )
    raw_dispatch = payload.get("dispatch")
    if not isinstance(raw_dispatch, list) or not raw_dispatch:
        raise ValueError("Director dispatch must be a non-empty list")
    dispatch = []
    for raw in raw_dispatch:
        if not isinstance(raw, dict):
            raise ValueError("Director action must be a JSON object")
        try:
            kind = AdaptiveActionKind(str(raw.get("kind", "")).strip())
        except ValueError as exc:
            raise ValueError(f"Director returned an unknown action: {raw.get('kind')}") from exc
        dispatch.append(
            ActionCard(
                action_id=str(raw.get("action_id", "")).strip(),
                kind=kind,
                source_node_ids=_strings(raw.get("source_node_ids", []), "source_node_ids"),
                plan_id=(str(raw["plan_id"]).strip() if raw.get("plan_id") else None),
                hypothesis=str(raw.get("hypothesis", "")).strip(),
                obligation=str(raw.get("obligation", "")).strip(),
                semantic_delta=str(raw.get("semantic_delta", "")).strip(),
                expected_observation=str(raw.get("expected_observation", "")).strip(),
                falsifier=str(raw.get("falsifier", "")).strip(),
                evidence_refs=_strings(raw.get("evidence_refs", []), "evidence_refs"),
                knowledge_refs=_strings(raw.get("knowledge_refs", []), "knowledge_refs"),
            )
        )
    raw_knowledge = payload.get("knowledge_decisions", [])
    if not isinstance(raw_knowledge, list):
        raise ValueError("knowledge_decisions must be a JSON list")
    knowledge_decisions = []
    for raw in raw_knowledge:
        if not isinstance(raw, dict):
            raise ValueError("knowledge decision must be a JSON object")
        knowledge_decisions.append(
            KnowledgeDecision(
                item_id=str(raw.get("item_id", "")).strip(),
                status=str(raw.get("status", "")).strip(),
                rationale=str(raw.get("rationale", "")).strip(),
            )
        )
    return DirectorDecision(
        assessment=assessment,
        rationale=str(payload.get("rationale", "")).strip(),
        dispatch=tuple(dispatch),
        knowledge_decisions=tuple(knowledge_decisions),
    )


def _director_contract_error(
    decision: DirectorDecision,
    snapshot: SearchSnapshot,
) -> Optional[ValueError]:
    try:
        if not decision.dispatch:
            raise ValueError("Director dispatch cannot be empty; use an explicit stop action")
        available = set(snapshot.available_actions)
        unavailable = sorted(
            {
                action.kind.value
                for action in decision.dispatch
                if action.kind.value not in available
            }
        )
        if unavailable:
            raise ValueError(
                "Director selected unavailable action(s): "
                + ", ".join(unavailable)
                + "; available_actions="
                + ", ".join(snapshot.available_actions)
            )
        if any(action.kind is AdaptiveActionKind.STOP for action in decision.dispatch):
            if len(decision.dispatch) != 1:
                raise ValueError("stop must be the only action in its round")
        verifier_remaining = snapshot.resource_remaining.get("verifier_calls")
        candidate_actions = sum(action.kind.produces_candidate for action in decision.dispatch)
        if verifier_remaining is not None and candidate_actions > verifier_remaining:
            raise ValueError(
                "dispatch requires more verifier calls than remain in the resource "
                f"envelope: requested={candidate_actions}, remaining={verifier_remaining}"
            )
        if (
            snapshot.max_rounds is not None
            and snapshot.round_index == snapshot.max_rounds
            and not any(
                action.kind.produces_candidate or action.kind is AdaptiveActionKind.STOP
                for action in decision.dispatch
            )
        ):
            raise ValueError(
                "final-round dispatch must produce verifier evidence or stop; "
                "review/replan/state-only work cannot be consumed after the safety horizon"
            )

        plans = {plan.plan_id for plan in snapshot.plans}
        candidates = {candidate.node_id: candidate for candidate in snapshot.candidates}
        evidence = {item.evidence_id: item for item in snapshot.evidence}
        knowledge = {str(item.get("item_id")) for item in snapshot.knowledge}
        semantic_keys = set()
        for action in decision.dispatch:
            kind = action.kind
            key = (
                kind.value,
                action.plan_id,
                action.source_node_ids,
                action.hypothesis.strip().lower(),
                action.obligation.strip().lower(),
            )
            if key in semantic_keys:
                raise ValueError(f"duplicate semantic action in dispatch: {action.action_id}")
            semantic_keys.add(key)
            if kind.produces_candidate and (not action.plan_id or action.plan_id not in plans):
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
                        raise ValueError(
                            "a correctness stop requires fail-closed passing evidence"
                        )
                elif any(item.evidence.passed for item in evidence.values()):
                    raise ValueError(
                        "source-free stop cannot ignore fail-closed passing evidence; "
                        "cite exactly one passed candidate"
                    )

            for node_id in action.source_node_ids:
                if kind is AdaptiveActionKind.BACKTRACK and node_id in plans:
                    continue
                if node_id not in candidates:
                    raise ValueError(
                        f"action {action.action_id} references unknown node {node_id}"
                    )
            for reference in action.evidence_refs:
                if reference not in evidence:
                    raise ValueError(
                        f"action {action.action_id} references unknown evidence {reference}"
                    )
            for reference in action.knowledge_refs:
                if reference not in knowledge:
                    raise ValueError(
                        f"action {action.action_id} references unknown knowledge {reference}"
                    )
        for item in decision.knowledge_decisions:
            if item.item_id not in knowledge:
                raise ValueError(
                    f"knowledge decision references unknown item {item.item_id}"
                )
            if item.status not in {"supported", "refuted", "conflicting", "superseded"}:
                raise ValueError(f"invalid knowledge decision status: {item.status}")
    except ValueError as error:
        return error
    else:
        return None


class AceAdaptiveDirector:
    def __init__(
        self,
        *,
        workspace: Path,
        model: str,
        timeout: int,
        ledger: CallLedger,
        max_tokens: int = 8192,
        reasoning_effort: str = "low",
    ) -> None:
        self.workspace = workspace
        self.model = model
        self.timeout = timeout
        self.ledger = ledger
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def decide(
        self,
        task: TaskContract,
        instruction: str,
        snapshot: SearchSnapshot,
    ) -> DirectorDecision:
        schema = {
            "assessment": {
                "spec_certainty": "assessment",
                "plan_concentration": "assessment",
                "implementation_risk": "assessment",
                "verification_observability": "assessment",
                "rationale": "evidence-grounded rationale",
            },
            "rationale": "global comparison and dispatch rationale",
            "dispatch": [
                {
                    "action_id": "unique within round",
                    "kind": "|".join(snapshot.available_actions),
                    "source_node_ids": [],
                    "plan_id": None,
                    "hypothesis": "semantic hypothesis",
                    "obligation": "bounded work",
                    "semantic_delta": (
                        "for refine/fork/fuse: exact input/state behavior changed from source"
                    ),
                    "expected_observation": "what should change",
                    "falsifier": "what rejects it",
                    "evidence_refs": [],
                    "knowledge_refs": [],
                }
            ],
            "knowledge_decisions": [
                {
                    "item_id": "existing knowledge id",
                    "status": "supported|refuted|conflicting|superseded",
                    "rationale": "evidence-grounded reason",
                }
            ],
        }
        prompt = f"""{instruction}

RUNTIME LEGAL ACTIONS (exact, capability list in the skill is only a superset):
{json.dumps(snapshot.available_actions)}

PUBLIC SPECIFICATION:
{task.specification}

GLOBAL SEARCH SNAPSHOT:
{json.dumps(snapshot.to_prompt_dict(), indent=2)}

Return strict JSON matching:
{json.dumps(schema, indent=2)}
The dispatch length is your semantic decision.  Do not fill worker slots merely
because they exist.  Use only available_actions and existing IDs. A stop action
must be the only dispatch item. STOP with one source means correctness delivery
and that source must have fail-closed PASS evidence. STOP with zero sources means
an unsuccessful fail-closed termination and is legal only when no passed candidate
exists. On the final round, review_spec/replan/state-only actions cannot affect a
later round: produce a verifiable candidate or stop. Return no RTL or Markdown.
"""
        try:
            content = self.ledger.invoke(
                workspace=self.workspace,
                role="adaptive_director",
                component="coordinator",
                model=self.model,
                system="You are RTL-Director, the global BW-RTL search controller. Return strict JSON.",
                prompt=prompt,
                timeout=self.timeout,
                temperature=0.2,
                max_tokens=self.max_tokens,
                thinking="enabled",
                reasoning_effort=self.reasoning_effort,
                response_format="json_object",
            )
            decision = _director_decision_from_content(content)
            contract_error = _director_contract_error(decision, snapshot)
            if contract_error is None:
                return decision
            raise contract_error
        except ResourceBudgetExceeded:
            raise
        except Exception as primary_error:
            self.ledger.mark_last_invalid(primary_error)
            primary_error_text = str(primary_error)

        repair_prompt = f"""{prompt}

YOUR PREVIOUS RESPONSE WAS REJECTED:
{primary_error_text}

Return one corrected full decision satisfying every action/source/evidence rule,
using only these available actions:
{json.dumps(snapshot.available_actions)}
Do not defend or repeat the invalid decision. Return strict JSON only.
"""
        repaired_content = ""
        try:
            repaired_content = self.ledger.invoke(
                workspace=self.workspace,
                role="adaptive_director_retry",
                component="coordinator",
                model=self.model,
                system=(
                    "You are RTL-Director repairing a rejected controller decision. "
                    "Return one compact strict JSON decision satisfying the supplied contract."
                ),
                prompt=repair_prompt,
                timeout=self.timeout,
                temperature=0.1,
                max_tokens=self.max_tokens,
                thinking="disabled",
                reasoning_effort=None,
                response_format="json_object",
            )
            repaired = _director_decision_from_content(repaired_content)
            repeated_error = _director_contract_error(repaired, snapshot)
            if repeated_error is not None:
                raise repeated_error
            return repaired
        except ResourceBudgetExceeded:
            raise
        except Exception as retry_error:
            self.ledger.mark_last_invalid(retry_error)
            retry_error_text = str(retry_error)

        compact_snapshot = {
            "round_index": snapshot.round_index,
            "max_rounds": snapshot.max_rounds,
            "available_actions": snapshot.available_actions,
            "plans": [_plan_dict(plan) for plan in snapshot.plans],
            "candidates": [candidate.to_summary() for candidate in snapshot.candidates],
            "evidence": [item.to_dict() for item in snapshot.evidence],
            "knowledge_ids": [item.get("item_id") for item in snapshot.knowledge],
            "resource_remaining": dict(snapshot.resource_remaining),
            "lineage_diagnostics": list(snapshot.lineage_diagnostics),
        }
        fallback_prompt = f"""Repair a BW-RTL Director decision after two rejected responses.

FIRST REJECTION: {primary_error_text}
SECOND REJECTION: {retry_error_text}
SECOND REJECTED RESPONSE:
{_bounded(repaired_content, 8000)}

LEGAL SNAPSHOT:
{_bounded(json.dumps(compact_snapshot, indent=2), 24000)}

Return one compact full JSON decision with keys assessment, rationale, dispatch,
and knowledge_decisions. dispatch must be non-empty. Use only listed actions and
IDs. Each action requires action_id, kind, source_node_ids, plan_id, hypothesis,
obligation, semantic_delta, expected_observation, falsifier, evidence_refs, and
knowledge_refs. semantic_delta is required for refine/fork/fuse. STOP must be
the sole action; cite exactly one PASS candidate when one exists. Do not output
analysis, RTL, or Markdown.
"""
        fallback_content = self.ledger.invoke(
            workspace=self.workspace,
            role="adaptive_director_fallback",
            component="coordinator",
            model=self.model,
            system="Return one legal compact RTL-Director JSON decision only.",
            prompt=fallback_prompt,
            timeout=self.timeout,
            temperature=0.0,
            max_tokens=min(self.max_tokens, 4096),
            thinking="disabled",
            reasoning_effort=None,
            response_format="json_object",
        )
        try:
            fallback = _director_decision_from_content(fallback_content)
            fallback_error = _director_contract_error(fallback, snapshot)
            if fallback_error is not None:
                raise fallback_error
            return fallback
        except Exception as fallback_error:
            self.ledger.mark_last_invalid(fallback_error)
            raise
