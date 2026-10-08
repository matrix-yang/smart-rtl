"""ACE role-call adapters for the generic RTL search mechanism.

This module owns language-model I/O only.  It does not evaluate candidates or
choose search-tree actions.  Planner and Analyst responses are strict JSON;
Builder responses use the same full-file HDL extraction and normalization path
as the accepted ACE CVDP runner.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from extract_solution import extract_code, split_target_files

from ..config import classify_task, system_prompt_for_task
from ..llm import run_role_call
from ..processes import write_json
from ..sv import normalize_generated_sv
from .contracts import CandidateArtifact, Diagnosis, PlanHypothesis, TaskContract
from .node import ActionKind, EDAEvidence, EDAStage


RoleCall = Callable[..., str]
_PLAN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")


class RoleResponseError(RuntimeError):
    """A provider response completed but cannot be consumed as a role answer."""


class ResourceBudgetExceeded(RuntimeError):
    """A role call was blocked because the configured resource envelope is spent."""


def _bounded(text: str, limit: int = 120_000) -> str:
    """Bound role context while retaining both the specification and log tail."""

    if len(text) <= limit:
        return text
    marker = "\n\n... [ROLE CONTEXT TRUNCATED] ...\n\n"
    remaining = max(2_000, limit - len(marker))
    head = int(remaining * 0.6)
    return text[:head] + marker + text[-(remaining - head) :]


def _json_object(text: str) -> Dict[str, Any]:
    """Parse one JSON object from a response that may contain a fence/preamble."""

    cleaned = re.sub(r"(?is)<think\b[^>]*>.*?</think>\s*", "", text).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.I | re.S)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(cleaned)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            if start < 0:
                continue
            try:
                payload, _end = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("role response did not contain a JSON object")


def _first_usage(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        usage = value.get("usage")
        if isinstance(usage, dict):
            return {
                str(key): item
                for key, item in usage.items()
                if isinstance(item, (int, float, str, bool)) or item is None
            }
        for item in value.values():
            found = _first_usage(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _first_usage(item)
            if found is not None:
                return found
    return None


@dataclass(slots=True)
class CallLedger:
    """Durable per-role call ledger, including missing telemetry explicitly."""

    run_dir: Path
    role_call: RoleCall = run_role_call
    max_calls: Optional[int] = None
    max_total_tokens: Optional[int] = None
    records: list[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _next_sequence: int = field(default=0, init=False, repr=False)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.records)

    def invoke(
        self,
        *,
        workspace: Path,
        role: str,
        component: str,
        model: str,
        system: str,
        prompt: str,
        timeout: int,
        temperature: float,
        max_tokens: Optional[int] = None,
        thinking: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        response_format: str = "text",
    ) -> str:
        with self._lock:
            if self.max_calls is not None and len(self.records) >= self.max_calls:
                raise ResourceBudgetExceeded(
                    f"LLM call budget exhausted: used={len(self.records)}, "
                    f"limit={self.max_calls}"
                )
            used_tokens = self._total_tokens_unlocked()
            if self.max_total_tokens is not None and used_tokens >= self.max_total_tokens:
                raise ResourceBudgetExceeded(
                    f"token budget exhausted: used={used_tokens}, "
                    f"limit={self.max_total_tokens}"
                )
            self._next_sequence += 1
            sequence = self._next_sequence
            call_dir = self.run_dir / "roles" / role / f"call_{sequence:03d}"
            call_dir.mkdir(parents=True, exist_ok=True)
            response_path = call_dir / "response.json"
            record: Dict[str, Any] = {
                "sequence": sequence,
                "role": role,
                "component": component,
                "model": model,
                "response_path": str(response_path),
                "status": "started",
                "usage": None,
                "token_telemetry": "missing",
                "request": {
                    "max_tokens": max_tokens,
                    "thinking": thinking,
                    "reasoning_effort": reasoning_effort,
                    "response_format": response_format,
                },
            }
            self.records.append(record)
            self._flush_unlocked()
        try:
            content = self.role_call(
                workspace=workspace,
                component=component,
                model=model,
                system=system,
                prompt=_bounded(prompt),
                out_json=response_path,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
            )
            if not response_path.exists():
                write_json(
                    response_path,
                    {"provider": "injected", "model": model, "content": content},
                )
            payload = json.loads(response_path.read_text(encoding="utf-8"))
            usage = _first_usage(payload)
            finish_reason = payload.get("finish_reason")
            truncated = bool(payload.get("truncated")) or finish_reason == "length"
            with self._lock:
                record.update(
                    {
                        "status": "ok",
                        "usage": usage,
                        "token_telemetry": "present" if usage is not None else "missing",
                        "finish_reason": finish_reason,
                        "truncated": truncated,
                    }
                )
            if truncated:
                with self._lock:
                    record["status"] = "truncated"
                raise RoleResponseError(
                    f"{role} response exhausted max_tokens before a final answer"
                )
            if not content.strip():
                with self._lock:
                    record["status"] = "empty"
                raise RoleResponseError(f"{role} response contained no final content")
            return content
        except Exception as exc:
            with self._lock:
                if record.get("status") in {"started", "ok"}:
                    record["status"] = "error"
                record["error"] = str(exc)[:2000]
            raise
        finally:
            with self._lock:
                self._flush_unlocked()

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return self._summary_unlocked()

    def _summary_unlocked(self) -> Dict[str, Any]:
        numeric_totals: Dict[str, float] = {}
        for record in self.records:
            usage = record.get("usage")
            if not isinstance(usage, dict):
                continue
            for key, value in usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_totals[key] = numeric_totals.get(key, 0) + value
        return {
            "calls": len(self.records),
            "successful_calls": sum(record.get("status") == "ok" for record in self.records),
            "failed_calls": sum(
                record.get("status") in {"error", "truncated", "empty", "invalid_response"}
                for record in self.records
            ),
            "calls_with_token_telemetry": sum(
                record.get("token_telemetry") == "present" for record in self.records
            ),
            "usage_totals": numeric_totals,
            "budget": {
                "max_calls": self.max_calls,
                "max_total_tokens": self.max_total_tokens,
                "calls_remaining": (
                    max(self.max_calls - len(self.records), 0)
                    if self.max_calls is not None
                    else None
                ),
                "tokens_remaining": (
                    max(self.max_total_tokens - int(numeric_totals.get("total_tokens", 0)), 0)
                    if self.max_total_tokens is not None
                    else None
                ),
            },
            "records": list(self.records),
        }

    def _total_tokens_unlocked(self) -> int:
        total = 0
        for record in self.records:
            usage = record.get("usage")
            if not isinstance(usage, dict):
                continue
            value = usage.get("total_tokens")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total += int(value)
        return total

    def _flush(self) -> None:
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        write_json(self.run_dir / "call_ledger.json", self._summary_unlocked())

    def mark_last_invalid(self, error: Exception) -> None:
        """Reclassify the latest syntactically completed call as schema-invalid."""

        with self._lock:
            if self.records and self.records[-1].get("status") == "ok":
                self.records[-1].update(
                    {"status": "invalid_response", "error": str(error)[:2000]}
                )
                self._flush_unlocked()


def _context_text(task: TaskContract) -> str:
    if not task.immutable_context:
        return "No immutable public context files were supplied."
    return "\n\n".join(
        f"# Context file: {path}\n```\n{content}\n```"
        for path, content in task.immutable_context
    )


def _artifact_text(candidate: Optional[CandidateArtifact]) -> str:
    if candidate is None:
        return "No parent candidate exists."
    return "\n\n".join(
        f"// TARGET_FILE: {path}\n{content}" for path, content in candidate.files
    )


class AceLLMPlanner:
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
        requested_count: int,
        existing_plans: Sequence[PlanHypothesis],
    ) -> Sequence[PlanHypothesis]:
        schema = {
            "plans": [
                {
                    "plan_id": "short_unique_id",
                    "summary": "implementation plan",
                    "distinguishing_assumption": "why this differs",
                    "invariants": ["public property to preserve"],
                }
            ]
        }
        prompt = f"""{instruction}

IMMUTABLE PUBLIC CONTEXT:
{_context_text(task)}

Return exactly {requested_count} plan objects as JSON matching this shape:
{json.dumps(schema, indent=2)}
Return JSON only.  Do not include Markdown or RTL.
"""
        content = self.ledger.invoke(
            workspace=self.workspace,
            role="planner",
            component="coordinator",
            model=self.model,
            system="You are the ACE-RTL planning role. Return strict JSON only and never write RTL.",
            prompt=prompt,
            timeout=self.timeout,
            temperature=0.4,
            max_tokens=self.max_tokens,
            thinking="enabled",
            reasoning_effort=self.reasoning_effort,
            response_format="json_object",
        )
        raw_plans = _json_object(content).get("plans")
        if not isinstance(raw_plans, list) or len(raw_plans) != requested_count:
            raise ValueError(f"Planner must return exactly {requested_count} plan objects")
        existing_ids = {plan.plan_id for plan in existing_plans}
        existing_summaries = {plan.summary.strip().lower() for plan in existing_plans}
        parsed = []
        for index, item in enumerate(raw_plans, start=1):
            if not isinstance(item, dict):
                raise ValueError("Planner plan entries must be JSON objects")
            raw_id = str(item.get("plan_id", "")).strip()
            plan_id = raw_id if _PLAN_ID_RE.fullmatch(raw_id) else f"p{len(existing_plans) + index:03d}"
            summary = str(item.get("summary", "")).strip()
            assumption = str(item.get("distinguishing_assumption", "")).strip()
            invariants_raw = item.get("invariants", [])
            if not isinstance(invariants_raw, list):
                raise ValueError("Planner invariants must be a JSON list")
            invariants = tuple(str(value).strip() for value in invariants_raw if str(value).strip())
            if plan_id in existing_ids or any(plan.plan_id == plan_id for plan in parsed):
                plan_id = f"p{len(existing_plans) + index:03d}"
            if not summary or not assumption:
                raise ValueError("Planner plans require summary and distinguishing_assumption")
            if summary.lower() in existing_summaries or any(
                plan.summary.strip().lower() == summary.lower() for plan in parsed
            ):
                raise ValueError("Planner returned a cosmetic duplicate plan")
            parsed.append(PlanHypothesis(plan_id, summary, assumption, invariants))
        return tuple(parsed)


class AceLLMBuilder:
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
        self.calls = 0

    def build(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        instruction: str,
        action: ActionKind,
        parent: Optional[CandidateArtifact],
        diagnosis: Optional[Diagnosis],
    ) -> CandidateArtifact:
        self.calls += 1
        targets = "\n".join(f"- {path}" for path in task.target_files)
        output_contract = (
            "Return one complete section per target headed exactly by "
            "`// TARGET_FILE: <path>`; include every target and no other file."
            if len(task.target_files) > 1
            else f"Return the complete raw source for {task.target_files[0]} only."
        )
        prompt = f"""{instruction}

PUBLIC SPECIFICATION:
{task.specification}

IMMUTABLE PUBLIC CONTEXT:
{_context_text(task)}

PARENT CANDIDATE:
{_artifact_text(parent)}

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
            role="builder",
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
        candidate_dir = self.run_dir / "candidates" / f"builder_{self.calls:03d}_{artifact.candidate_hash}"
        for name, code in artifact.files:
            output_path = candidate_dir.joinpath(*PurePosixPath(name).parts)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code, encoding="utf-8")
        write_json(
            candidate_dir / "candidate.json",
            {
                "candidate_hash": artifact.candidate_hash,
                "action": action.value,
                "plan_id": plan.plan_id,
                "parent_hash": parent.candidate_hash if parent else None,
                "diagnosis": diagnosis.rationale if diagnosis else None,
                "files": [name for name, _content in artifact.files],
                "contract_issues": list(artifact.contract_issues(task)),
            },
        )
        self.candidates[artifact.candidate_hash] = artifact
        return artifact


class AceLLMAnalyst:
    def __init__(
        self,
        *,
        workspace: Path,
        model: str,
        timeout: int,
        ledger: CallLedger,
        max_tokens: int = 8192,
        retry_max_tokens: int = 2048,
        reasoning_effort: str = "low",
    ) -> None:
        self.workspace = workspace
        self.model = model
        self.timeout = timeout
        self.ledger = ledger
        self.max_tokens = max_tokens
        self.retry_max_tokens = retry_max_tokens
        self.reasoning_effort = reasoning_effort
        self.retries = 0
        self.fallbacks = 0
        self.failure_events: list[Dict[str, str]] = []

    def analyze(
        self,
        task: TaskContract,
        plan: PlanHypothesis,
        candidate: CandidateArtifact,
        evidence: EDAEvidence,
        instruction: str,
    ) -> Diagnosis:
        schema = {
            "failure_class": "concise category",
            "rationale": "evidence-grounded explanation",
            "repair_obligation": "one bounded next obligation",
            "localized_failure": None,
            "plan_conflict": False,
            "distinct_hypotheses_available": False,
        }
        prompt = f"""{instruction}

PUBLIC SPECIFICATION:
{task.specification}

IMMUTABLE PUBLIC CONTEXT:
{_context_text(task)}

CANDIDATE RTL:
{_artifact_text(candidate)}

Return one JSON diagnosis matching:
{json.dumps(schema, indent=2)}
`localized_failure` must be true, false, or null. Return JSON only and never emit RTL.
"""
        try:
            content = self.ledger.invoke(
                workspace=self.workspace,
                role="analyst",
                component="reflector",
                model=self.model,
                system="You are the ACE-RTL failure-analysis role. Diagnose public EDA evidence and return strict JSON only.",
                prompt=prompt,
                timeout=self.timeout,
                temperature=0.1,
                max_tokens=self.max_tokens,
                thinking="enabled",
                reasoning_effort=self.reasoning_effort,
                response_format="json_object",
            )
            return self._parse_diagnosis(content)
        except Exception as primary_error:
            self.ledger.mark_last_invalid(primary_error)
            self.retries += 1
            self.failure_events.append(
                {"phase": "primary", "error": str(primary_error)[:500]}
            )

        retry_prompt = f"""Return the required diagnosis JSON now, with no analysis or preamble.
Use only these bounded facts:
- plan: {plan.plan_id} — {plan.summary}
- EDA stage: {evidence.original_stage}
- failure signature: {evidence.failure_signature}
- public feedback: {evidence.public_feedback or "none"}
- candidate hash: {candidate.candidate_hash}

Required keys: failure_class, rationale, repair_obligation,
localized_failure, plan_conflict, distinct_hypotheses_available.
localized_failure must be true, false, or null. Return JSON only.
"""
        try:
            content = self.ledger.invoke(
                workspace=self.workspace,
                role="analyst_retry",
                component="reflector",
                model=self.model,
                system="Return one compact JSON diagnosis only. Do not reason aloud and do not emit RTL.",
                prompt=retry_prompt,
                timeout=self.timeout,
                temperature=0.0,
                max_tokens=self.retry_max_tokens,
                thinking="disabled",
                reasoning_effort=None,
                response_format="json_object",
            )
            return self._parse_diagnosis(content)
        except Exception as retry_error:
            self.ledger.mark_last_invalid(retry_error)
            self.fallbacks += 1
            self.failure_events.append(
                {"phase": "retry", "error": str(retry_error)[:500]}
            )
            return self._fallback_diagnosis(evidence, retry_error)

    @staticmethod
    def _parse_diagnosis(content: str) -> Diagnosis:
        payload = _json_object(content)
        failure_class = str(payload.get("failure_class", "")).strip()
        rationale = str(payload.get("rationale", "")).strip()
        repair = str(payload.get("repair_obligation", "")).strip()
        localized = payload.get("localized_failure")
        if localized not in (True, False, None):
            raise ValueError("Analyst localized_failure must be true, false, or null")
        plan_conflict = payload.get("plan_conflict", False)
        distinct = payload.get("distinct_hypotheses_available", False)
        if not isinstance(plan_conflict, bool) or not isinstance(distinct, bool):
            raise ValueError("Analyst decision flags must be booleans")
        if not failure_class or not rationale or not repair:
            raise ValueError("Analyst diagnosis omitted required text fields")
        return Diagnosis(
            failure_class=failure_class,
            rationale=rationale,
            repair_obligation=repair,
            localized_failure=localized,
            plan_conflict=plan_conflict,
            distinct_hypotheses_available=distinct,
        )

    @staticmethod
    def _fallback_diagnosis(evidence: EDAEvidence, error: Exception) -> Diagnosis:
        localized = evidence.stage in {EDAStage.COMPILE, EDAStage.ELABORATE}
        if evidence.stage is EDAStage.SIMULATION:
            localized = None
        return Diagnosis(
            failure_class="analyst_response_unavailable",
            rationale=(
                "The structured Analyst response was unavailable after one bounded retry; "
                f"the deterministic policy is using EDA stage {evidence.original_stage}. "
                f"Adapter detail: {str(error)[:160]}"
            ),
            repair_obligation=(
                "Make one minimal evidence-grounded correction for failure signature: "
                + evidence.failure_signature[:300]
            ),
            localized_failure=localized,
            plan_conflict=False,
            distinct_hypotheses_available=False,
        )

    def telemetry(self) -> Dict[str, Any]:
        return {
            "primary_max_tokens": self.max_tokens,
            "retry_max_tokens": self.retry_max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "retries": self.retries,
            "fallbacks": self.fallbacks,
            "failure_events": list(self.failure_events),
        }
