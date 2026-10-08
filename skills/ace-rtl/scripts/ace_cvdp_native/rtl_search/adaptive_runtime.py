"""Benchmark-neutral Smart-RTL role and engine construction.

The benchmark entrypoints own task discovery and fail-closed verification.
Everything that defines the adaptive search policy is assembled here so CVDP
and ChipVerilog cannot silently drift to different role sets or budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .ace_adapter import CallLedger
from .adaptive_ace_adapter import (
    AceAdaptiveAnalyst,
    AceAdaptiveBuilder,
    AceAdaptiveCandidateJudge,
    AceAdaptiveDebater,
    AceAdaptiveDirector,
    AceAdaptivePlanner,
)
from .adaptive_engine import AdaptiveRTLDirectorEngine
from .adaptive_roles import AdaptiveRoleSet
from .skill_instructions import BWRTLSkillInstructions


@dataclass(frozen=True, slots=True)
class AdaptiveRuntimeConfig:
    planner_model: str
    generator_model: str
    analyst_model: str
    director_model: str
    debater_model: str
    judge_model: str
    planner_max_tokens: int
    builder_max_tokens: int
    analyst_max_tokens: int
    director_max_tokens: int
    debater_max_tokens: int
    judge_max_tokens: int
    planner_reasoning_effort: str
    builder_reasoning_effort: str
    analyst_reasoning_effort: str
    director_reasoning_effort: str
    debater_reasoning_effort: str
    judge_reasoning_effort: str
    llm_timeout: int
    physical_workers: int
    max_rounds: int
    max_total_tokens: Optional[int] = None
    max_llm_calls: Optional[int] = None
    max_verifier_calls: Optional[int] = None
    enable_candidate_judge: bool = True

    @property
    def resource_budget(self) -> dict[str, int]:
        return {
            key: value
            for key, value in {
                "total_tokens": self.max_total_tokens,
                "llm_calls": self.max_llm_calls,
                "verifier_calls": self.max_verifier_calls,
            }.items()
            if value is not None
        }


@dataclass(slots=True)
class AdaptiveRuntime:
    ledger: CallLedger
    builder: AceAdaptiveBuilder
    roles: AdaptiveRoleSet
    engine: AdaptiveRTLDirectorEngine


def role_cost_history(
    ledger: CallLedger,
    *,
    planner_floor: int,
    builder_floor: int,
    director_floor: int,
    debater_floor: int,
    judge_floor: int,
) -> dict[str, Any]:
    """Summarize observed role costs and conservative completion reservations."""

    summary = ledger.summary()
    grouped: dict[str, dict[str, int]] = {}
    telemetry_calls = 0
    for record in summary.get("records", []):
        role = str(record.get("role", "unknown"))
        item = grouped.setdefault(
            role,
            {"calls": 0, "reported_total_tokens": 0, "max_reported_total_tokens": 0},
        )
        item["calls"] += 1
        usage = record.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), (int, float)):
            item["reported_total_tokens"] += int(usage["total_tokens"])
            item["max_reported_total_tokens"] = max(
                item["max_reported_total_tokens"], int(usage["total_tokens"])
            )
            telemetry_calls += 1
    for item in grouped.values():
        item["average_reported_total_tokens"] = (
            item["reported_total_tokens"] // item["calls"] if item["calls"] else 0
        )

    def observed_max(prefix: str, floor: int) -> int:
        return max(
            [floor]
            + [
                item["max_reported_total_tokens"]
                for role, item in grouped.items()
                if role.startswith(prefix)
            ]
        )

    reservations = {
        "current_director_decision": observed_max(
            "adaptive_director", director_floor
        ),
        "candidate_action": observed_max("adaptive_builder", builder_floor),
        "planner_action": observed_max("adaptive_planner", planner_floor),
        "discuss_action": observed_max("adaptive_debater", debater_floor),
        "candidate_judge_action": observed_max(
            "adaptive_candidate_judge", judge_floor
        ),
    }
    return {
        "by_runtime_role": grouped,
        "telemetry_complete": telemetry_calls == summary.get("calls", 0),
        "minimum_completion_reservation_total_tokens": reservations,
        "semantic_action_costs": {
            "seed_plan|refine|fork|fuse": (
                "one Builder call + optional Candidate Judge call + one verifier "
                "call + one Analyst call on a non-infrastructure failure"
            ),
            "review_spec|replan": "one Planner call",
            "discuss": "one Debater call",
            "backtrack|park|prune|stop": "no role call after Director dispatch",
        },
    }


def build_adaptive_runtime(
    *,
    workspace: Path,
    run_dir: Path,
    cid: str,
    verifier: Any,
    instructions: BWRTLSkillInstructions,
    config: AdaptiveRuntimeConfig,
) -> AdaptiveRuntime:
    """Build the same adaptive method around a benchmark-specific verifier."""

    ledger = CallLedger(
        run_dir,
        max_calls=config.max_llm_calls,
        max_total_tokens=config.max_total_tokens,
    )
    builder = AceAdaptiveBuilder(
        workspace=workspace,
        cid=cid,
        model=config.generator_model,
        timeout=config.llm_timeout,
        ledger=ledger,
        run_dir=run_dir,
        max_tokens=config.builder_max_tokens,
        reasoning_effort=config.builder_reasoning_effort,
    )
    judge = (
        AceAdaptiveCandidateJudge(
            workspace=workspace,
            model=config.judge_model,
            timeout=config.llm_timeout,
            ledger=ledger,
            max_tokens=config.judge_max_tokens,
            reasoning_effort=config.judge_reasoning_effort,
        )
        if config.enable_candidate_judge
        else None
    )
    roles = AdaptiveRoleSet(
        planner=AceAdaptivePlanner(
            workspace=workspace,
            model=config.planner_model,
            timeout=config.llm_timeout,
            ledger=ledger,
            max_tokens=config.planner_max_tokens,
            reasoning_effort=config.planner_reasoning_effort,
        ),
        builder=builder,
        verifier=verifier,
        analyst=AceAdaptiveAnalyst(
            workspace=workspace,
            model=config.analyst_model,
            timeout=config.llm_timeout,
            ledger=ledger,
            max_tokens=config.analyst_max_tokens,
            reasoning_effort=config.analyst_reasoning_effort,
        ),
        director=AceAdaptiveDirector(
            workspace=workspace,
            model=config.director_model,
            timeout=config.llm_timeout,
            ledger=ledger,
            max_tokens=config.director_max_tokens,
            reasoning_effort=config.director_reasoning_effort,
        ),
        debater=AceAdaptiveDebater(
            workspace=workspace,
            model=config.debater_model,
            timeout=config.llm_timeout,
            ledger=ledger,
            max_tokens=config.debater_max_tokens,
            reasoning_effort=config.debater_reasoning_effort,
        ),
        probe=None,
        judge=judge,
        api_call_counter=lambda: ledger.count,
        resource_usage_callback=lambda: {
            "total_tokens": int(
                ledger.summary().get("usage_totals", {}).get("total_tokens", 0)
            ),
            "llm_calls": ledger.count,
            "verifier_calls": int(verifier.calls),
        },
        resource_cost_callback=lambda: role_cost_history(
            ledger,
            planner_floor=config.planner_max_tokens,
            builder_floor=config.builder_max_tokens,
            director_floor=config.director_max_tokens,
            debater_floor=config.debater_max_tokens,
            judge_floor=config.judge_max_tokens if judge is not None else 0,
        ),
    )
    engine = AdaptiveRTLDirectorEngine(
        roles=roles,
        instructions=instructions,
        physical_workers=config.physical_workers,
        max_rounds=config.max_rounds,
        checkpoint_dir=run_dir,
        resource_budget=config.resource_budget,
    )
    return AdaptiveRuntime(ledger=ledger, builder=builder, roles=roles, engine=engine)
