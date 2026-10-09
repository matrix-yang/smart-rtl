#!/usr/bin/env python3
"""Run skill-defined adaptive RTL-Director on exactly one ChipVerilog task."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ace_rtl_agent.utils.chipverilog_integration import discover_chipverilog_tasks

from ..diagnostics import git_commit_or_unknown, maybe_reexec_workspace_python
from ..processes import Tee, ensure_under, write_json
from .adaptive_runtime import AdaptiveRuntimeConfig, build_adaptive_runtime
from .chipverilog_adapter import (
    ChipVerilogVerifier,
    public_task_metadata,
    task_contract_from_chipverilog_task,
)
from .live import _auto_bind_adapter, _write_selected
from .skill_instructions import BWRTLSkillInstructions


DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("resource limits must be positive")
    return value


def _env_optional_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return _positive_int(raw)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Smart-RTL adaptive search on ChipVerilog"
    )
    parser.add_argument("--workspace", default=os.getcwd())
    parser.add_argument(
        "--chipverilog-root",
        default=os.environ.get("CHIPVERILOG_ROOT", "/workspace/chipverilog"),
    )
    parser.add_argument("--id", help="Exactly one ChipVerilog public task id")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--artifacts-root", default="artifacts/experiment")
    parser.add_argument("--planner-model", default=DEFAULT_MODEL)
    parser.add_argument("--generator-model", default=DEFAULT_MODEL)
    parser.add_argument("--analyst-model", default=DEFAULT_MODEL)
    parser.add_argument("--director-model", default=DEFAULT_MODEL)
    parser.add_argument("--debater-model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--planner-max-tokens", type=int, default=_env_int("BW_PLANNER_MAX_TOKENS", 8192)
    )
    parser.add_argument(
        "--builder-max-tokens", type=int, default=_env_int("BW_BUILDER_MAX_TOKENS", 32000)
    )
    parser.add_argument(
        "--analyst-max-tokens", type=int, default=_env_int("BW_ANALYST_MAX_TOKENS", 6144)
    )
    parser.add_argument(
        "--director-max-tokens", type=int, default=_env_int("BW_DIRECTOR_MAX_TOKENS", 8192)
    )
    parser.add_argument(
        "--debater-max-tokens", type=int, default=_env_int("BW_DEBATER_MAX_TOKENS", 6144)
    )
    parser.add_argument(
        "--judge-max-tokens", type=int, default=_env_int("BW_JUDGE_MAX_TOKENS", 4096)
    )
    for role in ("planner", "builder", "analyst", "director", "debater", "judge"):
        parser.add_argument(
            f"--{role}-reasoning-effort",
            choices=["low", "high", "max"],
            default="low",
        )
    parser.add_argument(
        "--physical-workers",
        type=int,
        default=_env_int("BW_PHYSICAL_WORKERS", 1),
        help="Execution concurrency only; never a branch-selection K",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=_env_int("BW_MAX_ROUNDS", 12),
        help="Fail-safe against an unresponsive Director, not a selection width",
    )
    parser.add_argument("--llm-timeout", type=int, default=1800)
    parser.add_argument("--evaluator-depth", type=int, default=16)
    parser.add_argument("--evaluator-timeout", type=int, default=120)
    parser.add_argument(
        "--max-total-tokens",
        type=_positive_int,
        default=_env_optional_int("BW_MAX_TOTAL_TOKENS"),
        help="Optional observed total-token envelope; overrun invalidates the trial",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=_positive_int,
        default=_env_optional_int("BW_MAX_LLM_CALLS"),
        help="Optional exact LLM-call envelope",
    )
    parser.add_argument(
        "--max-verifier-calls",
        type=_positive_int,
        default=_env_optional_int("BW_MAX_VERIFIER_CALLS"),
        help="Optional exact fail-closed evaluator-call envelope",
    )
    parser.add_argument(
        "--disable-candidate-judge",
        action="store_false",
        dest="enable_candidate_judge",
        help="Compatibility mode for historical runs; aligned runs keep the Judge enabled",
    )
    return parser.parse_args(argv)


def _workspace_path(raw: str, workspace: Path, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = workspace / path
    return ensure_under(path, workspace, label)


def _new_run_dir(args: argparse.Namespace, workspace: Path) -> Path:
    if args.run_dir:
        run_dir = _workspace_path(args.run_dir, workspace, "run directory")
    else:
        root = _workspace_path(args.artifacts_root, workspace, "artifacts root")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = root / f"chipverilog_adaptive_{args.id}_{stamp}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _preflight(root: Path, task_count: int) -> dict:
    tools = {name: shutil.which(name) for name in ("iverilog", "vvp", "yosys")}
    evaluator = root / "tools" / "formal_equivalence.py"
    missing = [name for name, path in tools.items() if path is None]
    if not evaluator.is_file():
        missing.append(str(evaluator))
    if task_count != 64:
        missing.append(f"task_count={task_count}, expected=64")
    return {
        "ok": not missing,
        "chipverilog_root": str(root),
        "task_count": task_count,
        "tools": tools,
        "evaluator": str(evaluator),
        "missing": missing,
    }


def _evaluation_summary(status: str, calls: dict, verifier: ChipVerilogVerifier) -> dict:
    passed = status == "passed"
    if passed:
        failure_mode = "none"
        next_action = "compare the held-out task result against the accepted baseline contract"
    else:
        failure_mode = status
        next_action = "inspect the durable role/evidence trace before any changed rerun"
    return {
        "outcome_summary": f"single-task adaptive smoke ended with status={status}",
        "evaluation_summary": {
            "status": status,
            "llm_calls": calls.get("calls", 0),
            "token_usage": calls.get("usage_totals", {}),
            "verifier_calls": verifier.calls,
            "evaluator_subprocess_calls": verifier.evaluator_subprocess_calls,
            "candidate_contract_rejections": verifier.contract_rejections,
        },
        "claim_update": "supported" if passed else "inconclusive",
        "baseline_relation": "auxiliary adapter smoke; not a ranked baseline comparison",
        "failure_mode": failure_mode,
        "next_action": next_action,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    workspace = Path(args.workspace).resolve()
    maybe_reexec_workspace_python(workspace)
    root = Path(args.chipverilog_root).resolve()
    tasks = discover_chipverilog_tasks(root)
    if args.list_tasks:
        print(json.dumps({"task_count": len(tasks), "tasks": [task.task_id for task in tasks]}))
        return 0
    if not args.id:
        raise ValueError("--id is required unless --list-tasks is used")
    by_id = {task.task_id: task for task in tasks}
    if args.id not in by_id:
        raise ValueError(f"unknown ChipVerilog task: {args.id}")
    benchmark_task = by_id[args.id]
    task = task_contract_from_chipverilog_task(benchmark_task)
    skill = BWRTLSkillInstructions.from_workspace(workspace)
    run_dir = _new_run_dir(args, workspace)
    log_file = (run_dir / "run.log").open("w", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print(f"RUN_DIR={run_dir}")

    preflight = _preflight(root, len(tasks))
    write_json(run_dir / "preflight.json", preflight)
    (run_dir / "public_task_prompt.md").write_text(task.specification, encoding="utf-8")
    metadata = {
        "benchmark": "ChipVerilog",
        "task": public_task_metadata(benchmark_task, root),
        "workspace": str(workspace),
        "workspace_commit": git_commit_or_unknown(workspace),
        "controller": "smart_rtl_adaptive_round_director/v2",
        "skill_root": str(skill.skill_root),
        "skill_version": skill.version,
        "compose_project": "smart-rtl",
        "docker_image": "smart-rtl:deepseek",
        "role_map": {
            "planner": args.planner_model,
            "builder": args.generator_model,
            "analyst": args.analyst_model,
            "director": args.director_model,
            "debater": args.debater_model,
            "judge": args.judge_model,
            "verifier": "chipverilog_formal_equivalence_fail_closed",
        },
        "role_controls": {
            role: {
                "reasoning_effort": getattr(args, f"{role}_reasoning_effort"),
                "max_tokens": getattr(args, f"{role}_max_tokens"),
            }
            for role in (
                "planner",
                "builder",
                "analyst",
                "director",
                "debater",
                "judge",
            )
        },
        "physical_workers": args.physical_workers,
        "max_rounds_safety_guard": args.max_rounds,
        "algorithmic_branch_width": None,
        "algorithmic_selection_k": None,
        "evaluator_depth": args.evaluator_depth,
        "evaluator_timeout": args.evaluator_timeout,
        "evaluator_seed": None,
        "resource_budget": {
            key: value
            for key, value in {
                "total_tokens": args.max_total_tokens,
                "llm_calls": args.max_llm_calls,
                "verifier_calls": args.max_verifier_calls,
            }.items()
            if value is not None
        },
        "candidate_judge_enabled": args.enable_candidate_judge,
        "exact_container_argv": [sys.executable, *sys.argv],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "credential_present": bool(os.environ.get("DEEPSEEK_API_KEY")),
            "base_url_configured": bool(os.environ.get("DEEPSEEK_BASE_URL")),
            "llm_adapter_configured": bool(os.environ.get("ACE_RTL_LLM_SCRIPT")),
        },
        "golden_or_hidden_exposed_to_roles": False,
    }
    write_json(run_dir / "metadata.json", metadata)

    if not preflight["ok"]:
        write_json(
            run_dir / "result.json",
            {"status": "infrastructure_failure", "preflight": preflight, "llm_api_calls": 0},
        )
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log_file.close()
        return 2

    ledger = None
    verifier: Optional[ChipVerilogVerifier] = None
    try:
        _auto_bind_adapter(workspace)
        verifier = ChipVerilogVerifier(
            chipverilog_root=root,
            benchmark_task=benchmark_task,
            run_dir=run_dir,
            evaluator_depth=args.evaluator_depth,
            evaluator_timeout=args.evaluator_timeout,
        )
        runtime = build_adaptive_runtime(
            workspace=workspace,
            run_dir=run_dir,
            cid="chipverilog",
            verifier=verifier,
            instructions=skill,
            config=AdaptiveRuntimeConfig(
                planner_model=args.planner_model,
                generator_model=args.generator_model,
                analyst_model=args.analyst_model,
                director_model=args.director_model,
                debater_model=args.debater_model,
                judge_model=args.judge_model,
                planner_max_tokens=args.planner_max_tokens,
                builder_max_tokens=args.builder_max_tokens,
                analyst_max_tokens=args.analyst_max_tokens,
                director_max_tokens=args.director_max_tokens,
                debater_max_tokens=args.debater_max_tokens,
                judge_max_tokens=args.judge_max_tokens,
                planner_reasoning_effort=args.planner_reasoning_effort,
                builder_reasoning_effort=args.builder_reasoning_effort,
                analyst_reasoning_effort=args.analyst_reasoning_effort,
                director_reasoning_effort=args.director_reasoning_effort,
                debater_reasoning_effort=args.debater_reasoning_effort,
                judge_reasoning_effort=args.judge_reasoning_effort,
                llm_timeout=args.llm_timeout,
                physical_workers=args.physical_workers,
                max_rounds=args.max_rounds,
                max_total_tokens=args.max_total_tokens,
                max_llm_calls=args.max_llm_calls,
                max_verifier_calls=args.max_verifier_calls,
                enable_candidate_judge=args.enable_candidate_judge,
            ),
        )
        ledger = runtime.ledger
        loop_result = runtime.engine.run(task)
        _write_selected(run_dir, runtime.builder, loop_result.selected_candidate_hash)
        calls = ledger.summary()
        payload = loop_result.to_dict()
        payload.update(
            {
                "call_telemetry": calls,
                "verifier_calls": verifier.calls,
                "evaluator_subprocess_calls": verifier.evaluator_subprocess_calls,
                "candidate_contract_rejections": verifier.contract_rejections,
                "evaluator_seed": None,
            }
        )
        write_json(run_dir / "result.json", payload)
        write_json(
            run_dir / "EVALUATION_SUMMARY.json",
            _evaluation_summary(loop_result.status, calls, verifier),
        )
        print(
            json.dumps(
                {
                    "status": loop_result.status,
                    "rounds": loop_result.rounds,
                    "selected_candidate_hash": loop_result.selected_candidate_hash,
                    "llm_api_calls": ledger.count,
                    "verifier_calls": verifier.calls,
                    "evaluator_subprocess_calls": verifier.evaluator_subprocess_calls,
                    "tokens": calls.get("usage_totals", {}),
                },
                indent=2,
            )
        )
        return 0 if loop_result.status == "passed" else 1
    except Exception as exc:
        payload = {
            "status": "adapter_error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "call_telemetry": ledger.summary() if ledger is not None else {},
            "verifier_calls": verifier.calls if verifier else 0,
            "evaluator_subprocess_calls": verifier.evaluator_subprocess_calls if verifier else 0,
            "candidate_contract_rejections": verifier.contract_rejections if verifier else 0,
        }
        write_json(run_dir / "result.json", payload)
        print(f"ADAPTER_ERROR: {exc}")
        return 2
    finally:
        if verifier is not None:
            verifier.close()
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
