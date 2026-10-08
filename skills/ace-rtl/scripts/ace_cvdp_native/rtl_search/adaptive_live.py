#!/usr/bin/env python3
"""Run skill-defined adaptive RTL-Director on exactly one CVDP datapoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

from ..diagnostics import git_commit_or_unknown, maybe_reexec_workspace_python
from ..harness import apply_eda_tool_setup, native_tool_preflight
from ..processes import Tee, load_cid_rows, unique_run_dir, write_json, write_jsonl
from .adaptive_runtime import AdaptiveRuntimeConfig, build_adaptive_runtime
from .cvdp_adapter import CVDPNativeVerifier, sanitize_live_row, task_contract_from_row
from .live import _auto_bind_adapter, _workspace_path, _write_selected
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
        description="Run one Smart-RTL adaptive search on CVDP"
    )
    parser.add_argument("--workspace", default=os.getcwd())
    parser.add_argument("--cvdp-repo", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cid", required=True)
    parser.add_argument("--id", required=True, help="Exactly one CVDP datapoint id")
    parser.add_argument("--outputs-root", default="outputs")
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
        default=_env_int("BW_PHYSICAL_WORKERS", 4),
        help="Execution concurrency only; never a branch-selection K",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=_env_int("BW_MAX_ROUNDS", 12),
        help="Fail-safe against an unresponsive Director, not a selection width",
    )
    parser.add_argument("--llm-timeout", type=int, default=1800)
    parser.add_argument("--evaluator-timeout", type=int, default=300)
    parser.add_argument("--evaluator-seed", type=int, default=1)
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


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    workspace = Path(args.workspace).resolve()
    maybe_reexec_workspace_python(workspace)
    dataset = _workspace_path(args.dataset, workspace, "dataset")
    outputs_root = _workspace_path(args.outputs_root, workspace, "outputs root")
    cvdp_repo = _workspace_path(
        args.cvdp_repo or str(workspace / "external/cvdp_benchmark"),
        workspace,
        "CVDP repo",
    )
    if not (cvdp_repo / "run_benchmark.py").exists():
        raise FileNotFoundError(f"CVDP repo not found: {cvdp_repo}")
    _auto_bind_adapter(workspace)

    rows = load_cid_rows(dataset, args.cid, [args.id])
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one row for cid={args.cid} id={args.id}, found {len(rows)}"
        )
    row = sanitize_live_row(rows[0])
    task = task_contract_from_row(row)
    skill = BWRTLSkillInstructions.from_workspace(workspace)

    run_dir = unique_run_dir(outputs_root, f"adaptive_rtl_search_{args.cid}", dataset.stem)
    log_file = (run_dir / "run.log").open("w", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print(f"RUN_DIR={run_dir}")

    filtered = run_dir / f"{dataset.stem}_{args.cid}_{args.id}.jsonl"
    write_jsonl(filtered, [row])
    eda_setup = apply_eda_tool_setup([row])
    preflight = native_tool_preflight([row])
    preflight["eda_tool_setup"] = eda_setup
    write_json(run_dir / "native_tool_preflight.json", preflight)
    metadata = {
        "dataset": str(dataset),
        "filtered_dataset": str(filtered),
        "cid": args.cid,
        "id": args.id,
        "target_files": list(task.target_files),
        "cvdp_repo": str(cvdp_repo),
        "cvdp_commit": git_commit_or_unknown(cvdp_repo),
        "controller": "smart_rtl_adaptive_round_director/v2",
        "skill_root": str(skill.skill_root),
        "skill_version": skill.version,
        "role_map": {
            "planner": args.planner_model,
            "builder": args.generator_model,
            "analyst": args.analyst_model,
            "director": args.director_model,
            "debater": args.debater_model,
            "judge": args.judge_model,
            "verifier": "native_cvdp_fail_closed",
        },
        "role_controls": {
            "planner": {
                "reasoning_effort": args.planner_reasoning_effort,
                "max_tokens": args.planner_max_tokens,
            },
            "builder": {
                "reasoning_effort": args.builder_reasoning_effort,
                "max_tokens": args.builder_max_tokens,
            },
            "analyst": {
                "reasoning_effort": args.analyst_reasoning_effort,
                "max_tokens": args.analyst_max_tokens,
                "context": "candidate_delta_plus_eda_and_relevant_ledger",
            },
            "director": {
                "reasoning_effort": args.director_reasoning_effort,
                "max_tokens": args.director_max_tokens,
                "dispatch_width": "model_selected_variable_length",
            },
            "debater": {
                "reasoning_effort": args.debater_reasoning_effort,
                "max_tokens": args.debater_max_tokens,
                "authority": "comparison_only_director_retains_scheduling",
            },
            "judge": {
                "enabled": args.enable_candidate_judge,
                "reasoning_effort": args.judge_reasoning_effort,
                "max_tokens": args.judge_max_tokens,
                "authority": "action_fidelity_only_no_hidden_correctness",
            },
        },
        "physical_workers": args.physical_workers,
        "max_rounds_safety_guard": args.max_rounds,
        "algorithmic_branch_width": None,
        "algorithmic_selection_k": None,
        "available_optional_probe_role": False,
        "evaluator_timeout": args.evaluator_timeout,
        "evaluator_seed": args.evaluator_seed,
        "resource_budget": {
            key: value
            for key, value in {
                "total_tokens": args.max_total_tokens,
                "llm_calls": args.max_llm_calls,
                "verifier_calls": args.max_verifier_calls,
            }.items()
            if value is not None
        },
        "reference_patch_bodies_persisted": False,
    }
    write_json(run_dir / "metadata.json", metadata)

    if not preflight["ok"]:
        write_json(
            run_dir / "result.json",
            {
                "status": "infrastructure_failure",
                "error": "missing native EDA tools",
                "native_tool_preflight": preflight,
                "llm_api_calls": 0,
            },
        )
        print(
            "INFRA_SETUP: missing required native EDA tool(s): "
            + ", ".join(preflight["missing_tools"])
        )
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        return 2

    verifier = CVDPNativeVerifier(
        cvdp_repo=cvdp_repo,
        row=row,
        row_jsonl=filtered,
        run_dir=run_dir,
        cid=args.cid,
        evaluator_timeout=args.evaluator_timeout,
        evaluator_seed=args.evaluator_seed,
    )
    runtime = build_adaptive_runtime(
        workspace=workspace,
        run_dir=run_dir,
        cid=args.cid,
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
    builder = runtime.builder

    try:
        loop_result = runtime.engine.run(task)
        _write_selected(run_dir, builder, loop_result.selected_candidate_hash)
        payload = loop_result.to_dict()
        payload["call_telemetry"] = ledger.summary()
        payload["verifier_calls"] = verifier.calls
        write_json(run_dir / "result.json", payload)
        print(
            json.dumps(
                {
                    "status": loop_result.status,
                    "rounds": loop_result.rounds,
                    "selected_candidate_hash": loop_result.selected_candidate_hash,
                    "llm_api_calls": ledger.count,
                    "verifier_calls": verifier.calls,
                },
                indent=2,
            )
        )
        return 0 if loop_result.status == "passed" else 1
    except Exception as exc:
        write_json(
            run_dir / "result.json",
            {
                "status": "adapter_error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "call_telemetry": ledger.summary(),
                "verifier_calls": verifier.calls,
            },
        )
        print(f"ADAPTER_ERROR: {exc}")
        return 2
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
