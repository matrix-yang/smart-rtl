#!/usr/bin/env python3
"""Run the generic RTL search mechanism on exactly one live CVDP datapoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path, PurePosixPath
from typing import Optional

from ..diagnostics import git_commit_or_unknown, maybe_reexec_workspace_python
from ..harness import apply_eda_tool_setup, native_tool_preflight
from ..llm import role_api_key
from ..processes import (
    Tee,
    ensure_under,
    load_cid_rows,
    unique_run_dir,
    write_json,
    write_jsonl,
)
from .ace_adapter import AceLLMAnalyst, AceLLMBuilder, AceLLMPlanner, CallLedger
from .cvdp_adapter import CVDPNativeVerifier, sanitize_live_row, task_contract_from_row
from .engine import RTLSearchEngine
from .instructions import GeneralACEInstructions
from .roles import RoleSet


DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one BW-RTL generic tree-search shakedown through ACE and native CVDP"
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
    parser.add_argument(
        "--planner-max-tokens", type=int, default=_env_int("BW_PLANNER_MAX_TOKENS", 8192)
    )
    parser.add_argument(
        "--builder-max-tokens", type=int, default=_env_int("BW_BUILDER_MAX_TOKENS", 32000)
    )
    parser.add_argument(
        "--analyst-max-tokens", type=int, default=_env_int("BW_ANALYST_MAX_TOKENS", 8192)
    )
    parser.add_argument(
        "--analyst-retry-max-tokens",
        type=int,
        default=_env_int("BW_ANALYST_RETRY_MAX_TOKENS", 2048),
    )
    parser.add_argument(
        "--planner-reasoning-effort", choices=["low", "high", "max"], default="low"
    )
    parser.add_argument(
        "--builder-reasoning-effort", choices=["low", "high", "max"], default="low"
    )
    parser.add_argument(
        "--analyst-reasoning-effort", choices=["low", "high", "max"], default="low"
    )
    parser.add_argument("--branch-width", type=int, default=2)
    parser.add_argument("--max-plans", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--llm-timeout", type=int, default=1800)
    parser.add_argument("--evaluator-timeout", type=int, default=300)
    parser.add_argument(
        "--evaluator-seed",
        type=int,
        default=1,
        help="Fixed COCOTB_RANDOM_SEED reused for every candidate in this search",
    )
    return parser.parse_args(argv)


def _workspace_path(raw: str, workspace: Path, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = workspace / path
    return ensure_under(path, workspace, label)


def _auto_bind_adapter(workspace: Path) -> None:
    if not os.environ.get("ACE_RTL_LLM_SCRIPT") and os.environ.get("DEEPSEEK_API_KEY"):
        adapter = (workspace / "skills/ace-rtl/scripts/deepseek_adapter.py").resolve()
        if adapter.exists():
            os.environ["ACE_RTL_LLM_SCRIPT"] = str(adapter)
            print(f"Auto-binding ACE_RTL_LLM_SCRIPT={adapter}")
    if not role_api_key() and not os.environ.get("ACE_RTL_LLM_SCRIPT"):
        raise RuntimeError("Set NVIDIA_API_KEY, or set DEEPSEEK_API_KEY/ACE_RTL_LLM_SCRIPT")


def _write_selected(run_dir: Path, builder: AceLLMBuilder, candidate_hash: Optional[str]) -> None:
    if not candidate_hash:
        return
    candidate = builder.candidates.get(candidate_hash)
    if candidate is None:
        raise RuntimeError(f"selected candidate {candidate_hash} is absent from Builder ledger")
    selected_dir = run_dir / "selected_candidate"
    for name, code in candidate.files:
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe selected candidate path: {name}")
        path = selected_dir.joinpath(*rel.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
    write_json(
        selected_dir / "manifest.json",
        {"candidate_hash": candidate_hash, "files": [name for name, _ in candidate.files]},
    )


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

    run_dir = unique_run_dir(outputs_root, f"rtl_search_{args.cid}", dataset.stem)
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
        "role_map": {
            "planner": args.planner_model,
            "builder": args.generator_model,
            "analyst": args.analyst_model,
            "director": "deterministic_tree_policy_v0",
        },
        "role_controls": {
            "planner": {
                "thinking": "enabled",
                "reasoning_effort": args.planner_reasoning_effort,
                "max_tokens": args.planner_max_tokens,
                "response_format": "json_object",
            },
            "builder": {
                "thinking": "enabled",
                "reasoning_effort": args.builder_reasoning_effort,
                "max_tokens": args.builder_max_tokens,
                "response_format": "text",
            },
            "analyst": {
                "thinking": "enabled",
                "reasoning_effort": args.analyst_reasoning_effort,
                "max_tokens": args.analyst_max_tokens,
                "response_format": "json_object",
                "retry_thinking": "disabled",
                "retry_max_tokens": args.analyst_retry_max_tokens,
            },
        },
        "branch_width": args.branch_width,
        "max_plans": args.max_plans,
        "max_steps": args.max_steps,
        "evaluator_timeout": args.evaluator_timeout,
        "evaluator_seed": args.evaluator_seed,
        "instruction_provider": "general_ace_role_contract/v0.1",
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
        print("INFRA_SETUP: missing required native EDA tool(s): " + ", ".join(preflight["missing_tools"]))
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        return 2

    ledger = CallLedger(run_dir)
    builder = AceLLMBuilder(
        workspace=workspace,
        cid=args.cid,
        model=args.generator_model,
        timeout=args.llm_timeout,
        ledger=ledger,
        run_dir=run_dir,
        max_tokens=args.builder_max_tokens,
        reasoning_effort=args.builder_reasoning_effort,
    )
    verifier = CVDPNativeVerifier(
        cvdp_repo=cvdp_repo,
        row=row,
        row_jsonl=filtered,
        run_dir=run_dir,
        cid=args.cid,
        evaluator_timeout=args.evaluator_timeout,
        evaluator_seed=args.evaluator_seed,
    )
    analyst = AceLLMAnalyst(
        workspace=workspace,
        model=args.analyst_model,
        timeout=args.llm_timeout,
        ledger=ledger,
        max_tokens=args.analyst_max_tokens,
        retry_max_tokens=args.analyst_retry_max_tokens,
        reasoning_effort=args.analyst_reasoning_effort,
    )
    roles = RoleSet(
        planner=AceLLMPlanner(
            workspace=workspace, model=args.planner_model, timeout=args.llm_timeout,
            ledger=ledger, max_tokens=args.planner_max_tokens,
            reasoning_effort=args.planner_reasoning_effort,
        ),
        builder=builder,
        verifier=verifier,
        analyst=analyst,
        api_call_counter=lambda: ledger.count,
    )
    engine = RTLSearchEngine(
        roles=roles,
        instructions=GeneralACEInstructions(),
        branch_width=args.branch_width,
        max_steps=args.max_steps,
        max_plans=args.max_plans,
    )

    try:
        loop_result = engine.run(task)
        _write_selected(run_dir, builder, loop_result.selected_candidate_hash)
        payload = loop_result.to_dict()
        payload["call_telemetry"] = ledger.summary()
        payload["verifier_calls"] = verifier.calls
        payload["analyst_telemetry"] = analyst.telemetry()
        write_json(run_dir / "result.json", payload)
        print(json.dumps({
            "status": loop_result.status,
            "selected_candidate_hash": loop_result.selected_candidate_hash,
            "llm_api_calls": loop_result.llm_api_calls,
            "verifier_calls": verifier.calls,
        }, indent=2))
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
                "analyst_telemetry": analyst.telemetry(),
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
