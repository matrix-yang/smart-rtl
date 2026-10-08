#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run ACE-RTL on one or more ChipVerilog tasks with durable summaries."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ace_rtl_agent.agent import AceRTLAgent  # noqa: E402
from ace_rtl_agent.agent_types import AgentConfig  # noqa: E402
from ace_rtl_agent.utils.chipverilog_integration import (  # noqa: E402
    UNKNOWN_STATUSES,
    discover_chipverilog_tasks,
    render_chipverilog_prompt,
)


def _default_workspace() -> Path:
    return SCRIPT_DIR.parents[2]


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ACE-RTL generator-reflector-coordinator loop on ChipVerilog."
    )
    parser.add_argument("--workspace", type=Path, default=_default_workspace())
    parser.add_argument(
        "--chipverilog-root",
        type=Path,
        default=os.environ.get("CHIPVERILOG_ROOT"),
        required=os.environ.get("CHIPVERILOG_ROOT") is None,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--modules", nargs="*")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--model",
        default=os.environ.get("ACE_RTL_MODEL", "deepseek-flash"),
    )
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--parallel-experiments", type=int, default=1)
    parser.add_argument("--iteration-timeout", type=int, default=1800)
    parser.add_argument("--eval-timeout", type=int, default=120)
    parser.add_argument("--eval-depth", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--initial-backoff", type=float, default=5.0)
    parser.add_argument("--llm-context-limit", type=int, default=300000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _select_tasks(args: argparse.Namespace):
    tasks = discover_chipverilog_tasks(args.chipverilog_root)
    if args.modules:
        selected = set(args.modules)
        tasks = [task for task in tasks if task.task_id in selected]
        missing = selected - {task.task_id for task in tasks}
        if missing:
            raise ValueError(f"unknown ChipVerilog modules: {sorted(missing)}")
    return tasks


def _configure_llm_adapter(workspace: Path) -> None:
    if os.environ.get("ACE_RTL_LLM_SCRIPT"):
        return
    if os.environ.get("DEEPSEEK_API_KEY"):
        adapter = workspace / "skills" / "ace-rtl" / "scripts" / "deepseek_adapter.py"
        if not adapter.is_file():
            raise FileNotFoundError(f"DeepSeek adapter not found: {adapter}")
        os.environ["ACE_RTL_LLM_SCRIPT"] = str(adapter)
        return
    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError(
            "set NVIDIA_API_KEY, DEEPSEEK_API_KEY, or ACE_RTL_LLM_SCRIPT before a non-dry run"
        )


def _find_task_summary(
    output_root: Path, run_id: str, model_slug: str, task_id: str
) -> Dict[str, Any] | None:
    matches = sorted(
        output_root.glob(
            f"**/chipverilog/{run_id}/{model_slug}/{task_id}/summary.json"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        return None
    try:
        return json.loads(matches[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _suite_summary(
    *,
    run_id: str,
    model: str,
    started_at: str,
    elapsed: float,
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    status_counts = Counter(str(record.get("status") or "missing") for record in records)
    passed = sum(1 for record in records if record.get("success") is True)
    unknown = sum(
        1
        for record in records
        if record.get("status") in UNKNOWN_STATUSES or record.get("status") is None
    )
    llm_usage = {
        "calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "usage_available_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": 0.0,
    }
    for record in records:
        task_usage = record.get("llm_usage")
        if not isinstance(task_usage, dict):
            continue
        for key in llm_usage:
            value = task_usage.get(key, 0)
            if isinstance(value, (int, float)):
                llm_usage[key] += value
    return {
        "benchmark": "ChipVerilog",
        "run_id": run_id,
        "model": model,
        "started_at": started_at,
        "elapsed_seconds": elapsed,
        "task_count": len(records),
        "task_success": passed,
        "task_success_rate": passed / len(records) if records else 0.0,
        "unknown_count": unknown,
        "status_counts": dict(status_counts),
        "llm_usage": llm_usage,
        "config": config,
        "tasks": records,
    }


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    args.chipverilog_root = args.chipverilog_root.resolve()
    output_root = (args.output_root or (workspace / "outputs")).resolve()
    run_id = _safe_component(args.run_id or _run_id())
    model_slug = _safe_component(args.model)
    tasks = _select_tasks(args)

    if args.list_tasks:
        for task in tasks:
            print(f"{task.task_id}\t{task.family}\t{task.target_top}")
        return 0

    if args.dry_run:
        print(
            json.dumps(
                [
                    {
                        "task_id": task.task_id,
                        "family": task.family,
                        "target_top": task.target_top,
                        "include_files": [path.name for path in task.include_files],
                        "dependency_files": [path.name for path in task.dependency_files],
                        "prompt_chars": len(render_chipverilog_prompt(task)),
                    }
                    for task in tasks
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if not 1 <= args.max_iterations <= 100:
        raise ValueError("max-iterations must be between 1 and 100")
    if not 1 <= args.parallel_experiments <= 10:
        raise ValueError("parallel-experiments must be between 1 and 10")

    _configure_llm_adapter(workspace)
    os.environ["ACE_RTL_WORKSPACE"] = str(workspace)
    os.environ["ACE_RTL_RUN_ID"] = run_id
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / "chipverilog" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    config_payload = {
        "max_iterations": args.max_iterations,
        "parallel_experiments": args.parallel_experiments,
        "iteration_timeout": args.iteration_timeout,
        "eval_timeout": args.eval_timeout,
        "eval_depth": args.eval_depth,
        "llm_context_limit": args.llm_context_limit,
        "temperature": args.temperature,
        "task_kind": "chipverilog_rtl_implementation",
        "candidate_contract_fail_closed": True,
        "target_reference_exposed_to_roles": False,
    }
    _write_json(
        run_root / "run_metadata.json",
        {
            "benchmark": "ChipVerilog",
            "chipverilog_root": str(args.chipverilog_root),
            "run_id": run_id,
            "model": args.model,
            "started_at": started_at,
            "selected_tasks": [task.task_id for task in tasks],
            "config": config_payload,
        },
    )

    config = AgentConfig(
        max_iterations=args.max_iterations,
        model_name=args.model,
        output_dir=str(output_root),
        enable_debug=args.debug,
        save_intermediate=True,
        timeout_per_iteration=args.iteration_timeout,
        use_systematic_debugging=True,
        disable_testbench_access=True,
        max_retries=args.max_retries,
        initial_backoff=args.initial_backoff,
        llm_context_limit=args.llm_context_limit,
        num_parallel_experiments=args.parallel_experiments,
        ace_rtl_temperature=args.temperature,
        benchmark_mode="chipverilog",
        chipverilog_path=str(args.chipverilog_root),
        chipverilog_eval_depth=args.eval_depth,
        chipverilog_eval_timeout=args.eval_timeout,
    )
    agent = AceRTLAgent(config, api_key=None)
    records: List[Dict[str, Any]] = []
    try:
        for index, task in enumerate(tasks, start=1):
            print(f"[{index}/{len(tasks)}] {task.task_id}", flush=True)
            task_start = time.monotonic()
            try:
                result = agent.process_datapoint(task.task_id)
                record = _find_task_summary(output_root, run_id, model_slug, task.task_id) or {
                    "task_id": task.task_id,
                    "family": task.family,
                    "target_top": task.target_top,
                    "success": bool(result.get("success", False)),
                    "status": None,
                }
            except Exception as exc:
                logging.exception("ChipVerilog task failed: %s", task.task_id)
                record = {
                    "task_id": task.task_id,
                    "family": task.family,
                    "target_top": task.target_top,
                    "success": False,
                    "status": "runner_error",
                    "error": str(exc),
                }
            record["wall_time_seconds"] = time.monotonic() - task_start
            records.append(record)
            partial = _suite_summary(
                run_id=run_id,
                model=args.model,
                started_at=started_at,
                elapsed=time.monotonic() - start,
                records=records,
                config=config_payload,
            )
            _write_json(run_root / "run_summary.partial.json", partial)
            print(
                f"[{index}/{len(tasks)}] {task.task_id}: "
                f"{record.get('status') or ('pass' if record.get('success') else 'unknown')}",
                flush=True,
            )
    finally:
        agent.cleanup()

    summary = _suite_summary(
        run_id=run_id,
        model=args.model,
        started_at=started_at,
        elapsed=time.monotonic() - start,
        records=records,
        config=config_payload,
    )
    _write_json(run_root / "run_summary.json", summary)
    return 0 if summary["task_success"] == summary["task_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
