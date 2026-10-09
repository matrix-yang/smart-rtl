#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from mage.chipverilog import ChipVerilogMageAgent, discover_tasks, render_public_spec
from mage.gen_config import get_llm, set_exp_setting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chipverilog-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--model-name", default="mage_chipmate_v9b")
    parser.add_argument("--model-id", default=os.environ.get("MAGE_MODEL", "ChipMATE-V-9B"))
    parser.add_argument("--api-base-url", default=os.environ.get("MAGE_API_BASE_URL", "http://192.168.16.17:30000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("MAGE_API_KEY", "EMPTY"))
    parser.add_argument("--context-window", type=int, default=32768)
    parser.add_argument("--spec-char-budget", type=int, default=48000)
    parser.add_argument("--llm-parallel-requests", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--manifest-name", default="run_manifest.json")
    parser.add_argument("--modules", nargs="*")
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--editor-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Rerun tasks with valid summary.json files; default is resume/skip.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_completed_summary(task_dir: Path, task_id: str) -> dict | None:
    summary_path = task_dir / "summary.json"
    rtl_path = task_dir / "rtl.sv"
    if not summary_path.is_file() or not rtl_path.is_file() or rtl_path.stat().st_size == 0:
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {"task_id", "passed", "status", "input_tokens", "output_tokens", "golden_tool_calls"}
    if not required.issubset(summary) or summary.get("task_id") != task_id:
        return None
    if not isinstance(summary.get("passed"), bool) or not isinstance(summary.get("status"), str):
        return None
    return summary


def archive_existing_task_dir(output_root: Path, task_id: str) -> Path | None:
    task_dir = output_root / f"chipverilog_{task_id}"
    if not task_dir.exists():
        return None
    archive_root = output_root / "archived_attempts"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = archive_root / f"chipverilog_{task_id}__{stamp}"
    suffix = 1
    while destination.exists():
        destination = archive_root / f"chipverilog_{task_id}__{stamp}_{suffix}"
        suffix += 1
    shutil.move(str(task_dir), str(destination))
    return destination


def select_shard(tasks: list, num_shards: int, shard_index: int) -> list:
    if num_shards < 1:
        raise ValueError("num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    return tasks[shard_index::num_shards]


def main() -> int:
    args = parse_args()
    if args.context_window <= args.max_tokens:
        raise SystemExit("context-window must be larger than max-tokens")
    if args.llm_parallel_requests < 1:
        raise SystemExit("llm-parallel-requests must be >= 1")
    if Path(args.manifest_name).name != args.manifest_name:
        raise SystemExit("manifest-name must be a filename, not a path")
    tasks = discover_tasks(args.chipverilog_root.resolve())
    if args.modules:
        selected = set(args.modules)
        tasks = [task for task in tasks if task.task_id in selected]
        missing = selected - {task.task_id for task in tasks}
        if missing:
            raise SystemExit(f"Unknown modules: {sorted(missing)}")
    try:
        tasks = select_shard(tasks, args.num_shards, args.shard_index)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.dry_run:
        print(json.dumps([{
            "task_id": task.task_id,
            "family": task.family,
            "target_top": task.target_top,
            "dependencies": [str(path) for path in task.dependency_files],
            "includes": [str(path) for path in task.include_files],
            "spec_chars": len(render_public_spec(task)),
            "bounded_spec_chars": len(render_public_spec(task, args.spec_char_budget)),
        } for task in tasks], indent=2))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    os.environ["DEEPSEEK_API_BASE_URL"] = args.api_base_url
    os.environ["DEEPSEEK_API_KEY"] = args.api_key
    os.environ["MAGE_CONTEXT_WINDOW"] = str(args.context_window)
    llm = get_llm(
        model=args.model_id,
        cfg_path=None,
        max_token=args.max_tokens,
        provider="deepseek",
        skip_probe=False,
    )
    set_exp_setting(temperature=args.temperature, top_p=args.top_p)
    agent = ChipVerilogMageAgent(llm, args.max_candidates, args.editor_steps)
    agent.token_counter.max_parallel_requests = args.llm_parallel_requests
    agent.set_output_path(str(args.output_root))
    agent.set_log_path(str(args.log_root))

    run_start = time.monotonic()
    records = []
    skipped_completed = []
    archived_attempts = []
    for index, task in enumerate(tasks, start=1):
        task_dir = args.output_root / f"chipverilog_{task.task_id}"
        completed = load_completed_summary(task_dir, task.task_id)
        if completed is not None and not args.rerun_completed:
            resumed = dict(completed)
            resumed["resume_skipped"] = True
            records.append(resumed)
            skipped_completed.append(task.task_id)
            print(f"[{index}/{len(tasks)}] {task.task_id}: resume skip ({completed['status']})", flush=True)
            continue
        archived = archive_existing_task_dir(args.output_root, task.task_id)
        if archived is not None:
            archived_attempts.append(str(archived))
            print(f"[{index}/{len(tasks)}] archived previous attempt -> {archived}", flush=True)
        start = time.monotonic()
        print(f"[{index}/{len(tasks)}] {task.task_id}", flush=True)
        original_spec = render_public_spec(task)
        bounded_spec = render_public_spec(task, args.spec_char_budget)
        agent.run_chipverilog(
            task,
            bounded_spec,
            args.chipverilog_root.resolve(),
            args.model_name,
            original_spec_chars=len(original_spec),
        )
        summary_path = args.output_root / f"chipverilog_{task.task_id}" / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["wall_time_seconds"] = time.monotonic() - start
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        records.append(summary)
        print(f"[{index}/{len(tasks)}] {task.task_id}: {summary['status']}", flush=True)
    manifest = {
        "model": args.model_id,
        "api_base_url": args.api_base_url,
        "context_window": args.context_window,
        "spec_char_budget": args.spec_char_budget,
        "feedback": "ChipVerilog golden evaluator on every candidate/edit",
        "max_candidates": args.max_candidates,
        "selected_edit_trajectories": 2,
        "editor_steps": args.editor_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "llm_parallel_requests": args.llm_parallel_requests,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "resume_enabled": not args.rerun_completed,
        "skipped_completed": skipped_completed,
        "archived_attempts": archived_attempts,
        "total_wall_time_seconds": time.monotonic() - run_start,
        "tasks": records,
    }
    (args.output_root / args.manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
