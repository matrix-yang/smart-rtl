#!/usr/bin/env python3
"""Run every cid003 task through the frozen BW-RTL controller with bounded concurrency.

This is a deterministic campaign scheduler, not part of the search algorithm.  It
keeps each task in its own Docker container and writes a restartable state file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROLE_FILES = {
    "director": "director.md",
    "planner": "planner.md",
    "builder": "builder.md",
    "analyst": "analyst.md",
    "debater": "debater.md",
    "verifier": "verifier.md",
    "knowledge": "knowledge-ledger.md",
    "judge": "judge.md",
}
TERMINAL_CONTAINER_STATES = {"exited", "dead"}
CREDENTIAL_ERROR_TERMS = (
    "insufficient balance",
    "insufficient_balance",
    "insufficient quota",
    "quota exceeded",
    "authentication",
    "invalid api key",
    "unauthorized",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_event(path: Path, event: dict[str, Any]) -> None:
    event = {"at": utc_now(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def run_command(
    command: list[str], cwd: Path, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def skill_version(repo: Path) -> str:
    root = repo / "skills" / "bw-rtl"
    core = (root / "SKILL.md").read_text(encoding="utf-8")
    roles = {
        role: (root / "references" / filename).read_text(encoding="utf-8")
        for role, filename in ROLE_FILES.items()
    }
    payload = core + "\n" + "\n".join(roles[name] for name in sorted(roles))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cid003_tasks(dataset: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with dataset.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            categories = row.get("categories") or []
            if "cid003" not in categories:
                continue
            task_id = str(row["id"])
            if not re.fullmatch(r"[A-Za-z0-9_]+", task_id):
                raise ValueError(f"unsafe task id: {task_id!r}")
            difficulty = next(
                (value for value in categories if value in {"easy", "medium", "hard"}),
                "unknown",
            )
            tasks.append({"id": task_id, "difficulty": difficulty})
    return tasks


def parse_reuse(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--reuse must be TASK_ID=RUN_DIR, got {value!r}")
        task_id, run_dir = value.split("=", 1)
        parsed[task_id] = Path(run_dir).resolve()
    return parsed


def container_status(repo: Path, name: str) -> str | None:
    completed = run_command(
        ["docker", "inspect", name, "--format", "{{.State.Status}}"], repo
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def container_exit_code(repo: Path, name: str) -> int | None:
    completed = run_command(
        ["docker", "inspect", name, "--format", "{{.State.ExitCode}}"], repo
    )
    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return None


def container_run_dir(repo: Path, name: str) -> Path | None:
    completed = run_command(["docker", "logs", name], repo)
    for line in completed.stdout.splitlines():
        if not line.startswith("RUN_DIR="):
            continue
        inside = line.split("=", 1)[1]
        prefix = "/workspace/ace/"
        if inside.startswith(prefix):
            return repo / inside[len(prefix) :]
        return Path(inside)
    return None


def result_summary(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {"result_status": "missing_run_dir"}
    result_path = run_dir / "result.json"
    metadata_path = run_dir / "metadata.json"
    if not result_path.is_file():
        return {
            "run_dir": str(run_dir),
            "result_path": str(result_path),
            "result_status": "missing_result",
        }
    result = read_json(result_path)
    telemetry = result.get("call_telemetry") or {}
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "result_path": str(result_path),
        "result_status": result.get("status", "unknown"),
        "rounds": result.get("rounds"),
        "candidate_count": len(result.get("candidates") or []),
        "verifier_calls": result.get("verifier_calls"),
        "selected_candidate_hash": result.get("selected_candidate_hash"),
        "llm_calls": telemetry.get("calls"),
        "successful_llm_calls": telemetry.get("successful_calls"),
        "failed_llm_calls": telemetry.get("failed_calls"),
        "usage_totals": telemetry.get("usage_totals") or {},
        "role_errors": result.get("role_errors") or [],
    }
    if metadata_path.is_file():
        summary["observed_skill_version"] = read_json(metadata_path).get("skill_version")
    return summary


def is_credential_failure(summary: dict[str, Any]) -> bool:
    text = json.dumps(summary.get("role_errors") or [], sort_keys=True).lower()
    return any(term in text for term in CREDENTIAL_ERROR_TERMS)


def campaign_counts(state: dict[str, Any]) -> dict[str, int]:
    counts = {
        "total": 0,
        "reused": 0,
        "pending": 0,
        "running": 0,
        "terminal": 0,
        "passed": 0,
        "failed_or_error": 0,
    }
    for task in state["tasks"].values():
        counts["total"] += 1
        status = task["campaign_status"]
        if status in counts:
            counts[status] += 1
        if status in {"reused", "terminal"}:
            if task.get("result_status") == "passed":
                counts["passed"] += 1
            else:
                counts["failed_or_error"] += 1
    return counts


def persist_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    state["counts"] = campaign_counts(state)
    write_json_atomic(path, state)


def launch_task(
    repo: Path,
    compose_file: Path,
    task: dict[str, Any],
    physical_workers: int,
    max_rounds: int,
    evaluator_seed: int,
    role_max_tokens: dict[str, int],
) -> subprocess.CompletedProcess[str]:
    command_inside = (
        "exec /opt/ace-rtl-venv/bin/python3 "
        "-m ace_cvdp_native.rtl_search.adaptive_live "
        "--workspace /workspace/ace "
        "--cvdp-repo /workspace/ace/external/cvdp_benchmark "
        "--dataset /workspace/ace/datasets/cvdp-benchmark-dataset/"
        "cvdp_v1.1.0_agentic_code_generation_no_commercial.jsonl "
        f"--cid cid003 --id {task['id']} "
        f"--physical-workers {physical_workers} "
        f"--max-rounds {max_rounds} --evaluator-seed {evaluator_seed}"
        f" --planner-max-tokens {role_max_tokens['planner']}"
        f" --builder-max-tokens {role_max_tokens['builder']}"
        f" --analyst-max-tokens {role_max_tokens['analyst']}"
        f" --director-max-tokens {role_max_tokens['director']}"
        f" --debater-max-tokens {role_max_tokens['debater']}"
        f" --judge-max-tokens {role_max_tokens['judge']}"
    )
    return run_command(
        [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "run",
            "-d",
            "--name",
            task["container"],
            "--no-deps",
            "ace-rtl-agentic",
            "/bin/bash",
            "-lc",
            command_inside,
        ],
        repo,
        timeout=180,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--physical-workers", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--evaluator-seed", type=int, default=1)
    parser.add_argument("--planner-max-tokens", type=int, default=8192)
    parser.add_argument("--builder-max-tokens", type=int, default=32000)
    parser.add_argument("--analyst-max-tokens", type=int, default=6144)
    parser.add_argument("--director-max-tokens", type=int, default=8192)
    parser.add_argument("--debater-max-tokens", type=int, default=6144)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--container-prefix", default="bw-rtl-cid003-full")
    parser.add_argument("--expected-skill-version", required=True)
    parser.add_argument("--reuse", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    campaign_dir = Path(args.campaign_dir).resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign_dir / "CAMPAIGN_STATE.json"
    events_path = campaign_dir / "EVENTS.jsonl"
    dataset = (
        repo
        / "datasets/cvdp-benchmark-dataset/"
        "cvdp_v1.1.0_agentic_code_generation_no_commercial.jsonl"
    )
    compose_file = repo / "compose.ace-rtl.yaml"
    reuse = parse_reuse(args.reuse)
    role_max_tokens = {
        role: getattr(args, f"{role}_max_tokens")
        for role in ("planner", "builder", "analyst", "director", "debater", "judge")
    }
    if any(value < 1 for value in role_max_tokens.values()):
        raise ValueError("role max-token limits must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.container_prefix):
        raise ValueError("container prefix contains unsafe characters")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not exported")
    observed_version = skill_version(repo)
    if observed_version != args.expected_skill_version:
        raise RuntimeError(
            f"skill drift before campaign: expected {args.expected_skill_version}, "
            f"observed {observed_version}"
        )

    tasks = load_cid003_tasks(dataset)
    if len(tasks) != 34:
        raise RuntimeError(f"expected 34 cid003 tasks, found {len(tasks)}")

    if state_path.is_file():
        state = read_json(state_path)
        expected_ids = [task["id"] for task in tasks]
        if state.get("task_order") != expected_ids:
            raise RuntimeError("existing campaign state does not match dataset task order")
    else:
        state = {
            "campaign_id": campaign_dir.name,
            "created_at": utc_now(),
            "dataset": str(dataset),
            "dataset_sha256": file_sha256(dataset),
            "expected_skill_version": args.expected_skill_version,
            "controller_code_policy": "frozen_until_all_tasks_terminal",
            "concurrency": args.concurrency,
            "physical_workers_per_task": args.physical_workers,
            "max_rounds": args.max_rounds,
            "evaluator_seed": args.evaluator_seed,
            "role_max_tokens": role_max_tokens,
            "container_prefix": args.container_prefix,
            "algorithmic_selection_k": None,
            "task_order": [task["id"] for task in tasks],
            "tasks": {},
            "paused_reason": None,
        }
        for index, task in enumerate(tasks, start=1):
            item: dict[str, Any] = {
                **task,
                "index": index,
                "container": f"{args.container_prefix}-{index:02d}",
                "campaign_status": "pending",
                "launch_attempts": 0,
            }
            if task["id"] in reuse:
                summary = result_summary(reuse[task["id"]])
                item.update(summary)
                item["campaign_status"] = "reused"
                item["evidence_origin"] = "same frozen protocol pre-campaign run"
                if item.get("observed_skill_version") != args.expected_skill_version:
                    raise RuntimeError(f"reuse skill mismatch for {task['id']}")
            state["tasks"][task["id"]] = item
        persist_state(state_path, state)
        append_event(
            events_path,
            {
                "event": "campaign_created",
                "task_count": len(tasks),
                "reused_count": len(reuse),
            },
        )

    credential_failures = 0
    while True:
        if skill_version(repo) != args.expected_skill_version:
            state["paused_reason"] = "controller_or_skill_drift"
            append_event(events_path, {"event": "campaign_paused", "reason": state["paused_reason"]})

        active = 0
        for task in state["tasks"].values():
            if task["campaign_status"] in {"reused", "terminal"}:
                continue
            observed_container_status = container_status(repo, task["container"])
            if observed_container_status in TERMINAL_CONTAINER_STATES:
                run_dir = container_run_dir(repo, task["container"])
                summary = result_summary(run_dir)
                task.update(summary)
                task["container_exit_code"] = container_exit_code(repo, task["container"])
                task["container_status"] = observed_container_status
                task["campaign_status"] = "terminal"
                task["finished_at"] = utc_now()
                append_event(
                    events_path,
                    {
                        "event": "task_terminal",
                        "task_id": task["id"],
                        "container": task["container"],
                        "container_exit_code": task["container_exit_code"],
                        "result_status": task.get("result_status"),
                        "run_dir": task.get("run_dir"),
                    },
                )
                if is_credential_failure(task):
                    credential_failures += 1
                else:
                    credential_failures = 0
                if credential_failures >= 2:
                    state["paused_reason"] = "two_consecutive_credential_or_quota_failures"
                    append_event(
                        events_path,
                        {"event": "campaign_paused", "reason": state["paused_reason"]},
                    )
            elif observed_container_status is not None:
                task["campaign_status"] = "running"
                task["container_status"] = observed_container_status
                active += 1
            elif task["campaign_status"] == "running":
                task["campaign_status"] = "pending"
                task["container_status"] = "missing"

        if state.get("paused_reason") is None:
            for task in state["tasks"].values():
                if active >= args.concurrency:
                    break
                if task["campaign_status"] != "pending":
                    continue
                task["launch_attempts"] += 1
                launched = launch_task(
                    repo,
                    compose_file,
                    task,
                    args.physical_workers,
                    args.max_rounds,
                    args.evaluator_seed,
                    role_max_tokens,
                )
                if launched.returncode != 0:
                    task["last_launch_error"] = launched.stdout[-4000:]
                    append_event(
                        events_path,
                        {
                            "event": "task_launch_failed",
                            "task_id": task["id"],
                            "attempt": task["launch_attempts"],
                        },
                    )
                    if task["launch_attempts"] >= 2:
                        task["campaign_status"] = "terminal"
                        task["result_status"] = "launch_error"
                        task["finished_at"] = utc_now()
                    continue
                task["campaign_status"] = "running"
                task["container_status"] = "running"
                task["started_at"] = utc_now()
                active += 1
                append_event(
                    events_path,
                    {
                        "event": "task_launched",
                        "task_id": task["id"],
                        "container": task["container"],
                    },
                )
                print(
                    f"[{utc_now()}] launched {task['id']} as {task['container']}",
                    flush=True,
                )

        persist_state(state_path, state)
        counts = state["counts"]
        print(
            f"[{utc_now()}] terminal/reused={counts['terminal'] + counts['reused']}/"
            f"{counts['total']} passed={counts['passed']} running={counts['running']} "
            f"pending={counts['pending']} paused={state.get('paused_reason')}",
            flush=True,
        )

        if counts["terminal"] + counts["reused"] == counts["total"]:
            state["completed_at"] = utc_now()
            state["campaign_status"] = "completed"
            persist_state(state_path, state)
            append_event(events_path, {"event": "campaign_completed", "counts": state["counts"]})
            return 0
        if state.get("paused_reason") and counts["running"] == 0:
            state["campaign_status"] = "paused"
            persist_state(state_path, state)
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # Preserve a visible scheduler-level failure.
        print(f"CAMPAIGN_SCHEDULER_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
