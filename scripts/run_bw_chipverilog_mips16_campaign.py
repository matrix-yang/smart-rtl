#!/usr/bin/env python3
"""Run an explicit ordered ChipVerilog task subset under one frozen protocol.

This restartable scheduler only assigns whole tasks. Search-branch decisions
remain the responsibility of each task's BW-RTL Director.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASKS = (
    "mips_EX_stage",
    "mips_ID_stage",
    "mips_IF_stage",
    "mips_MEM_stage",
    "mips_WB_stage",
    "mips_alu",
    "mips_data_mem",
    "mips_hazard_detection_unit",
    "mips_instruction_mem",
    "mips_mips_16_core_top",
    "mips_register_file",
)
TERMINAL_CONTAINER_STATES = {"exited", "dead"}
CREDENTIAL_ERROR_TERMS = (
    "insufficient balance",
    "insufficient_balance",
    "insufficient quota",
    "quota exceeded",
    "authentication",
    "invalid api key",
    "unauthorized",
    "payment required",
)
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": utc_now(), **event}, sort_keys=True) + "\n")


def run_command(
    command: list[str], cwd: Path, timeout: int = 180
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def controller_fingerprint(repo: Path) -> str:
    patterns = (
        "skills/bw-rtl/SKILL.md",
        "skills/bw-rtl/references/*.md",
        "skills/ace-rtl/scripts/ace_cvdp_native/rtl_search/*.py",
        "skills/ace-rtl/scripts/ace_rtl_agent/utils/chipverilog_integration.py",
        "skills/ace-rtl/scripts/deepseek_adapter.py",
        "compose.ace-rtl.yaml",
        "scripts/run_bw_chipverilog_mips16_campaign.py",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in repo.glob(pattern) if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(repo).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def image_id(repo: Path) -> str:
    completed = run_command(
        [
            "docker",
            "image",
            "inspect",
            "smart-rtl:deepseek",
            "--format",
            "{{.Id}}",
        ],
        repo,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError("cannot resolve frozen ChipVerilog image id")
    return completed.stdout.strip()


def container_field(repo: Path, name: str, template: str) -> str | None:
    completed = run_command(["docker", "inspect", name, "--format", template], repo)
    return completed.stdout.strip() if completed.returncode == 0 else None


def container_status(repo: Path, name: str) -> str | None:
    return container_field(repo, name, "{{.State.Status}}")


def container_exit_code(repo: Path, name: str) -> int | None:
    raw = container_field(repo, name, "{{.State.ExitCode}}")
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def container_has_credential_error(repo: Path, name: str) -> bool:
    completed = run_command(["docker", "logs", "--tail", "300", name], repo)
    text = completed.stdout.lower()
    return any(term in text for term in CREDENTIAL_ERROR_TERMS)


def result_summary(run_dir: Path) -> dict[str, Any]:
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
        "evaluator_subprocess_calls": result.get("evaluator_subprocess_calls"),
        "candidate_contract_rejections": result.get("candidate_contract_rejections"),
        "selected_candidate_hash": result.get("selected_candidate_hash"),
        "llm_calls": telemetry.get("calls"),
        "successful_llm_calls": telemetry.get("successful_calls"),
        "failed_llm_calls": telemetry.get("failed_calls"),
        "usage_totals": telemetry.get("usage_totals") or {},
        "role_errors": result.get("role_errors") or [],
    }
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        summary["observed_skill_version"] = metadata.get("skill_version")
        summary["observed_controller"] = metadata.get("controller")
    return summary


def parse_reuse(values: list[str], task_ids: tuple[str, ...]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--reuse must be TASK_ID=RUN_DIR, got {value!r}")
        task_id, raw = value.split("=", 1)
        if task_id not in task_ids:
            raise ValueError(f"reuse task is outside selected task set: {task_id}")
        parsed[task_id] = Path(raw).resolve()
    return parsed


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


def aggregate_summary(state: dict[str, Any]) -> dict[str, Any]:
    usage_keys = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    usage = {key: 0 for key in usage_keys}
    llm_calls = verifier_calls = 0
    for task in state["tasks"].values():
        llm_calls += int(task.get("llm_calls") or 0)
        verifier_calls += int(task.get("verifier_calls") or 0)
        for key in usage:
            usage[key] += int((task.get("usage_totals") or {}).get(key) or 0)
    return {
        "campaign_id": state["campaign_id"],
        "status": state.get("campaign_status", "running"),
        "counts": campaign_counts(state),
        "usage_totals": usage,
        "llm_calls": llm_calls,
        "verifier_calls": verifier_calls,
        "baseline_relation": "paired mips_16 task set; same official evaluator, different agent loop",
        "tasks": {
            task_id: {
                key: value
                for key, value in task.items()
                if key
                in {
                    "campaign_status",
                    "result_status",
                    "rounds",
                    "candidate_count",
                    "verifier_calls",
                    "llm_calls",
                    "usage_totals",
                    "run_dir",
                }
            }
            for task_id, task in state["tasks"].items()
        },
    }


def launch_task(
    repo: Path,
    compose_file: Path,
    task: dict[str, Any],
    physical_workers: int,
    max_rounds: int,
) -> subprocess.CompletedProcess[str]:
    relative_run_dir = Path(task["run_dir"]).resolve().relative_to(repo).as_posix()
    command_inside = " ".join(
        [
            "exec /opt/ace-rtl-venv/bin/python -m",
            "ace_cvdp_native.rtl_search.adaptive_chipverilog_live",
            "--workspace /workspace/ace",
            "--chipverilog-root /workspace/chipverilog",
            f"--id {task['id']}",
            f"--run-dir {relative_run_dir}",
            f"--physical-workers {physical_workers}",
            f"--max-rounds {max_rounds}",
            f"--planner-model {MODEL}",
            f"--generator-model {MODEL}",
            f"--analyst-model {MODEL}",
            f"--director-model {MODEL}",
            f"--debater-model {MODEL}",
            f"--judge-model {MODEL}",
            "--planner-max-tokens 8192",
            "--builder-max-tokens 32000",
            "--analyst-max-tokens 6144",
            "--director-max-tokens 8192",
            "--debater-max-tokens 6144",
            "--judge-max-tokens 4096",
            "--planner-reasoning-effort low",
            "--builder-reasoning-effort low",
            "--analyst-reasoning-effort low",
            "--director-reasoning-effort low",
            "--debater-reasoning-effort low",
            "--judge-reasoning-effort low",
            "--llm-timeout 1800",
            "--evaluator-depth 16",
            "--evaluator-timeout 120",
        ]
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
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--physical-workers", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--expected-code-fingerprint", required=True)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    parser.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help="Comma-separated explicit task order; defaults to all mips_16 tasks",
    )
    parser.add_argument("--family", default="mips_16")
    parser.add_argument("--container-prefix", default="smart-rtl-chip-mips16")
    parser.add_argument("--reuse", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency != 1:
        raise ValueError("this campaign is locked to cross-task concurrency=1")
    if args.physical_workers != 1:
        raise ValueError("this campaign is locked to physical_workers=1")
    repo = Path(args.repo).resolve()
    task_ids = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    if not task_ids:
        raise ValueError("--tasks must contain at least one task id")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("--tasks contains duplicate task ids")
    if not args.family.strip() or not args.container_prefix.strip():
        raise ValueError("--family and --container-prefix must be non-empty")
    campaign_dir = Path(args.campaign_dir).resolve()
    if repo not in campaign_dir.parents:
        raise ValueError("campaign directory must be inside the isolated repository")
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign_dir / "CAMPAIGN_STATE.json"
    events_path = campaign_dir / "EVENTS.jsonl"
    summary_path = campaign_dir / "CAMPAIGN_SUMMARY.json"
    compose_file = repo / "compose.ace-rtl.yaml"
    evaluator = Path(
        os.environ.get(
            "CHIPVERILOG_ROOT", str(repo / "external/ChipVerilog")
        )
    ).resolve() / "tools" / "formal_equivalence.py"
    reuse = parse_reuse(args.reuse, task_ids)

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not exported")
    if controller_fingerprint(repo) != args.expected_code_fingerprint:
        raise RuntimeError("controller code fingerprint differs before campaign")
    if file_sha256(evaluator) != args.expected_evaluator_sha256:
        raise RuntimeError("ChipVerilog evaluator differs before campaign")
    frozen_image_id = image_id(repo)

    if state_path.is_file():
        state = read_json(state_path)
        if state.get("task_order") != list(task_ids):
            raise RuntimeError("existing campaign task order mismatch")
        if state.get("expected_code_fingerprint") != args.expected_code_fingerprint:
            raise RuntimeError(
                "existing campaign controller fingerprint differs; start a new campaign "
                "instead of mixing controller versions"
            )
        if state.get("expected_evaluator_sha256") != args.expected_evaluator_sha256:
            raise RuntimeError(
                "existing campaign evaluator fingerprint differs; start a new campaign "
                "instead of mixing evaluator versions"
            )
        if state.get("docker_image_id") != frozen_image_id:
            raise RuntimeError(
                "existing campaign image differs; start a new campaign instead of "
                "mixing container images"
            )
    else:
        state = {
            "campaign_id": campaign_dir.name,
            "campaign_status": "running",
            "created_at": utc_now(),
            "benchmark": "ChipVerilog",
            "family": args.family,
            "task_order": list(task_ids),
            "controller_code_policy": "frozen_until_all_tasks_terminal",
            "expected_code_fingerprint": args.expected_code_fingerprint,
            "evaluator_path": str(evaluator),
            "expected_evaluator_sha256": args.expected_evaluator_sha256,
            "docker_image": "smart-rtl:deepseek",
            "docker_image_id": frozen_image_id,
            "compose_project": "smart-rtl",
            "model": MODEL,
            "concurrency": 1,
            "physical_workers_per_task": 1,
            "max_rounds": args.max_rounds,
            "evaluator_depth": 16,
            "evaluator_timeout": 120,
            "evaluator_seed": None,
            "algorithmic_selection_k": None,
            "paused_reason": None,
            "tasks": {},
        }
        for index, task_id in enumerate(task_ids, start=1):
            item: dict[str, Any] = {
                "id": task_id,
                "family": args.family,
                "index": index,
                "container": f"{args.container_prefix}-{index:02d}",
                "run_dir": str(campaign_dir / "tasks" / task_id),
                "campaign_status": "pending",
                "launch_attempts": 0,
            }
            if task_id in reuse:
                item.update(result_summary(reuse[task_id]))
                item["campaign_status"] = "reused"
                item["evidence_origin"] = "same frozen adapter/protocol pre-campaign smoke"
                if item.get("result_status") != "passed":
                    raise RuntimeError(f"refusing non-passing reuse for {task_id}")
            state["tasks"][task_id] = item
        persist_state(state_path, state)
        append_event(
            events_path,
            {
                "event": "campaign_created",
                "family": args.family,
                "task_count": len(task_ids),
                "reused_count": len(reuse),
            },
        )

    while True:
        drift_reason = None
        if controller_fingerprint(repo) != args.expected_code_fingerprint:
            drift_reason = "controller_code_drift"
        elif file_sha256(evaluator) != args.expected_evaluator_sha256:
            drift_reason = "evaluator_drift"
        elif image_id(repo) != state["docker_image_id"]:
            drift_reason = "docker_image_drift"
        if drift_reason and state.get("paused_reason") is None:
            state["paused_reason"] = drift_reason
            append_event(events_path, {"event": "campaign_paused", "reason": drift_reason})

        active = 0
        for task in state["tasks"].values():
            if task["campaign_status"] in {"reused", "terminal"}:
                continue
            observed = container_status(repo, task["container"])
            if observed in TERMINAL_CONTAINER_STATES:
                task.update(result_summary(Path(task["run_dir"])))
                task["container_exit_code"] = container_exit_code(repo, task["container"])
                task["container_status"] = observed
                task["campaign_status"] = "terminal"
                task["finished_at"] = utc_now()
                credential_failure = container_has_credential_error(repo, task["container"])
                append_event(
                    events_path,
                    {
                        "event": "task_terminal",
                        "task_id": task["id"],
                        "result_status": task.get("result_status"),
                        "exit_code": task.get("container_exit_code"),
                        "credential_failure": credential_failure,
                    },
                )
                if credential_failure and state.get("paused_reason") is None:
                    state["paused_reason"] = "credential_or_quota_failure"
                    append_event(
                        events_path,
                        {"event": "campaign_paused", "reason": state["paused_reason"]},
                    )
            elif observed is not None:
                task["campaign_status"] = "running"
                task["container_status"] = observed
                active += 1
            elif task["campaign_status"] == "running":
                if Path(task["run_dir"]).joinpath("result.json").is_file():
                    task.update(result_summary(Path(task["run_dir"])))
                    task["campaign_status"] = "terminal"
                    task["container_status"] = "missing_after_result"
                    task["finished_at"] = utc_now()
                else:
                    task["campaign_status"] = "terminal"
                    task["result_status"] = "orphaned_without_result"
                    task["container_status"] = "missing"
                    task["finished_at"] = utc_now()

        if state.get("paused_reason") is None and active < 1:
            next_task = next(
                (
                    task
                    for task in state["tasks"].values()
                    if task["campaign_status"] == "pending"
                ),
                None,
            )
            if next_task is not None:
                next_task["launch_attempts"] += 1
                launched = launch_task(
                    repo,
                    compose_file,
                    next_task,
                    args.physical_workers,
                    args.max_rounds,
                )
                if launched.returncode != 0:
                    next_task["last_launch_error"] = launched.stdout[-4000:]
                    if next_task["launch_attempts"] >= 2:
                        next_task["campaign_status"] = "terminal"
                        next_task["result_status"] = "launch_error"
                        next_task["finished_at"] = utc_now()
                    append_event(
                        events_path,
                        {
                            "event": "task_launch_failed",
                            "task_id": next_task["id"],
                            "attempt": next_task["launch_attempts"],
                        },
                    )
                else:
                    next_task["campaign_status"] = "running"
                    next_task["container_status"] = "running"
                    next_task["started_at"] = utc_now()
                    active = 1
                    append_event(
                        events_path,
                        {
                            "event": "task_launched",
                            "task_id": next_task["id"],
                            "container": next_task["container"],
                        },
                    )
                    print(f"[{utc_now()}] launched {next_task['id']}", flush=True)

        persist_state(state_path, state)
        write_json_atomic(summary_path, aggregate_summary(state))
        counts = state["counts"]
        print(
            f"[{utc_now()}] done={counts['terminal'] + counts['reused']}/{counts['total']} "
            f"passed={counts['passed']} running={counts['running']} "
            f"pending={counts['pending']} paused={state.get('paused_reason')}",
            flush=True,
        )

        if counts["terminal"] + counts["reused"] == counts["total"]:
            state["completed_at"] = utc_now()
            state["campaign_status"] = "completed"
            persist_state(state_path, state)
            write_json_atomic(summary_path, aggregate_summary(state))
            append_event(events_path, {"event": "campaign_completed", "counts": state["counts"]})
            return 0
        if state.get("paused_reason") and counts["running"] == 0:
            state["campaign_status"] = "paused"
            persist_state(state_path, state)
            write_json_atomic(summary_path, aggregate_summary(state))
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CAMPAIGN_SCHEDULER_ERROR: {type(exc).__name__}: {exc}")
        raise
