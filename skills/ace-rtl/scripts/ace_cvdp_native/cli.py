# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACE-RTL native CVDP runner internals."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import logging
import multiprocessing
import os
import queue
import re
import shutil
import shlex
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from .config import *
from .processes import *
from .harness import *
from .sv import *
from .diagnostics import *
from .llm import *


def select_generation_processor_class(dataset_processor):
    """Return the processor used by the native iterative generation loop.

    Agentic CVDP rows are normalized to the Copilot input/output schema before
    reaching this function.  The official ``AgenticProcessor`` launches an
    external Docker agent and does not call this runner's ``model.prompt``;
    selecting it here would silently evaluate the unmodified input workspace.
    """

    processor = getattr(dataset_processor, "CopilotProcessor", None)
    if not isinstance(processor, type):
        raise AttributeError("CVDP CopilotProcessor is required for native generation")
    return processor


def resolve_target_files(row: Dict[str, Any]) -> List[str]:
    """Resolve only files explicitly declared as generated targets.

    ``patch`` is authoritative for agentic rows.  In particular, cid005's RTL
    context contains reusable child modules; those files must never become
    generation targets merely because target metadata is malformed.
    """

    task_id = row.get("id", "<unknown>")
    patch = row.get("patch")
    output = row.get("output")
    output_context = output.get("context") if isinstance(output, dict) else None

    if isinstance(patch, dict):
        candidates = list(patch.keys())
        if isinstance(output_context, dict) and list(output_context.keys()) != candidates:
            raise ValueError(
                f"agentic target mismatch for {task_id}: patch and output.context differ"
            )
    elif isinstance(output_context, dict):
        candidates = list(output_context.keys())
    else:
        raise ValueError(
            f"no declared generated targets for {task_id}; expected patch or output.context"
        )

    if not candidates:
        raise ValueError(f"declared generated target list is empty for {task_id}")
    invalid = [name for name in candidates if not isinstance(name, str) or not name.strip()]
    if invalid:
        raise ValueError(f"invalid generated target path(s) for {task_id}: {invalid}")
    return candidates


def normalize_agentic_row_for_native(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an agentic CVDP row without changing its evaluation contract.

    The Copilot processor needs ``input``/``output`` fields, while official
    agentic rows expose public files in ``context`` and writable targets in
    ``patch``.  Only target *paths* cross this boundary: patch contents are
    deliberately blanked so a dataset variant containing reference patches
    cannot leak them into model generation.
    """
    if not isinstance(row, dict):
        raise TypeError("CVDP row must be a JSON object")

    row = copy.deepcopy(row)
    patch = row.get("patch")
    if patch is None:
        # Native/non-agentic rows already use the processor's input/output
        # schema.  Validate their target contract before returning them.
        resolve_target_files(row)
        return row
    if not isinstance(patch, dict) or not patch:
        raise ValueError(f"agentic row {row.get('id')} has an empty or invalid patch map")

    context = row.get("context")
    if not isinstance(context, dict):
        raise ValueError(f"agentic row {row.get('id')} has no public context map")

    target_paths = list(patch.keys())
    existing_input = row.get("input")
    input_payload = copy.deepcopy(existing_input) if isinstance(existing_input, dict) else {}
    input_payload["prompt"] = str(row.get("prompt", input_payload.get("prompt", "")))
    input_payload["context"] = copy.deepcopy(context)
    row["input"] = input_payload

    existing_output = row.get("output")
    output_payload = copy.deepcopy(existing_output) if isinstance(existing_output, dict) else {}
    output_payload["context"] = {name: "" for name in target_paths}
    output_payload.setdefault("response", str(row.get("response", "")))
    row["output"] = output_payload

    resolve_target_files(row)
    return row


def run_attempt(
    *,
    workspace: Path,
    cvdp_repo: Path,
    row: Dict[str, Any],
    row_jsonl: Path,
    attempt_root: Path,
    attempt_index: int,
    args: argparse.Namespace,
    stop_event: Any,
) -> Dict[str, Any]:
    sys.path.insert(0, str(cvdp_repo))
    install_native_harness_patch()
    from src import dataset_processor

    task_id = row["id"]
    target_files = resolve_target_files(row)
    target_file = target_files[0]
    logger = logging.getLogger(f"ace_cvdp_native_runner.{task_id}")
    if len(target_files) > 1:
        logger.warning(
            "multiple target files detected for %s, using first declared file=%s: %s",
            task_id,
            target_file,
            ", ".join(target_files),
        )
    logger.debug("resolved target files for %s: %s", task_id, ", ".join(target_files))

    input_data = row.get("input") or {}
    specification_parts: List[str] = []
    prompt = str(input_data.get("prompt", "") or row.get("prompt", "") or "")
    if prompt:
        specification_parts.append(prompt)
    input_context = input_data.get("context") or {}
    if isinstance(input_context, dict):
        for name, content in input_context.items():
            specification_parts.append(f"\n# Context file: {name}\n```systemverilog\n{content}\n```")
    elif isinstance(row.get("context"), dict):
        for name, content in row.get("context", {}).items():
            specification_parts.append(f"\n# Context file: {name}\n```systemverilog\n{content}\n```")
    if not specification_parts:
        specification_parts.append("")
    specification = "\n".join(part for part in specification_parts if part)

    logger = logging.getLogger(f"ace_cvdp_native_runner.{task_id}.attempt{attempt_index}")
    focused_debugger = FocusedDebugger(
        logger,
        model_name=args.reflector_model,
        api_key=role_api_key(),
        max_retries=3,
        initial_backoff=5.0,
    )
    fresh_start_coordinator = FreshStartCoordinator(
        args.coordinator_model,
        logger,
        api_key=role_api_key(),
        max_retries=3,
        initial_backoff=5.0,
    )
    fresh_start_coordinator.focused_debugger = focused_debugger

    previous_candidates: Dict[str, str] = {}
    reflection = "Initial generation. No evaluator feedback is available yet."
    coordinator = "Generate a complete implementation from the public prompt and context."
    reflection_metadata: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []
    previous_candidate_hash: Optional[str] = None

    for iteration in range(1, args.max_iterations + 1):
        if stop_event.is_set():
            return {
                "attempt": attempt_index,
                "status": "stopped_after_sibling_pass",
                "history": history,
            }

        iteration_dir = attempt_root / f"iter_{iteration:02d}"
        work_prefix = iteration_dir / "cvdp_work"
        model = AceIterationModel(
            workspace=workspace,
            iteration_dir=iteration_dir,
            cid=args.cid,
            generator_model=args.generator_model,
            llm_timeout=args.llm_timeout,
            reflection=reflection,
            coordinator=coordinator,
            previous_candidate=previous_candidates or None,
        )

        processor_cls = select_generation_processor_class(dataset_processor)
        processor = processor_cls(
            filename=str(row_jsonl),
            golden=False,
            threads=1,
            debug=False,
            host=False,
            prefix=str(work_prefix),
            network_name=None,
            manage_network=False,
        )
        # These CVDP generation categories are evaluated by executable harness
        # services; avoid constructing an unrelated subjective scorer that would
        # require a separate subjective-scoring model key.
        processor._sbj_model_config = None
        processor._sbj_model_instance = None
        processor.process_json()
        prepared_id, obj, repo = processor.prepare(issue=task_id, model=model)
        if prepared_id != task_id:
            raise RuntimeError(f"unexpected prepared id {prepared_id} for {task_id}")
        prepared_harness_dir = Path(getattr(repo, "issue_path", "") or (work_prefix / task_id))
        fresh_start_coordinator.set_harness_directory(prepared_harness_dir)
        reflector_harness_context = build_reflector_harness_context(
            prepared_harness_dir,
            target_file,
        )

        if model.preflight_issues:
            result = make_preflight_result(model.preflight_issues, model.postprocess_actions)
            write_json(iteration_dir / "preflight_result.json", result)
        else:
            previous_native_timeout = os.environ.get("CVDP_NATIVE_TIMEOUT")
            os.environ["CVDP_NATIVE_TIMEOUT"] = str(args.evaluator_timeout)
            try:
                result = processor.run(id=task_id, obj=obj, repo=repo, model=model)
            finally:
                if previous_native_timeout is None:
                    os.environ.pop("CVDP_NATIVE_TIMEOUT", None)
                else:
                    os.environ["CVDP_NATIVE_TIMEOUT"] = previous_native_timeout
        write_json(iteration_dir / "real_evaluator_result.json", result)

        candidate_files = read_generated_candidate_files(model.generated_files)
        if target_file not in candidate_files:
            fallback_candidates = [
                name for name in target_files if name in candidate_files
            ]
            if fallback_candidates:
                target_file = fallback_candidates[0]
            elif candidate_files:
                target_file = sorted(candidate_files.keys())[0]
            else:
                raise RuntimeError(
                    f"generator produced no readable files for {task_id} attempt {attempt_index}"
                )
            logger.warning(
                "target file %s missing from generated outputs for %s, fallback to %s",
                target_files[0],
                task_id,
                target_file,
            )
        candidate_path = Path(model.generated_files[target_file])
        candidate = candidate_path.read_text(encoding="utf-8", errors="replace")
        current_hash = candidate_bundle_hash(candidate_files)
        candidate_changed = previous_candidate_hash is None or current_hash != previous_candidate_hash
        passed = result_passed(result)
        if passed:
            stop_event.set()
            if not try_claim_first_pass(
                dp_root=attempt_root.parent,
                task_id=task_id,
                attempt_index=attempt_index,
                iteration=iteration,
                candidate_path=candidate_path,
            ):
                claim = read_first_pass_claim(attempt_root.parent)
                logger.info(
                    "Suppressing duplicate PASS from attempt %s because attempt %s already claimed this datapoint",
                    attempt_index,
                    claim.get("attempt") if claim else "unknown",
                )
                history.append(
                    {
                        "iteration": iteration,
                        "passed": True,
                        "result": result,
                        "summary": "suppressed duplicate pass after sibling first-pass claim",
                        "primary_failure": {
                            "stage": "PASS",
                            "message": "candidate passed after a sibling already claimed first pass",
                            "signature": "duplicate pass suppressed",
                        },
                        "candidate_hash": current_hash,
                        "candidate_changed": candidate_changed,
                        "candidate_path": str(candidate_path),
                        "suppressed_duplicate_pass": True,
                    }
                )
                return {
                    "attempt": attempt_index,
                    "status": "stopped_after_sibling_pass",
                    "winner_attempt": claim.get("attempt") if claim else None,
                    "suppressed_duplicate_pass": True,
                    "passed_iteration": iteration,
                    "candidate_path": str(candidate_path),
                    "history": history,
                }
        summary = summarize_result(result)
        primary_failure = primary_failure_from_summary(result, summary)
        error_report = build_evaluator_error_report(result, summary)
        (iteration_dir / "evaluator_error_report.txt").write_text(
            error_report, encoding="utf-8", errors="replace"
        )
        record_reflector_iteration_result(
            focused_debugger=focused_debugger,
            iteration=iteration,
            error_report=error_report,
            metadata=reflection_metadata,
            passed=passed,
            code_before=previous_candidates.get(target_file, ""),
            code_after=candidate,
        )
        history.append(
            {
                "iteration": iteration,
                "passed": passed,
                "result": result,
                "summary": summary,
                "primary_failure": primary_failure,
                "candidate_hash": current_hash,
                "candidate_changed": candidate_changed,
                "candidate_path": str(candidate_path),
                "generated_files": dict(model.generated_files),
                "evaluator_error_report": str(iteration_dir / "evaluator_error_report.txt"),
            }
        )

        print(
            f"{task_id} attempt {attempt_index} iteration {iteration}: "
            f"{'PASS' if passed else 'FAIL'} ({primary_failure['stage']})"
        )
        if passed:
            return {
                "attempt": attempt_index,
                "status": "passed",
                "passed_iteration": iteration,
                "candidate_path": str(candidate_path),
                "history": history,
            }

        if primary_failure["stage"] == "INFRA_SETUP":
            stop_event.set()
            return {
                "attempt": attempt_index,
                "status": "infra_error",
                "failed_iteration": iteration,
                "candidate_path": str(candidate_path),
                "history": history,
            }

        if stop_event.is_set():
            return {
                "attempt": attempt_index,
                "status": "stopped_after_sibling_pass",
                "history": history,
            }

        coordinator, fresh_start_hint = call_coordinator(
            fresh_start_coordinator,
            iteration_dir,
            iteration,
            passed,
            candidate,
            error_report,
        )
        if fresh_start_hint:
            focused_debugger.clear_history()
            fresh_start_coordinator.mark_fresh_start(iteration)
            reflection = (
                "Fresh start requested after repeated no-progress functional failures. "
                "Ignore the previous candidate architecture and local repair history. "
                "Re-derive a complete implementation from only the public prompt and "
                "public context. Preserve the public module interface, parameters, reset "
                "polarity, clocking, latency/protocol requirements, and target file path. "
                "Ensure every checked output is driven by an active datapath on the same "
                "cycle as its valid/done flag, not left at reset/default values."
            )
            reflection_metadata = None

        previous_candidates = {} if fresh_start_hint else candidate_files
        previous_candidate_hash = None if fresh_start_hint else current_hash
        if iteration >= args.max_iterations:
            break

        if stop_event.is_set():
            return {
                "attempt": attempt_index,
                "status": "stopped_after_sibling_pass",
                "history": history,
            }

        if fresh_start_hint:
            continue

        reflection, reflection_metadata = call_reflector(
            focused_debugger,
            iteration_dir,
            task_id,
            target_file,
            candidate,
            error_report,
            iteration,
            specification,
            reflector_harness_context,
        )

    return {
        "attempt": attempt_index,
        "status": "failed",
        "history": history,
    }


def attempt_worker_main(
    result_queue: multiprocessing.Queue,
    run_log: str,
    attempt_kwargs: Dict[str, Any],
) -> None:
    """Run one datapoint attempt in an isolated process.

    Attempts must be OS processes rather than threads so the parent can stop
    sibling attempts immediately once one attempt passes, including attempts
    blocked in an LLM client call or evaluator subprocess.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = open(run_log, "a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    attempt_index = attempt_kwargs.get("attempt_index")
    try:
        result = run_attempt(**attempt_kwargs)
    except BaseException as exc:
        traceback.print_exc()
        error_message = str(exc)
        result = {
            "attempt": attempt_index,
            "status": "error",
            "error": error_message,
            "primary_failure": primary_failure_from_attempt_error(error_message),
            "traceback": traceback.format_exc(),
        }
    try:
        result_queue.put(result)
    finally:
        logging.shutdown()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


def stop_attempt_process(
    *,
    process: multiprocessing.Process,
    attempt_root: Path,
    dp_root: Path,
    winner_attempt: int,
    pre_signaled_pids: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    killed_pids: List[int] = list(pre_signaled_pids or [])
    killed_pids.extend(terminate_pids(attempt_process_pids(process, attempt_root), grace_seconds=0.2))
    process.join(timeout=0.5)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.5)
    killed_pids.extend(terminate_pids(pids_matching_path(attempt_root), grace_seconds=0.1))
    cleanup = cleanup_attempt_root(attempt_root, dp_root)
    return {
        "attempt": int(attempt_root.name.rsplit("_", 1)[-1]),
        "status": "stopped_after_sibling_pass",
        "winner_attempt": winner_attempt,
        "terminated_pids": sorted(set(killed_pids)),
        "cleanup": cleanup,
    }


def passed_result_from_claim(claim: Dict[str, Any], *, exitcode: Optional[int] = None) -> Dict[str, Any]:
    return {
        "attempt": int(claim["attempt"]),
        "status": "passed",
        "passed_iteration": claim.get("iteration"),
        "candidate_path": claim.get("candidate_path"),
        "first_pass_claim": claim,
        "result_source": "first_pass_claim",
        "exitcode": exitcode,
    }


def run_datapoint(
    *,
    workspace: Path,
    cvdp_repo: Path,
    row: Dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    task_id = row["id"]
    dp_root = run_dir / "datapoints" / task_id
    dp_root.mkdir(parents=True, exist_ok=True)
    row_jsonl = dp_root / "row.jsonl"
    write_jsonl(row_jsonl, [row])

    ctx = multiprocessing.get_context("spawn")
    stop_event = ctx.Event()
    result_queue: multiprocessing.Queue = ctx.Queue()
    attempt_results: List[Dict[str, Any]] = []
    processes: Dict[int, multiprocessing.Process] = {}
    attempt_roots: Dict[int, Path] = {}
    reported_attempts: set[int] = set()
    passed_attempt: Optional[int] = None

    def stop_sibling_attempts(winner_attempt: int) -> None:
        pre_signaled_pids: Dict[int, List[int]] = {}
        for sibling, process in processes.items():
            if sibling == winner_attempt:
                continue
            if process.is_alive():
                pre_signaled_pids[sibling] = request_attempt_stop(process, attempt_roots[sibling])

        for sibling, process in processes.items():
            if sibling == winner_attempt:
                continue
            if sibling in reported_attempts:
                if process.is_alive():
                    cleanup_result = stop_attempt_process(
                        process=process,
                        attempt_root=attempt_roots[sibling],
                        dp_root=dp_root,
                        winner_attempt=winner_attempt,
                        pre_signaled_pids=pre_signaled_pids.get(sibling),
                    )
                    for existing in attempt_results:
                        if existing.get("attempt") == sibling:
                            existing["cleanup_after_sibling_pass"] = cleanup_result
                            break
                else:
                    cleanup = cleanup_attempt_root(attempt_roots[sibling], dp_root)
                    for existing in attempt_results:
                        if existing.get("attempt") == sibling:
                            existing["cleanup_after_sibling_pass"] = cleanup
                            break
                write_json(dp_root / "attempt_results.partial.json", attempt_results)
                continue

            stopped = stop_attempt_process(
                process=process,
                attempt_root=attempt_roots[sibling],
                dp_root=dp_root,
                winner_attempt=winner_attempt,
                pre_signaled_pids=pre_signaled_pids.get(sibling),
            )
            reported_attempts.add(sibling)
            attempt_results.append(stopped)
            write_json(dp_root / "attempt_results.partial.json", attempt_results)

    try:
        for i in range(1, args.attempts + 1):
            attempt_root = dp_root / f"attempt_{i}"
            attempt_roots[i] = attempt_root
            process = ctx.Process(
                target=attempt_worker_main,
                args=(
                    result_queue,
                    str(run_dir / "run.log"),
                    {
                        "workspace": workspace,
                        "cvdp_repo": cvdp_repo,
                        "row": copy.deepcopy(row),
                        "row_jsonl": row_jsonl,
                        "attempt_root": attempt_root,
                        "attempt_index": i,
                        "args": args,
                        "stop_event": stop_event,
                    },
                ),
            )
            process.start()
            processes[i] = process

        while len(reported_attempts) < args.attempts:
            claim = read_first_pass_claim(dp_root)
            if claim and passed_attempt is None:
                passed_attempt = int(claim["attempt"])
                stop_event.set()
                stop_sibling_attempts(passed_attempt)

            try:
                result = result_queue.get(timeout=0.05)
            except queue.Empty:
                result = None

            if result is not None:
                attempt = result.get("attempt")
                if isinstance(attempt, int) and attempt not in reported_attempts:
                    if (
                        result.get("status") == "passed"
                        and passed_attempt is not None
                        and attempt != passed_attempt
                    ):
                        result = dict(result)
                        result["status"] = "stopped_after_sibling_pass"
                        result["winner_attempt"] = passed_attempt
                        result["suppressed_duplicate_pass"] = True
                    reported_attempts.add(attempt)
                    attempt_results.append(result)
                    write_json(dp_root / "attempt_results.partial.json", attempt_results)
                    if result.get("status") == "passed" and passed_attempt is None:
                        passed_attempt = attempt
                        stop_event.set()
                        stop_sibling_attempts(passed_attempt)

            for attempt, process in list(processes.items()):
                if attempt in reported_attempts or process.is_alive():
                    continue
                process.join(timeout=0)
                if attempt not in reported_attempts:
                    if passed_attempt is not None and attempt == passed_attempt:
                        claim = read_first_pass_claim(dp_root)
                        if claim and int(claim["attempt"]) == attempt:
                            result = passed_result_from_claim(claim, exitcode=process.exitcode)
                        else:
                            result = {
                                "attempt": attempt,
                                "status": "error",
                                "exitcode": process.exitcode,
                                "error": "winning attempt exited without reporting a result or first-pass claim",
                                "primary_failure": primary_failure_from_attempt_error(
                                    "winning attempt exited without reporting a result or first-pass claim"
                                ),
                            }
                    elif passed_attempt is not None:
                        result = {
                            "attempt": attempt,
                            "status": "stopped_after_sibling_pass",
                            "exitcode": process.exitcode,
                            "winner_attempt": passed_attempt,
                            "cleanup": cleanup_attempt_root(attempt_roots[attempt], dp_root),
                        }
                    else:
                        result = {
                            "attempt": attempt,
                            "status": "error",
                            "exitcode": process.exitcode,
                            "primary_failure": primary_failure_from_attempt_error(
                                "attempt process exited without reporting a result"
                            ),
                        }
                        result["error"] = "attempt process exited without reporting a result"
                    reported_attempts.add(attempt)
                    attempt_results.append(result)
                    write_json(dp_root / "attempt_results.partial.json", attempt_results)
    except KeyboardInterrupt:
        stop_event.set()
        for attempt, process in processes.items():
            if attempt not in reported_attempts:
                stop_attempt_process(
                    process=process,
                    attempt_root=attempt_roots[attempt],
                    dp_root=dp_root,
                    winner_attempt=-1,
                )
        raise
    finally:
        stop_event.set()
        for attempt, process in processes.items():
            if process.is_alive() and attempt not in reported_attempts:
                terminate_processes_for_path(attempt_roots[attempt])
                terminate_process_tree(process.pid)
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
        result_queue.close()
        result_queue.join_thread()

    passed_attempts = [r for r in attempt_results if r.get("status") == "passed"]
    claim = read_first_pass_claim(dp_root)
    if claim and not passed_attempts:
        winner_attempt = int(claim["attempt"])
        recovered = passed_result_from_claim(claim, exitcode=None)
        recovered["result_source"] = "first_pass_claim_final_recovery"
        for idx, existing in enumerate(attempt_results):
            if existing.get("attempt") == winner_attempt:
                attempt_results[idx] = recovered
                break
        else:
            attempt_results.append(recovered)
        passed_attempts = [recovered]
    final = {
        "id": task_id,
        "category": row.get("categories", [None])[0],
        "difficulty": row.get("categories", [None, None])[1],
        "passed": bool(passed_attempts),
        "passed_attempt": passed_attempts[0].get("attempt") if passed_attempts else None,
        "stopped_after_first_pass": sum(
            1 for r in attempt_results if r.get("status") == "stopped_after_sibling_pass"
        ),
        "attempts": attempt_results,
    }
    write_json(dp_root / "result.json", final)
    return final


def write_results_md(
    *,
    path: Path,
    dataset: Path,
    cid: str,
    role_map: Dict[str, str],
    cvdp_repo: Path,
    cvdp_commit: str,
    rows: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    passed_ids = sorted(r["id"] for r in results if r.get("passed"))
    failed_ids = sorted(r["id"] for r in results if not r.get("passed"))
    total = len(rows)
    pass_rate = (len(passed_ids) / total) if total else 0.0
    stopped = sum(
        1
        for r in results
        for a in r.get("attempts", [])
        if a.get("status") == "stopped_after_sibling_pass"
    )
    text = f"""\
# ACE-RTL CVDP Results

- Dataset: `{dataset}`
- CID: `{cid}`
- Row count: {total}
- Role models: `{json.dumps(role_map, sort_keys=True)}`
- CVDP repo: `{cvdp_repo}`
- CVDP commit: `{cvdp_commit}`
- Backend: workspace-local native harness (`CVDP_NATIVE_HARNESS=1`)
- Datapoint workers: {args.datapoint_workers}
- Iteration budget: {args.max_iterations} per attempt
- Attempts per datapoint: {args.attempts}
- Sibling attempts stopped after pass: {stopped}

## Pass Rate

- {len(passed_ids)} / {total} = {pass_rate:.4f}

## Passed IDs

{chr(10).join(f"- `{x}`" for x in passed_ids) if passed_ids else "- None"}

## Failed IDs

{chr(10).join(f"- `{x}`" for x in failed_ids) if failed_ids else "- None"}

## Artifact Paths

- Raw result summary: `summary.json`
- Per-datapoint artifacts: `datapoints/<id>/`
- Run log: `run.log`

Integrity note: role prompts used the workspace-local CVDP task data, generated
candidates, role history, and evaluator reports produced by the local native
harness. No external scripts are required at runtime.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=os.getcwd())
    parser.add_argument("--cvdp-repo", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cid", required=True)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--generator-model", default="nvidia/nemotron-3-ultra-550b-a55b")
    parser.add_argument("--reflector-model", default="nvidia/nemotron-3-ultra-550b-a55b")
    parser.add_argument("--coordinator-model", default="nvidia/nemotron-3-ultra-550b-a55b")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument(
        "--datapoint-workers",
        type=int,
        default=4,
        help="Number of CVDP datapoints to run in parallel by default",
    )
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--llm-timeout", type=int, default=1800)
    parser.add_argument(
        "--evaluator-timeout",
        type=int,
        default=env_int(
            "CVDP_NATIVE_TIMEOUT",
            env_int("DOCKER_TIMEOUT", DEFAULT_EVALUATOR_TIMEOUT_SECONDS),
        ),
        help="Seconds before a native CVDP evaluator service is killed and reported as candidate timeout",
    )
    parser.add_argument("--id", action="append", default=None, help="Optional datapoint id filter")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for shakedown/debug")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    maybe_reexec_workspace_python(workspace)
    dataset = ensure_under(Path(args.dataset), workspace, "dataset")
    outputs_root = ensure_under(Path(args.outputs_root), workspace, "outputs root")
    cvdp_repo = Path(args.cvdp_repo or (workspace / "external/cvdp_benchmark")).resolve()
    ensure_under(cvdp_repo, workspace, "CVDP repo")
    if not (cvdp_repo / "run_benchmark.py").exists():
        raise FileNotFoundError(f"CVDP repo not found: {cvdp_repo}")
    # Backward-compatible auto-binding: if only DeepSeek API key is provided,
    # use the repository's default deepseek adapter automatically.
    if not os.environ.get("ACE_RTL_LLM_SCRIPT") and os.environ.get("DEEPSEEK_API_KEY"):
        default_adapter = (workspace / "skills/ace-rtl/scripts/deepseek_adapter.py").resolve()
        if default_adapter.exists():
            os.environ["ACE_RTL_LLM_SCRIPT"] = str(default_adapter)
            print(f"Auto-binding ACE_RTL_LLM_SCRIPT={default_adapter}")
        elif not role_api_key():
            raise RuntimeError(
                "NVIDIA_API_KEY must be set (or ACE_RTL_LLM_SCRIPT must point to a valid adapter). "
                "DEEPSEEK_API_KEY is set but default deepseek adapter was not found."
            )

    if not role_api_key() and not os.environ.get("ACE_RTL_LLM_SCRIPT"):
        raise RuntimeError("NVIDIA_API_KEY must be set, or set ACE_RTL_LLM_SCRIPT")

    run_dir = unique_run_dir(outputs_root, args.cid, dataset.stem)
    log_file = (run_dir / "run.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print(f"RUN_DIR={run_dir}")

    sys.path.insert(0, str(cvdp_repo))
    install_native_harness_patch()

    commit = git_commit_or_unknown(cvdp_repo)
    rows = [normalize_agentic_row_for_native(row) for row in load_cid_rows(dataset, args.cid, args.id)]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"no rows found for {args.cid} in {dataset}")
    filtered = run_dir / f"{dataset.stem}_{args.cid}.jsonl"
    write_jsonl(filtered, rows)

    role_map = {
        "generator": args.generator_model,
        "reflector": args.reflector_model,
        "coordinator": args.coordinator_model,
    }
    write_json(
        run_dir / "metadata.json",
        {
            "dataset": str(dataset),
            "filtered_dataset": str(filtered),
            "cid": args.cid,
            "row_count": len(rows),
            "role_map": role_map,
            "cvdp_repo": str(cvdp_repo),
            "cvdp_commit": commit,
            "attempts": args.attempts,
            "datapoint_workers": args.datapoint_workers,
            "max_iterations": args.max_iterations,
            "evaluator_timeout": args.evaluator_timeout,
            "native_harness": True,
        },
    )

    indexed_rows = list(enumerate(rows, start=1))
    eda_setup = apply_eda_tool_setup(rows)
    tool_preflight = native_tool_preflight(rows)
    tool_preflight["eda_tool_setup"] = eda_setup
    write_json(run_dir / "native_tool_preflight.json", tool_preflight)
    if not tool_preflight["ok"]:
        missing = ", ".join(tool_preflight["missing_tools"])
        print(f"INFRA_SETUP: missing required native simulator tool(s): {missing}")
        for row_id, tools in sorted(tool_preflight["missing_by_row"].items()):
            print(f"INFRA_SETUP: {row_id} requires: {', '.join(tools)}")
        results = [
            {
                "_row_index": index,
                "id": row["id"],
                "category": row.get("categories", [None])[0],
                "difficulty": row.get("categories", [None, None])[1],
                "passed": False,
                "error": (
                    "INFRA_SETUP: missing required native simulator tool(s): "
                    + ", ".join(tool_preflight["missing_by_row"].get(row["id"], []))
                ),
                "attempts": [],
            }
            for index, row in indexed_rows
        ]
        for result in results:
            result.pop("_row_index", None)
        summary = {
            "dataset": str(dataset),
            "cid": args.cid,
            "row_count": len(rows),
            "passed_ids": [],
            "failed_ids": sorted(r["id"] for r in results),
            "preflight_failed": True,
            "native_tool_preflight": tool_preflight,
            "results": results,
        }
        write_json(run_dir / "summary.json", summary)
        write_results_md(
            path=run_dir / "results.md",
            dataset=dataset,
            cid=args.cid,
            role_map=role_map,
            cvdp_repo=cvdp_repo,
            cvdp_commit=commit,
            rows=rows,
            results=results,
            args=args,
        )
        print(f"completed run with INFRA_SETUP preflight failure: {run_dir}")
        return 2

    results: List[Dict[str, Any]] = []
    datapoint_workers = max(1, min(args.datapoint_workers, len(indexed_rows)))

    def run_indexed_datapoint(index: int, row: Dict[str, Any]) -> Dict[str, Any]:
        print(f"=== Datapoint {index}/{len(rows)}: {row['id']} ===")
        result = run_datapoint(
            workspace=workspace,
            cvdp_repo=cvdp_repo,
            row=row,
            run_dir=run_dir,
            args=args,
        )
        result["_row_index"] = index
        return result

    if datapoint_workers == 1:
        for index, row in indexed_rows:
            result = run_indexed_datapoint(index, row)
            results.append(result)
            write_json(run_dir / "summary.partial.json", results)
    else:
        print(f"Running up to {datapoint_workers} CVDP datapoints in parallel")
        with concurrent.futures.ThreadPoolExecutor(max_workers=datapoint_workers) as executor:
            futures = {
                executor.submit(run_indexed_datapoint, index, row): (index, row)
                for index, row in indexed_rows
            }
            for future in concurrent.futures.as_completed(futures):
                index, row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "_row_index": index,
                        "id": row["id"],
                        "category": row.get("categories", [None])[0],
                        "difficulty": row.get("categories", [None, None])[1],
                        "passed": False,
                        "error": str(exc),
                        "attempts": [],
                    }
                results.append(result)
                results.sort(key=lambda item: item.get("_row_index", 0))
                write_json(run_dir / "summary.partial.json", results)

    for result in results:
        result.pop("_row_index", None)

    summary = {
        "dataset": str(dataset),
        "cid": args.cid,
        "row_count": len(rows),
        "passed_ids": sorted(r["id"] for r in results if r.get("passed")),
        "failed_ids": sorted(r["id"] for r in results if not r.get("passed")),
        "results": results,
    }
    write_json(run_dir / "summary.json", summary)
    write_results_md(
        path=run_dir / "results.md",
        dataset=dataset,
        cid=args.cid,
        role_map=role_map,
        cvdp_repo=cvdp_repo,
        cvdp_commit=commit,
        rows=rows,
        results=results,
        args=args,
    )
    print(f"completed run: {run_dir}")
    return 0
