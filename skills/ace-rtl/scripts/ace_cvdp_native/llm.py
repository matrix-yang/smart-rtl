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

from ace_rtl_agent.focused_debugger import FocusedDebugger
from ace_rtl_agent.fresh_start_coordinator import FreshStartCoordinator
from .config import *
from .sv import *
from .diagnostics import *
from .processes import write_json

def run_role_call(
    *,
    workspace: Path,
    component: str,
    model: str,
    system: str,
    prompt: str,
    out_json: Path,
    timeout: int,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    thinking: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    response_format: str = "text",
) -> str:
    prompt_path = out_json.with_suffix(".prompt.txt")
    stdout_path = out_json.with_suffix(".stdout.txt")
    prompt_path.write_text(prompt, encoding="utf-8")
    cmd = [
        sys.executable,
        str(workspace / "skills/ace-rtl/scripts/llm_call.py"),
        "--component",
        component,
        "--model",
        model,
        "--system",
        system,
        "--prompt-file",
        str(prompt_path),
        "--output",
        str(out_json),
        "--timeout",
        str(timeout),
        "--temperature",
        str(temperature),
    ]
    if max_tokens is not None:
        cmd.extend(["--max-tokens", str(max_tokens)])
    if thinking is not None:
        cmd.extend(["--thinking", thinking])
    if reasoning_effort is not None:
        cmd.extend(["--reasoning-effort", reasoning_effort])
    if response_format != "text":
        cmd.extend(["--response-format", response_format])
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=str(workspace),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout + 120,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{component} LLM call failed; see {stdout_path}")
    return load_response_content(out_json)


class AceIterationModel:
    def __init__(
        self,
        *,
        workspace: Path,
        iteration_dir: Path,
        cid: str,
        generator_model: str,
        llm_timeout: int,
        reflection: str,
        coordinator: str,
        previous_candidate: Optional[Any],
    ) -> None:
        self.workspace = workspace
        self.iteration_dir = iteration_dir
        self.cid = cid
        self.generator_model = generator_model
        self.llm_timeout = llm_timeout
        self.reflection = reflection
        self.coordinator = coordinator
        self.previous_candidate = previous_candidate
        self.requires_evaluation = True
        self.generated_files: Dict[str, str] = {}
        self.preflight_issues: List[str] = []
        self.postprocess_actions: List[str] = []

    def prompt(
        self,
        prompt: str,
        schema: str = None,
        prompt_log: str = "",
        files: Optional[List[str]] = None,
        timeout: int = 60,
        category: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        files = files or []
        if not files:
            raise ValueError("ACE-RTL runner expected at least one target file")

        self.iteration_dir.mkdir(parents=True, exist_ok=True)
        if prompt_log:
            Path(prompt_log).parent.mkdir(parents=True, exist_ok=True)

        target_file = files[0]
        candidate_root = self.iteration_dir / "candidate"
        candidate_path = candidate_root / target_file
        candidate_path.parent.mkdir(parents=True, exist_ok=True)

        previous = format_candidate_bundle(files, self.previous_candidate)
        task_kind = classify_task(target_file, prompt)
        task_guidance = guidance_for_task(task_kind)
        response_label = response_label_for_task(task_kind, target_file, files)
        target_file_list = "\n".join(f"- {name}" for name in files)
        if len(files) > 1:
            output_contract = (
                "Return every target file listed above as a complete replacement. "
                "Use one section per file headed exactly by `// TARGET_FILE: <path>`, "
                "where `<path>` matches the target file path. Do not omit unchanged "
                "targets, do not add undeclared target files, and do not return a patch."
            )
        else:
            output_contract = (
                f"Return the complete {response_label} only. Do not include JSON or commentary."
            )
        full_prompt = f"""\
{prompt}

ACE-RTL task guidance:
{task_guidance}

Target file(s):
{target_file_list}

Coordinator guidance:
{self.coordinator}

Reflector guidance:
{self.reflection}

Previous candidate, if any:
```systemverilog
{previous}
```

{output_contract}
"""
        system = system_prompt_for_task(task_kind, multi_file=len(files) > 1)
        if prompt_log:
            Path(prompt_log).write_text(system + "\n\n" + full_prompt, encoding="utf-8")

        response_json = self.iteration_dir / "generator.response.json"
        run_role_call(
            workspace=self.workspace,
            component="generator",
            model=self.generator_model,
            system=system,
            prompt=full_prompt,
            out_json=response_json,
            timeout=self.llm_timeout,
        )

        extract_stdout = self.iteration_dir / "extract.stdout.txt"
        cmd = [
            sys.executable,
            str(self.workspace / "skills/ace-rtl/scripts/extract_solution.py"),
            "--input",
            str(response_json),
        ]
        if len(files) > 1:
            cmd.extend(["--target-dir", str(candidate_root)])
        else:
            cmd.extend(["--output", str(candidate_path)])
        proc = subprocess.run(
            cmd,
            cwd=str(self.workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        extract_stdout.write_text(proc.stdout, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"generator extraction failed; see {extract_stdout}")

        generated_fallback = candidate_root / "generated.sv"
        if len(files) > 1 and not candidate_path.exists() and generated_fallback.exists():
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(generated_fallback, candidate_path)
            self.postprocess_actions.append(
                f"mapped unsectioned multi-file response to primary target {target_file}"
            )

        missing_targets = [name for name in files if not (candidate_root / name).exists()]
        if missing_targets:
            self.preflight_issues.extend(
                f"generator response omitted declared target file: {name}"
                for name in missing_targets
            )

        normalized_contents: Dict[str, str] = {}
        file_actions: List[str] = []
        for name in files:
            path = candidate_root / name
            if not path.exists():
                continue
            raw_code = path.read_text(encoding="utf-8", errors="replace")
            code, actions = normalize_generated_sv(raw_code)
            if code != raw_code:
                raw_path = path.with_suffix(path.suffix + ".raw")
                raw_path.write_text(raw_code, encoding="utf-8")
                path.write_text(code, encoding="utf-8")
            normalized_contents[name] = code
            file_actions.extend(f"{name}: {action}" for action in actions)
            self.preflight_issues.extend(
                f"{name}: {issue}" for issue in high_confidence_candidate_issues(code, self.cid)
            )
            self.generated_files[name] = str(path)

        self.postprocess_actions.extend(file_actions)
        if target_file not in normalized_contents:
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text("", encoding="utf-8")
            normalized_contents[target_file] = ""
            self.generated_files[target_file] = str(candidate_path)

        code = normalized_contents[target_file]
        write_json(
            self.iteration_dir / "postprocess_result.json",
            {
                "target_file": target_file,
                "target_files": files,
                "candidate_hash": candidate_hash(code),
                "candidate_bundle_hash": candidate_bundle_hash(normalized_contents),
                "file_hashes": {
                    name: candidate_hash(content)
                    for name, content in sorted(normalized_contents.items())
                },
                "generated_files": self.generated_files,
                "actions": self.postprocess_actions,
                "preflight_issues": self.preflight_issues,
            },
        )
        if len(files) == 1:
            return {"direct_text": code}, True
        return {"code": [{name: normalized_contents[name]} for name in files if name in normalized_contents]}, True


def role_api_key() -> Optional[str]:
    return os.environ.get("NVIDIA_API_KEY")


def build_evaluator_error_report(result: Dict[str, Any], summary: str) -> str:
    parts = [
        "# CVDP evaluator summary",
        "```json",
        summary,
        "```",
    ]
    remaining = MAX_REPORT_BYTES
    native_reserve = min(NATIVE_RUNDIR_REPORT_RESERVE_BYTES, MAX_REPORT_BYTES // 4)
    native_rundirs: List[Path] = []
    for idx, test in enumerate(result.get("tests", []), start=1):
        if not isinstance(test, dict):
            continue
        header = (
            f"\n# Service result {idx}: result={test.get('result')} "
            f"execution={test.get('execution')} log={test.get('log')}"
        )
        parts.append(header)
        remaining -= len(header.encode("utf-8", errors="replace"))
        error_msg = test.get("error_msg")
        if error_msg:
            block = f"\nerror_msg: {error_msg}\n"
            parts.append(block)
            remaining -= len(block.encode("utf-8", errors="replace"))
        raw_log = test.get("log")
        if remaining <= 0 or not raw_log:
            continue
        log_path = Path(raw_log)
        if not log_path.exists():
            continue
        try:
            available_for_log = remaining - native_reserve
            if available_for_log < 4096:
                available_for_log = remaining
            log_budget = max(0, min(MAX_SERVICE_LOG_BYTES, available_for_log, remaining))
            if log_budget <= 0:
                continue
            text = read_bounded_text(log_path, log_budget)
        except OSError as exc:
            text = f"[could not read evaluator log: {exc}]"
        native_rundirs.extend(rundir_candidates_from_service_log(log_path, text))
        parts.append("\n```text\n")
        parts.append(text)
        parts.append("\n```")
        remaining -= len(text.encode("utf-8", errors="replace"))
    native_diagnostics = collect_native_rundir_diagnostics(native_rundirs, remaining)
    if native_diagnostics:
        parts.append(native_diagnostics)
    return "\n".join(parts)


def build_reflector_harness_context(
    harness_dir: Optional[Path],
    target_file: str,
    max_total_chars: int = 90000,
) -> str:
    """Read bounded, non-answer-bearing CVDP context for reflector feedback."""
    if not harness_dir:
        return ""
    root = Path(harness_dir)
    if not root.exists() or not root.is_dir():
        return ""

    target_norm = target_file.replace("\\", "/")
    is_rtl_generation_target = target_norm.startswith("rtl/")
    parts: List[str] = [
        "# === CVDP HARNESS CONTEXT ===",
        (
            "Sanitized for RTL repair: evaluator source, checker code, and "
            "reference-model implementations are intentionally omitted."
            if is_rtl_generation_target
            else "Bounded immutable context for non-RTL target repair."
        ),
    ]
    total = sum(len(part) for part in parts)

    def add_file(path: Path, label: str, max_file_chars: int = 20000) -> None:
        nonlocal total
        if total >= max_total_chars or not path.exists() or not path.is_file():
            return
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        if len(content) > max_file_chars:
            marker = "\n... [FILE TRUNCATED] ...\n"
            keep = max_file_chars - len(marker)
            head = max(0, int(keep * 0.65))
            tail = max(0, keep - head)
            content = content[:head] + marker + content[-tail:]
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = str(path)
        block = f"\n# File: {rel} ({label})\n{content}\n"
        remaining = max_total_chars - total
        if len(block) > remaining:
            block = block[:remaining] + "\n... [HARNESS CONTEXT TRUNCATED] ...\n"
        parts.append(block)
        total += len(block)

    add_file(root / "docker-compose.yml", "native service definitions", max_file_chars=12000)

    src_dir = root / "src"
    if src_dir.exists() and not is_rtl_generation_target:
        for pattern, label in [
            (".env", "CVDP env/scoring knobs"),
            ("*.py", "CVDP pytest runner/checker"),
            ("*.sv", "immutable source/UUT HDL"),
            ("*.v", "immutable source/UUT HDL"),
            ("*.svh", "source include"),
            ("*.vh", "source include"),
        ]:
            for path in sorted(src_dir.glob(pattern)):
                if path.is_file() and path.name not in {"__init__.py"}:
                    add_file(path, label)

    # Include RTL helper files only when they are not the generated target.
    # The candidate itself is already passed separately as BUGGY CODE.
    rtl_dir = root / "rtl"
    if rtl_dir.exists():
        for pattern in ("*.sv", "*.v", "*.svh", "*.vh"):
            for path in sorted(rtl_dir.glob(pattern)):
                try:
                    rel = path.relative_to(root).as_posix()
                except ValueError:
                    rel = str(path)
                if path.is_file() and rel != target_norm:
                    add_file(path, "immutable/helper RTL context")

    return "\n".join(parts) if len(parts) > 1 else ""


def call_reflector(
    focused_debugger: FocusedDebugger,
    iteration_dir: Path,
    task_id: str,
    target_file: str,
    candidate: str,
    error_report: str,
    iteration: int,
    specification: str,
    harness_context: str,
) -> Tuple[str, Dict[str, Any]]:
    focused_prompt, focused_error, fix_guidance = focused_debugger.create_focused_prompt(
        buggy_code=candidate,
        error_report=error_report,
        iteration=iteration,
        specification=specification,
        testbench_code=harness_context or None,
        context_code=None,
        per_signal_mismatches=None,
    )
    metadata = {
        "approach": "focused",
        "component": "FocusedDebugger.create_focused_prompt",
        "task_id": task_id,
        "target_file": target_file,
        "iteration": iteration,
        "focused_error": focused_error,
        "guidance": fix_guidance,
        "error_reasoning": fix_guidance,
        "harness_context_chars": len(harness_context or ""),
    }
    content = f"""\
Focused error:
{focused_error}

Fix guidance:
{fix_guidance}

Focused implementation prompt from reused reflector component:
{focused_prompt}
"""
    (iteration_dir / "reflector.response.prompt.txt").write_text(
        focused_prompt, encoding="utf-8", errors="replace"
    )
    write_json(
        iteration_dir / "reflector.response.json",
        {
            "content": content,
            "metadata": metadata,
            "focused_error": focused_error,
            "fix_guidance": fix_guidance,
        },
    )
    return content, metadata


def record_reflector_iteration_result(
    focused_debugger: FocusedDebugger,
    iteration: int,
    error_report: str,
    metadata: Optional[Dict[str, Any]],
    passed: bool,
    code_before: str,
    code_after: str,
) -> None:
    if not metadata or metadata.get("approach") != "focused":
        return

    num_passing, num_total = focused_debugger.parse_test_results(error_report)
    test_results = f"{num_passing}/{num_total} tests passed"
    if passed:
        outcome = "solved"
    elif focused_debugger.fix_history:
        previous = focused_debugger.fix_history[-1]
        try:
            previous_passing = int(previous.test_results.split("/", 1)[0])
        except (ValueError, IndexError):
            previous_passing = num_passing
        if num_passing > previous_passing:
            outcome = "improved"
        elif num_passing < previous_passing:
            outcome = "regressed"
        else:
            outcome = "no_change"
    else:
        outcome = "no_change"

    guidance = str(metadata.get("guidance", ""))
    focused_error = str(metadata.get("focused_error", "Unknown error"))
    if code_before.strip() and code_after.strip():
        implementation_changes, fix_effectiveness = focused_debugger.extract_implementation_changes(
            code_before=code_before,
            code_after=code_after,
            guidance=guidance,
            focused_error=focused_error,
            new_error_report=error_report,
        )
    else:
        implementation_changes = "Code changes not tracked"
        fix_effectiveness = "Effectiveness not analyzed"
    focused_debugger.record_attempt(
        iteration=iteration,
        focused_error=focused_error,
        error_reasoning=str(metadata.get("error_reasoning", "")),
        guidance=guidance,
        implementation_changes=implementation_changes,
        fix_effectiveness=fix_effectiveness,
        test_results=test_results,
        outcome=outcome,
        test_passed=passed,
        mismatch_count=-1,
    )


def call_coordinator(
    fresh_start_coordinator: FreshStartCoordinator,
    iteration_dir: Path,
    iteration: int,
    passed: bool,
    candidate: str,
    error_report: str,
) -> Tuple[str, Optional[str]]:
    fresh_start_coordinator.add_iteration_result(
        iteration=iteration,
        test_passed=passed,
        error_report=error_report,
        rtl_code=candidate,
        mismatch_count=-1,
    )
    should_restart, reason = fresh_start_coordinator.should_fresh_start(iteration)
    content = (
        "FreshStartCoordinator decision:\n"
        f"- decision: {'fresh_start' if should_restart else 'continue'}\n"
        f"- reason: {reason}\n\n"
        "Next generator instruction:\n"
        + (
            "Start from the public specification again. Do not preserve the previous "
            "candidate structure, and do not carry forward local reflector repair steps. "
            "Use only the public specification, module interface, tool compatibility "
            "requirements, and the generic failure class described in the restart reason."
            if should_restart
            else "Revise the current candidate using the reflector's focused error and fix guidance."
        )
    )
    (iteration_dir / "coordinator.response.prompt.txt").write_text(
        "FreshStartCoordinator.should_fresh_start was called with the accumulated "
        "iteration history and the latest evaluator report.\n\n"
        f"Latest evaluator report:\n{error_report}",
        encoding="utf-8",
        errors="replace",
    )
    write_json(
        iteration_dir / "coordinator.response.json",
        {
            "content": content,
            "component": "FreshStartCoordinator.should_fresh_start",
            "decision": "fresh_start" if should_restart else "continue",
            "reason": reason,
            "restart_summary": fresh_start_coordinator.get_restart_summary(),
        },
    )
    return content, reason if should_restart else None
