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

def git_commit_or_unknown(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def maybe_reexec_workspace_python(workspace: Path) -> None:
    if os.environ.get("ACE_RTL_NATIVE_REEXECED") == "1":
        return
    venv_python = workspace / ".cvdp-native" / "bin" / "python3"
    if not venv_python.exists():
        return
    try:
        current = Path(sys.executable).resolve()
        desired = venv_python.resolve()
    except OSError:
        return
    if current == desired:
        return
    env = os.environ.copy()
    env["ACE_RTL_NATIVE_REEXECED"] = "1"
    os.execve(str(desired), [str(desired), *sys.argv], env)


SAFE_DIAG_RE = re.compile(
    r"(syntax error|compile|compil|elaborat|iverilog|verilator|vvp|yosys|error:|"
    r"undefined|undeclared|no such file|failed to build|cannot find|invalid module|"
    r"\*[EF],|timescale directive missing)",
    re.IGNORECASE,
)
UNSAFE_DIAG_RE = re.compile(
    r"(expected|actual|mismatch|assert|traceback|test_|scoreboard|reference|golden)",
    re.IGNORECASE,
)
DIAG_NOISE_RE = re.compile(
    r"(triggered by setting any of text, encoding, errors or universal_newlines|"
    r"\berrors:\s*0,\s*warnings:\s*\d+\b|"
    r"^xmsim:\s+\*[EF],(?:ERRSEV|FATSEV)\b|"
    r"^Generating native compiled code:|"
    r"^Loading native compiled code:|"
    r"^ERROR\s+Xcelium:runner\.py:\d+\s+Simulation failed:\s*\d+|"
    r"^(?:INFO|ERROR)\s+\w+:runner\.py:\d+\s+Running command\b|"
    r"^Testbench (?:detected|failed with).*\berrors\b|"
    r"^=+ Test .*Error Conditions =+$|"
    r"^(?:SOME )?TESTS FAILED!.*\berrors?\b|"
    r"^Test failed with\s+\d+\s+errors?\b|"
    r"^\S+\.s?v:\d+.*\$fatal.*\berrors\b|"
    r"^Errors:\s+\d+\s*$)",
    re.IGNORECASE,
)


def safe_diagnostics(log_paths: Iterable[str], limit: int = 80) -> List[str]:
    lines: List[str] = []
    for raw in log_paths:
        if not raw:
            continue
        path = Path(raw)
        if not path.exists():
            continue
        try:
            for line in read_bounded_text(path, MAX_SERVICE_LOG_BYTES).splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.lower() == "elaborating the design hierarchy:":
                    continue
                if DIAG_NOISE_RE.search(stripped):
                    continue
                if SAFE_DIAG_RE.search(stripped) and not UNSAFE_DIAG_RE.search(stripped):
                    lines.append(stripped[:500])
                if len(lines) >= limit:
                    return lines
        except OSError:
            continue
    return lines


COMPILATION_FAILURE_RE = re.compile(
    r"(command\s+'\[\s*['\"]?iverilog|process\s+'iverilog'\s+terminated|"
    r"iverilog.*terminated with error|\.s?v:\d+:\s*(?:error|syntax error)|"
    r"(?:xrun|irun|ncvlog|xmvlog|xmelab|xmsim):\s+\*[EF],|"
    r"returned non-zero exit status\s+\d+)",
    re.IGNORECASE,
)
SIM_LOG_COMPILER_DIAG_RE = re.compile(
    r"(syntax error|malformed statement|elaboration failed|"
    r"\*[EF],|CUMSTS|timescale directive missing|Modules w/wo timescales|timescale of|"
    r"\berrors?\b|no such file or directory|cannot (?:open|find)|"
    r"unable to (?:open|find)|not found|failed with \d+\s+errors?)",
    re.IGNORECASE,
)
SETUP_DIAG_RE = re.compile(
    r"(ModuleNotFoundError|ImportError while importing test module|"
    r"No module named|command not found|"
    r"Failed to initialize (?:IMC|coverage|database|license|simulator|tool|environment|vmanager)|"
    r"serialization exception|Valid data file not found|No Coverage Library was set|MDV_XLM_HOME|"
    r"missing environment variable|KeyError: ['\"]?(?:VERILOG_SOURCES|TOPLEVEL|MODULE)|"
    r"System virtual memory limit exceeded|consider using 64bit mode|"
    r"pytest collection|Interrupted: \d+ errors? during collection)",
    re.IGNORECASE,
)


def log_text_indicates_compilation_failure(log_paths: Iterable[str]) -> bool:
    for raw in log_paths:
        if not raw:
            continue
        path = Path(raw)
        if not path.exists():
            continue
        try:
            text = read_bounded_text(path, MAX_SERVICE_LOG_BYTES)
        except OSError:
            continue
        if COMPILATION_FAILURE_RE.search(text):
            return True
    return False


def compiler_diagnostics_from_sim_logs(log_paths: Iterable[str], limit: int = 80) -> List[str]:
    roots: List[Path] = []
    source_logs: List[Path] = []
    for raw in log_paths:
        if not raw:
            continue
        path = Path(raw)
        if path.exists() and path.is_file() and path not in source_logs:
            source_logs.append(path)
        for root in [path.parent, path.parent.parent if path.parent else None]:
            if root and root.exists() and root not in roots:
                roots.append(root)

    lines: List[str] = []
    seen: set[str] = set()
    log_candidates: List[Path] = []
    for source in source_logs:
        if source not in log_candidates:
            log_candidates.append(source)
    for root in roots:
        try:
            for name in ("xrun.log", "irun.log", "sim.log", "build.log", "simulation.log"):
                for path in sorted(root.rglob(name)):
                    if path not in log_candidates:
                        log_candidates.append(path)
        except OSError:
            continue
    for sim_log in log_candidates[:80]:
        try:
            text = read_bounded_text(sim_log, MAX_SERVICE_LOG_BYTES)
        except OSError:
            continue
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if not SIM_LOG_COMPILER_DIAG_RE.search(stripped):
                continue
            if stripped.lower() == "elaborating the design hierarchy:":
                continue
            if DIAG_NOISE_RE.search(stripped):
                continue
            if UNSAFE_DIAG_RE.search(stripped):
                continue
            stripped = re.sub(r"/home/[^:\s]+/", ".../", stripped)
            if stripped in seen:
                continue
            seen.add(stripped)
            lines.append(stripped[:500])
            if len(lines) >= limit:
                return lines
    return lines


def setup_diagnostics(log_paths: Iterable[str], limit: int = 40) -> List[str]:
    lines: List[str] = []
    seen: set[str] = set()
    for raw in log_paths:
        if not raw:
            continue
        path = Path(raw)
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if (
            "*E,LOCALP" in text
            and "illegal localparam in list of parameters" in text
            and "/harness/" in text
            and "/src/" in text
        ):
            line = (
                "Cadence rejected immutable harness src/*.sv with LOCALP/IEEE-2001; "
                "use the compatible Xcelium/VMANAGER setup instead of feeding this "
                "back as a candidate edit."
            )
            if line not in seen:
                seen.add(line)
                lines.append(line)
                if len(lines) >= limit:
                    return lines
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped or not SETUP_DIAG_RE.search(stripped):
                continue
            stripped = re.sub(r"/home/[^:\s]+/", ".../", stripped)
            if stripped in seen:
                continue
            seen.add(stripped)
            lines.append(stripped[:500])
            if len(lines) >= limit:
                return lines
    return lines


def read_bounded_text(path: Path, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size <= max_bytes:
            return handle.read().decode("utf-8", errors="replace")
        marker = b"\n... [REPORT TRUNCATED] ...\n"
        if max_bytes <= len(marker):
            return handle.read(max_bytes).decode("utf-8", errors="replace")
        keep = max_bytes - len(marker)
        head = int(keep * 0.7)
        tail = keep - head
        data = handle.read(head) + marker
        if tail:
            handle.seek(max(size - tail, 0))
            data += handle.read(tail)
        return data.decode("utf-8", errors="replace")


def read_native_rundir_log_excerpt(log_path: Path, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    if max_bytes < 1024:
        return log_path.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
    return read_bounded_text(log_path, max(0, min(MAX_NATIVE_RUNDIR_LOG_BYTES, max_bytes)))


def rundir_candidates_from_service_log(log_path: Path, log_text: str) -> List[Path]:
    candidates: List[Path] = []
    if log_path.parent.name == "reports":
        case_id = log_path.stem.split("_", 1)[0]
        if case_id:
            candidates.append(log_path.parent.parent / "harness" / case_id / "rundir")

    for match in re.finditer(r"(/[^ \n\r\t]+/harness/[^/\s]+/rundir)", log_text):
        candidates.append(Path(match.group(1)))
    for match in re.finditer(r"(/[^ \n\r\t:]+/harness/[^/\s:]+)/(?:src|rtl)(?:/|[:\s])", log_text):
        candidates.append(Path(match.group(1)) / "rundir")

    unique: List[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved not in unique:
            unique.append(resolved)
    return unique


def coverage_gap_summary(rundir: Path) -> str:
    coverage_path = rundir / "coverage.log"
    env_path = rundir.parent / "src" / ".env"
    if not coverage_path.exists() or not env_path.exists():
        return ""
    try:
        env_text = env_path.read_text(encoding="utf-8", errors="replace")
        coverage_text = coverage_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    target_match = re.search(r"\bTARGET\s*=\s*([0-9]+(?:\.[0-9]+)?)", env_text)
    if not target_match:
        return ""
    target = float(target_match.group(1))

    gaps: List[float] = []
    below_target: List[str] = []
    for line in coverage_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.lower().startswith("name"):
            continue
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", stripped)
        if not match:
            continue
        pct = float(match.group(1))
        gap = max(0.0, target - pct)
        gaps.append(gap)
        if gap > 0 and len(below_target) < 8:
            below_target.append(f"- {stripped} (gap to TARGET {target:g}: {gap:.2f})")

    if not gaps:
        return ""
    details = "\n".join(below_target) if below_target else "- all reported scopes meet TARGET"
    return (
        "\n# Coverage progress summary\n"
        f"TARGET={target:g}; total positive coverage gap={sum(gaps):.2f}\n"
        f"{details}\n"
    )


def collect_native_rundir_diagnostics(rundirs: Iterable[Path], remaining_bytes: int) -> str:
    sections: List[str] = []
    seen: set[Path] = set()
    seen_logs: set[Path] = set()
    remaining = max(0, remaining_bytes)
    for rundir in rundirs:
        if remaining <= 0:
            break
        try:
            resolved = rundir.resolve()
        except OSError:
            resolved = rundir
        if resolved in seen or not resolved.exists() or not resolved.is_dir():
            continue
        seen.add(resolved)

        summary = coverage_gap_summary(resolved)
        if summary and remaining > 0:
            if len(summary.encode("utf-8", errors="replace")) > remaining:
                summary = summary.encode("utf-8", errors="replace")[:remaining].decode(
                    "utf-8", errors="replace"
                )
            sections.append(summary)
            remaining -= len(summary.encode("utf-8", errors="replace"))

        harness_root = resolved.parent if resolved.name == "rundir" else resolved
        for name in NATIVE_RUNDIR_LOG_NAMES:
            if remaining <= 0:
                break
            log_paths: List[Path] = []
            direct = resolved / name
            if direct.exists() and direct.is_file():
                log_paths.append(direct)
            try:
                for nested in sorted(harness_root.rglob(name)):
                    if nested.exists() and nested.is_file() and nested not in log_paths:
                        log_paths.append(nested)
            except OSError:
                pass
            for log_path in log_paths[:20]:
                if remaining <= 0:
                    break
                try:
                    log_resolved = log_path.resolve()
                except OSError:
                    log_resolved = log_path
                if log_resolved in seen_logs:
                    continue
                seen_logs.add(log_resolved)
                try:
                    text = read_native_rundir_log_excerpt(log_path, remaining).strip()
                except OSError:
                    continue
                if not text:
                    continue
                block = f"\n--- {name} ({log_path}) ---\n{text}\n"
                if len(block.encode("utf-8", errors="replace")) > remaining:
                    block = block.encode("utf-8", errors="replace")[:remaining].decode(
                        "utf-8", errors="replace"
                    )
                sections.append(block)
                remaining -= len(block.encode("utf-8", errors="replace"))

    if not sections:
        return ""
    return (
        "\n\n================ NATIVE CVDP RUNDIR DIAGNOSTICS ================\n"
        "These bounded artifacts come from the active native harness rundir and "
        "are required for useful reflector feedback.\n"
        + "".join(sections)
        + "=================================================================\n"
    )


def make_preflight_result(issues: List[str], actions: List[str]) -> Dict[str, Any]:
    return {
        "category": "LOCAL_PREFLIGHT",
        "difficulty": None,
        "errors": 1,
        "preflight_failures": issues,
        "postprocess_actions": actions,
        "tests": [
            {
                "result": 1,
                "error_msg": "candidate rejected by local ACE-RTL preflight before private evaluator",
                "execution": 0.0,
                "log": None,
                "pid": None,
            }
        ],
    }


def summarize_result(result: Dict[str, Any]) -> str:
    tests = result.get("tests", [])
    log_paths = [test.get("log") for test in tests if isinstance(test, dict)]
    diag: List[str] = []
    if log_text_indicates_compilation_failure(log_paths):
        compiler_diag = compiler_diagnostics_from_sim_logs(log_paths)
        fallback_diag = safe_diagnostics(log_paths)
        diag = compiler_diag + [x for x in fallback_diag if x not in compiler_diag]
    setup_diag = setup_diagnostics(log_paths)
    payload = {
        "category": result.get("category"),
        "difficulty": result.get("difficulty"),
        "errors": result.get("errors"),
        "preflight_failures": result.get("preflight_failures") or [],
        "postprocess_actions": result.get("postprocess_actions") or [],
        "service_results": [
            {
                "result": test.get("result"),
                "error_msg": test.get("error_msg"),
                "execution": round(float(test.get("execution", 0.0)), 3),
                "log_basename": Path(test["log"]).name if test.get("log") else None,
            }
            for test in tests
            if isinstance(test, dict)
        ],
        "safe_compile_diagnostics": diag,
        "setup_diagnostics": setup_diag,
    }
    return json.dumps(payload, indent=2)


def primary_failure_from_summary(result: Dict[str, Any], summary: str) -> Dict[str, Any]:
    try:
        payload = json.loads(summary)
    except json.JSONDecodeError:
        payload = {}

    preflight = payload.get("preflight_failures") or []
    diagnostics = payload.get("safe_compile_diagnostics") or []
    setup = payload.get("setup_diagnostics") or []
    if preflight:
        message = str(preflight[0])
        stage = "LOCAL_PREFLIGHT"
    elif setup:
        message = str(setup[0])
        stage = "INFRA_SETUP"
    elif diagnostics:
        message = str(diagnostics[0])
        stage = "CANDIDATE_COMPILE"
    elif result.get("errors", 0):
        message = "candidate compiled and ran but failed one or more CVDP checks"
        stage = "CANDIDATE_FUNCTIONAL"
    else:
        message = "no failure detected"
        stage = "PASS"

    signature = re.sub(r":\d+:", ":N:", message)
    signature = re.sub(r"\s+", " ", signature).strip()[:240]
    return {
        "stage": stage,
        "message": message,
        "signature": signature,
    }


def primary_failure_from_attempt_error(message: str) -> Dict[str, Any]:
    text = str(message or "").strip() or "attempt failed before evaluation"
    low = text.lower()
    if "llm call failed" in low or "failed to get chat response" in low:
        stage = "INFRA_SETUP"
        kind = "llm_role_call"
    elif "generator extraction failed" in low:
        stage = "LOCAL_PREFLIGHT"
        kind = "generator_extraction"
    elif "process exited without reporting" in low or "winning attempt exited" in low:
        stage = "INFRA_SETUP"
        kind = "attempt_process"
    else:
        stage = "INFRA_SETUP"
        kind = "attempt_exception"
    signature = re.sub(r":\d+:", ":N:", text)
    signature = re.sub(r"\s+", " ", signature).strip()[:240]
    return {
        "stage": stage,
        "kind": kind,
        "message": text[:1000],
        "signature": signature,
    }


def load_response_content(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("content", "")).strip()
