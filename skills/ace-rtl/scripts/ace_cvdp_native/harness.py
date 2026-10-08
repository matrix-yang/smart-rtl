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

def parse_env_file(path: Path, issue_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        values[key] = map_container_path(value, issue_path)
    return values


def env_file_paths(service_env_file: Any, issue_path: Path) -> List[Path]:
    raw_paths: List[str]
    if service_env_file is None:
        raw_paths = []
    elif isinstance(service_env_file, str):
        raw_paths = [service_env_file]
    elif isinstance(service_env_file, list):
        raw_paths = [str(item) for item in service_env_file]
    else:
        raw_paths = []

    paths = []
    for raw in raw_paths:
        mapped = Path(map_container_path(str(raw), issue_path))
        if not mapped.is_absolute():
            mapped = issue_path / mapped
        paths.append(mapped)

    default_env = issue_path / "src" / ".env"
    if default_env.exists() and default_env not in paths:
        paths.append(default_env)
    return paths


def compose_env(service_env: Any, service_env_file: Any, issue_path: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(issue_path / "rundir")
    env["XDG_CACHE_HOME"] = str(issue_path / "rundir" / ".cache")
    env["PYTHONNOUSERSITE"] = "1"
    python_prefix = Path(sys.executable).resolve().parent.parent
    python_bin = str(python_prefix / "bin")
    python_lib = python_prefix / "lib"
    env["PYGPI_PYTHON_BIN"] = str(Path(sys.executable).resolve())
    env["PATH"] = os.pathsep.join([python_bin, env.get("PATH", "")]).rstrip(os.pathsep)
    if python_lib.exists():
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(python_lib), env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(issue_path / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    for env_path in env_file_paths(service_env_file, issue_path):
        env.update(parse_env_file(env_path, issue_path))

    if isinstance(service_env, dict):
        for key, value in service_env.items():
            env[str(key)] = map_container_path(str(value), issue_path)
    elif isinstance(service_env, list):
        for item in service_env:
            if not isinstance(item, str) or "=" not in item:
                continue
            key, value = item.split("=", 1)
            env[key] = map_container_path(value, issue_path)
    return env


def rewrite_native_harness_paths(issue_path: Path) -> List[str]:
    """Rewrite container paths and configured tool commands in copied native harness files."""
    changed: List[str] = []
    text_suffixes = {".py", ".cmd", ".tcl", ".sh", ".env"}
    for root in (issue_path / "src",):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in text_suffixes:
                continue
            try:
                original = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            mapped = rewrite_native_tool_commands(map_container_path(original, issue_path))
            if path.suffix == ".py":
                mapped = rewrite_yosys_stats_parser_compat(mapped)
            if mapped == original:
                continue
            try:
                path.write_text(mapped, encoding="utf-8")
                changed.append(str(path.relative_to(issue_path)))
            except OSError:
                continue
    return changed


def rewrite_yosys_stats_parser_compat(text: str) -> str:
    """Allow copied native harnesses to parse both legacy and compact Yosys stats."""
    marker = "def parse_yosys_log(log_path):"
    if marker not in text or "return stats, has_error" not in text:
        return text
    if "_ace_extract_compact_yosys_stats" in text:
        return text

    helper = """
def _ace_extract_compact_yosys_stats(log_path):
    stats = {}
    stat_names = {
        "cells": "Number of cells",
        "wires": "Number of wires",
        "wire bits": "Number of wire bits",
        "memories": "Number of memories",
        "memory bits": "Number of memory bits",
        "processes": "Number of processes",
    }
    try:
        with open(log_path, 'r') as file:
            for line in file:
                match = re.search(r'^\\s*(\\d+)\\s+(wire bits|memory bits|cells|wires|memories|processes)\\b', line)
                if match:
                    stats[stat_names[match.group(2)]] = int(match.group(1))
    except OSError:
        pass
    return stats

"""

    parse_start = text.find(marker)
    parse_end = text.find("\ndef ", parse_start + len(marker))
    if parse_end == -1:
        parse_end = len(text)

    function_text = text[parse_start:parse_end]
    return_match = re.search(r"(?m)^(?P<indent>[ \t]*)return stats, has_error\s*$", function_text)
    if not return_match:
        return text

    indent = return_match.group("indent")
    replacement = (
        f"{indent}for key, value in _ace_extract_compact_yosys_stats(log_path).items():\n"
        f"{indent}    stats.setdefault(key, value)\n\n"
        f"{indent}return stats, has_error"
    )
    function_text = (
        function_text[: return_match.start()]
        + replacement
        + function_text[return_match.end() :]
    )
    return text[:parse_start] + helper + function_text + text[parse_end:]


def rewrite_native_tool_commands(text: str) -> str:
    """Apply configured native Cadence command names to copied harness scripts."""
    replacements = {
        "xrun": os.environ.get("CVDP_NATIVE_XRUN_CMD", "").strip(),
        "imc": os.environ.get("CVDP_NATIVE_IMC_CMD", "").strip(),
    }
    rewritten = text
    for default_tool, configured in replacements.items():
        if not configured or tool_executable(configured) == default_tool:
            continue
        rewritten = re.sub(rf"\b{re.escape(default_tool)}\b", configured, rewritten)
    rewritten = inject_cadence_default_timescale(rewritten)
    return rewritten


def inject_cadence_default_timescale(text: str) -> str:
    """Set a default Cadence timescale for modules without a directive."""
    lines: List[str] = []
    changed = False
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        newline = line[len(body) :]
        if CADENCE_TIMESCALE_RE.search(body):
            lines.append(line)
            continue

        def add_timescale(match: re.Match[str]) -> str:
            nonlocal changed
            changed = True
            return f"{match.group(1)} {CADENCE_DEFAULT_TIMESCALE_ARGS}"

        updated = CADENCE_COMMAND_RE.sub(add_timescale, body)
        lines.append(updated + newline)
    return "".join(lines) if changed else text


def row_harness_text(row: Dict[str, Any]) -> str:
    def flatten(value: Any) -> Iterable[str]:
        if isinstance(value, dict):
            for nested in value.values():
                yield from flatten(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from flatten(nested)
        elif value is not None:
            yield str(value)

    harness = row.get("harness") or {}
    return "\n".join(flatten(harness))


def service_names_from_row(row: Dict[str, Any]) -> List[str]:
    harness = row.get("harness") or {}
    compose_text = harness.get("docker-compose.yml") if isinstance(harness, dict) else None
    if not isinstance(compose_text, str):
        return []
    try:
        data = yaml.safe_load(compose_text) or {}
    except yaml.YAMLError:
        return []
    services = data.get("services") or {}
    return [str(name) for name in services.keys()]


def tool_executable(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    return parts[0] if parts else command


def required_native_tools_for_row(row: Dict[str, Any]) -> List[str]:
    text = row_harness_text(row)
    service_names = service_names_from_row(row)
    required: List[str] = []
    needs_icarus = (
        any(name.lower() in {"icarus", "iverilog"} for name in service_names)
        or re.search(r"(?m)^\s*SIM\s*=\s*icarus\s*$", text) is not None
        or re.search(r"get_runner\(\s*[\"']icarus[\"']\s*\)", text) is not None
        or re.search(r"\b(?:iverilog|vvp)\b", text) is not None
    )
    needs_verilator = (
        any(name.lower() == "verilator" for name in service_names)
        or re.search(r"(?m)^\s*SIM\s*=\s*verilator\s*$", text) is not None
        or re.search(r"get_runner\(\s*[\"']verilator[\"']\s*\)", text) is not None
        or re.search(r"\bverilator\b", text) is not None
    )
    needs_xcelium = (
        any(name.lower() in {"xrun", "xcelium"} for name in service_names)
        or re.search(r"(?m)^\s*SIM\s*=\s*xcelium\s*$", text) is not None
        or re.search(r"get_runner\(\s*[\"']xcelium[\"']\s*\)", text) is not None
        or re.search(r"\bxrun\b", text) is not None
    )
    if needs_icarus:
        required.extend(["iverilog", "vvp"])
    if needs_verilator:
        required.append("verilator")
    if needs_xcelium:
        required.append(os.environ.get("CVDP_NATIVE_XRUN_CMD", "xrun"))
    if needs_xcelium or re.search(r"\bimc\b", text):
        required.append(os.environ.get("CVDP_NATIVE_IMC_CMD", "imc"))
    if re.search(r"\byosys\b", text):
        required.append("yosys")

    unique: List[str] = []
    for tool in required:
        if tool and tool not in unique:
            unique.append(tool)
    return unique


def row_needs_cadence_coverage(row: Dict[str, Any]) -> bool:
    text = row_harness_text(row)
    return re.search(r"\bxrun\b", text) is not None and re.search(r"\bimc\b", text) is not None


def native_tool_preflight(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    missing_by_row: Dict[str, List[str]] = {}
    required_by_row: Dict[str, List[str]] = {}
    for row in rows:
        required = required_native_tools_for_row(row)
        if not required:
            continue
        row_id = str(row.get("id", "<unknown>"))
        required_by_row[row_id] = required
        missing = [
            tool
            for tool in required
            if not shutil.which(tool_executable(tool))
        ]
        if missing:
            missing_by_row[row_id] = missing
    missing_tools = sorted({tool for tools in missing_by_row.values() for tool in tools})
    return {
        "ok": not missing_by_row,
        "required_by_row": required_by_row,
        "missing_by_row": missing_by_row,
        "missing_tools": missing_tools,
    }


def _eda_tool_root_candidates() -> List[Path]:
    candidates: List[Path] = []
    for env_name in (
        "XCELIUM_HOME",
        "VMANAGER_HOME",
        "INCISIVE_HOME",
        "CADENCE_HOME",
        "VERILATOR_HOME",
        "YOSYS_HOME",
        "IVERILOG_HOME",
    ):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        root = Path(raw)
        candidates.extend(
            [
                root / "bin",
                root / "tools.lnx86/bin",
                root / "tools.lnx86/bin/64bit",
                root / "tools.lnx86/inca/bin",
            ]
        )
    return candidates


def apply_eda_tool_setup(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if os.environ.get("ACE_RTL_DISABLE_EDA_SETUP"):
        return {"enabled": False, "reason": "ACE_RTL_DISABLE_EDA_SETUP is set"}

    needs_cadence_coverage = any(row_needs_cadence_coverage(row) for row in rows)
    if needs_cadence_coverage:
        os.environ.setdefault("CVDP_NATIVE_XRUN_CMD", "xrun")
        os.environ.setdefault("CVDP_NATIVE_IMC_CMD", "imc")
        if os.environ.get("XCELIUM_HOME"):
            os.environ.setdefault("MDV_XLM_HOME", os.environ["XCELIUM_HOME"])

    required = sorted(
        {
            tool_executable(tool)
            for row in rows
            for tool in required_native_tools_for_row(row)
        }
    )
    if not required:
        return {"enabled": True, "required_tools": [], "applied_path_entries": []}

    detected_before = {tool: shutil.which(tool) for tool in required}
    applied_paths = prepend_env_paths(_eda_tool_root_candidates())
    needs_cadence = any(Path(tool_executable(tool)).name in {"xrun", "irun", "imc"} for tool in required)
    license_setup = detect_cadence_license_environment() if needs_cadence else {"configured": False}
    detected_after = {tool: shutil.which(tool) for tool in required}
    return {
        "enabled": True,
        "required_tools": required,
        "detected_before": detected_before,
        "applied_path_entries": applied_paths,
        "license_setup": license_setup,
        "detected_after": detected_after,
        "missing_after": [tool for tool, found in detected_after.items() if not found],
    }


def install_native_harness_patch() -> None:
    """Patch CVDP Repository.obj_harness in this process only."""
    from src import repository as cvdp_repository

    if getattr(cvdp_repository.Repository.obj_harness, "_ace_native_harness_patch", False):
        return

    def native_obj_harness(self: Any, issue_path: str = "", logfile: str = "", uut: str = None):
        issue = Path(issue_path).resolve()
        docker_compose = issue / "docker-compose.yml"
        if not docker_compose.exists():
            return original_obj_harness(self, issue_path, logfile, uut)
        rewritten_files = rewrite_native_harness_paths(issue)

        with docker_compose.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        services = data.get("services") or {}
        results = []
        error_count = 0
        for service_name, service in services.items():
            print(f"Running native service: {service_name}")
            service_log = f'{logfile}{f"_{service_name}" if len(services) > 1 else ""}.txt'
            command = compose_command(service.get("command", ""), issue)
            if not command.strip():
                result = {
                    "result": 1,
                    "log": service_log,
                    "error_msg": "native harness service has no command",
                    "execution": 0.0,
                    "pid": None,
                }
                results.append(result)
                error_count += 1
                continue

            workdir_raw = service.get("working_dir") or "/code/rundir"
            workdir = Path(map_container_path(str(workdir_raw), issue))
            workdir.mkdir(parents=True, exist_ok=True)
            (issue / "rundir" / "harness" / ".cache").mkdir(parents=True, exist_ok=True)
            (issue / "rundir" / ".cache").mkdir(parents=True, exist_ok=True)

            env = compose_env(service.get("environment"), service.get("env_file"), issue)
            start = time.time()
            timeout = env_int(
                "CVDP_NATIVE_TIMEOUT",
                env_int("DOCKER_TIMEOUT", DEFAULT_EVALUATOR_TIMEOUT_SECONDS),
            )
            killed_pids: List[int] = []
            with open(service_log, "w", encoding="utf-8", errors="replace") as out:
                if rewritten_files:
                    out.write("$ ace-native path rewrite: " + ", ".join(rewritten_files) + "\n")
                out.write(f"$ {command}\n")
                out.flush()
                proc: Optional[subprocess.Popen[Any]] = None
                code = 1
                msg: Optional[str] = None
                pid: Optional[int] = None
                try:
                    proc = subprocess.Popen(
                        command,
                        shell=True,
                        cwd=str(workdir),
                        env=env,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    pid = proc.pid
                    code = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    code = 124
                    msg = f"native harness timeout after {timeout}s"
                    pid = proc.pid if proc else getattr(exc, "pid", None)
                    out.write(f"\n{msg}; terminating native service subprocesses\n")
                    out.flush()
                    kill_candidates: List[int] = []
                    if proc and proc.pid:
                        kill_candidates.extend(descendant_pids(proc.pid))
                        kill_candidates.append(proc.pid)
                    kill_candidates.extend(pids_matching_path(issue))
                    killed_pids.extend(terminate_pids(kill_candidates, grace_seconds=1.0))
                    if proc and proc.poll() is None:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            proc.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    # A timed-out shell can orphan pytest/vvp children. The CVDP
                    # service has completed from ACE-RTL's perspective, so no
                    # native harness process rooted in this issue should remain.
                    killed_pids.extend(terminate_pids(pids_matching_path(issue), grace_seconds=0.2))

            result = {
                "result": code,
                "log": service_log,
                "error_msg": msg,
                "execution": time.time() - start,
                "pid": pid,
                "terminated_pids": sorted(set(killed_pids)),
            }
            results.append(result)
            error_count += 0 if code == 0 else 1

        return results, error_count

    original_obj_harness = cvdp_repository.Repository.obj_harness
    native_obj_harness._ace_native_harness_patch = True  # type: ignore[attr-defined]
    cvdp_repository.Repository.obj_harness = native_obj_harness


def result_passed(result: Dict[str, Any]) -> bool:
    if result.get("errors", 1) != 0:
        return False
    return all(test.get("result", 1) == 0 for test in result.get("tests", []))
