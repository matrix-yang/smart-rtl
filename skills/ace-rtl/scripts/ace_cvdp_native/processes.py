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

class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def ensure_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"{label} must stay under workspace root: {resolved}")
    return resolved


def unique_run_dir(outputs_root: Path, cid: str, dataset_stem: str) -> Path:
    base = outputs_root / f"cvdp_{cid}_{dataset_stem}_{now_stamp()}"
    run_dir = base
    i = 1
    while run_dir.exists():
        i += 1
        run_dir = Path(f"{base}_{i}")
    run_dir.mkdir(parents=True)
    return run_dir


def load_cid_rows(dataset: Path, cid: str, ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    wanted = set(ids or [])
    rows: List[Dict[str, Any]] = []
    with dataset.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if cid not in row.get("categories", []):
                continue
            if wanted and row["id"] not in wanted:
                continue
            rows.append(row)
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def pid_is_alive(pid: int) -> bool:
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
        tail = stat.rsplit(")", 1)[1].strip().split()
        if tail and tail[0] == "Z":
            return False
    except OSError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def proc_table() -> Dict[int, int]:
    table: Dict[int, int] = {}
    for entry in Path("/proc").iterdir() if Path("/proc").exists() else []:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            # /proc/<pid>/stat has "pid (comm) state ppid ..." and comm may
            # contain spaces, so split after the final ")".
            tail = stat.rsplit(")", 1)[1].strip().split()
            table[int(entry.name)] = int(tail[1])
        except (OSError, ValueError, IndexError):
            continue
    return table


def descendant_pids(root_pid: int) -> List[int]:
    table = proc_table()
    children: Dict[int, List[int]] = {}
    for pid, ppid in table.items():
        children.setdefault(ppid, []).append(pid)
    found: List[int] = []
    stack = list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        found.append(pid)
        stack.extend(children.get(pid, []))
    return found


def proc_matches_path(pid: int, root: Path) -> bool:
    marker = str(root.resolve())
    proc = Path("/proc") / str(pid)
    try:
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
        if marker in cmdline:
            return True
    except OSError:
        pass

    try:
        cwd = Path(os.readlink(proc / "cwd")).resolve()
        root_resolved = root.resolve()
        return cwd == root_resolved or root_resolved in cwd.parents
    except OSError:
        return False


def filtered_victim_pids(pids: Iterable[int]) -> List[int]:
    own_pid = os.getpid()
    protected = {own_pid, os.getppid()}
    return sorted({pid for pid in pids if pid not in protected}, reverse=True)


def signal_pids(pids: Iterable[int], sig: signal.Signals = signal.SIGTERM) -> List[int]:
    victims = filtered_victim_pids(pids)
    for pid in victims:
        if not pid_is_alive(pid):
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    return victims


def pids_matching_path(root: Path) -> List[int]:
    if not Path("/proc").exists():
        return []
    pids: List[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if proc_matches_path(pid, root):
            pids.append(pid)
    return pids


def terminate_pids(pids: Iterable[int], *, grace_seconds: float = 1.0) -> List[int]:
    victims = filtered_victim_pids(pids)
    if not victims:
        return []

    for sig in (signal.SIGTERM, signal.SIGKILL):
        signal_pids(victims, sig)
        deadline = time.time() + grace_seconds
        while time.time() < deadline:
            if not any(pid_is_alive(pid) for pid in victims):
                break
            time.sleep(0.05)
    return victims


def terminate_processes_for_path(root: Path) -> List[int]:
    return terminate_pids(pids_matching_path(root))


def terminate_process_tree(pid: Optional[int]) -> List[int]:
    if not pid:
        return []
    return terminate_pids(descendant_pids(pid) + [pid])


def attempt_process_pids(process: multiprocessing.Process, attempt_root: Path) -> List[int]:
    pids: List[int] = []
    if process.pid:
        pids.extend(descendant_pids(process.pid))
        pids.append(process.pid)
    pids.extend(pids_matching_path(attempt_root))
    return filtered_victim_pids(pids)


def request_attempt_stop(process: multiprocessing.Process, attempt_root: Path) -> List[int]:
    return signal_pids(attempt_process_pids(process, attempt_root), signal.SIGTERM)


def cleanup_attempt_root(attempt_root: Path, dp_root: Path) -> Dict[str, Any]:
    result = {"attempt_dir": str(attempt_root), "attempt_dir_removed": False}
    try:
        resolved = attempt_root.resolve()
        dp_resolved = dp_root.resolve()
        if resolved != dp_resolved and dp_resolved in resolved.parents and resolved.exists():
            shutil.rmtree(resolved)
            result["attempt_dir_removed"] = True
    except OSError as exc:
        result["cleanup_error"] = str(exc)
    return result


def first_pass_claim_path(dp_root: Path) -> Path:
    return dp_root / "first_pass_claim.json"


def try_claim_first_pass(
    *,
    dp_root: Path,
    task_id: str,
    attempt_index: int,
    iteration: int,
    candidate_path: Path,
) -> bool:
    claim_path = first_pass_claim_path(dp_root)
    payload = {
        "id": task_id,
        "attempt": attempt_index,
        "iteration": iteration,
        "candidate_path": str(candidate_path),
        "claimed_at": now_stamp(),
        "pid": os.getpid(),
    }
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(claim_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return True


def read_first_pass_claim(dp_root: Path) -> Optional[Dict[str, Any]]:
    claim_path = first_pass_claim_path(dp_root)
    if not claim_path.exists():
        return None
    try:
        payload = json.loads(claim_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload.get("attempt"), int):
        return None
    return payload


def map_container_path(text: str, issue_path: Path) -> str:
    replacements = {
        "/code/rundir": issue_path / "rundir",
        "/code/verif": issue_path / "verif",
        "/code/docs": issue_path / "docs",
        "/code/rtl": issue_path / "rtl",
        "/code/src": issue_path / "src",
        "/rundir": issue_path / "rundir",
        "/src": issue_path / "src",
        "/code": issue_path,
    }
    pattern = re.compile("|".join(re.escape(key) for key in replacements))
    return pattern.sub(lambda match: str(replacements[match.group(0)]), text)


def compose_command(command: Any, issue_path: Path) -> str:
    if isinstance(command, list):
        mapped = [map_container_path(str(part), issue_path) for part in command]
        return " ".join(shlex.quote(part) for part in mapped)
    return map_container_path(str(command), issue_path)


def prepend_env_paths(paths: Iterable[Path]) -> List[str]:
    existing = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    preferred: List[str] = []
    for path in paths:
        text = str(path)
        if path.exists() and text not in preferred:
            preferred.append(text)
    if preferred:
        remainder = [entry for entry in existing if entry not in preferred]
        os.environ["PATH"] = os.pathsep.join(preferred + remainder)
    return preferred


def detect_cadence_license_environment() -> Dict[str, Any]:
    env_state = {
        name: bool(os.environ.get(name))
        for name in CADENCE_LICENSE_ENV_VARS
    }
    configured = any(env_state.values())
    return {
        "configured": configured,
        "env_present": env_state,
        "message": (
            "Cadence/license environment variables are present"
            if configured
            else "No common EDA license environment variables are set"
        ),
    }
