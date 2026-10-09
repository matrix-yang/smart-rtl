#!/usr/bin/env python3
"""Three-role MAGE-style CVDP agent for an OpenAI-compatible model service."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any


CODE_ROOT = Path("/code")
RUNTIME_ROOT = Path("/tmp/mage")
API_BASE_URL = os.environ.get(
    "MAGE_API_BASE_URL", "http://192.168.16.17:30000/v1"
).rstrip("/")
API_KEY = os.environ.get("MAGE_API_KEY", "EMPTY")
MODEL = os.environ.get("MAGE_MODEL", "ChipMATE-V-9B")
API_TIMEOUT = int(os.environ.get("MAGE_API_TIMEOUT", "1800"))
API_RETRIES = int(os.environ.get("MAGE_API_RETRIES", "3"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAGE_MAX_OUTPUT_TOKENS", "8192"))
SNAPSHOT_CHAR_BUDGET = int(
    os.environ.get("MAGE_CONTEXT_CHAR_BUDGET", "48000")
)
TEMPERATURE = float(os.environ.get("MAGE_TEMPERATURE", "0.2"))
TOP_P = float(os.environ.get("MAGE_TOP_P", "0.95"))

RTL_SUFFIXES = {".sv", ".v", ".svh", ".vh"}
TEXT_SUFFIXES = RTL_SUFFIXES | {".md", ".txt"}
EXCLUDED_PARTS = {
    ".git",
    ".cache",
    "__pycache__",
    "harness",
    "rundir",
    "reference",
    "references",
}

WORKSPACE_RULES = """CVDP benchmark rules:
- Work only from the supplied public task, RTL, documentation, and verification context.
- Never modify prompt.json, rundir, benchmark infrastructure, hidden/reference files,
  documentation, or public verification files to force a pass.
- Implement only requested RTL deliverables under rtl/.
- Keep module names, ports, widths, reset polarity, timing, and latency exact.
- Use synthesizable SystemVerilog/Verilog unless the task explicitly says otherwise.
"""

FILE_PROTOCOL = """Return only zero or more complete file blocks in this exact form:
<<<FILE: rtl/relative_name.sv>>>
complete file content
<<<END_FILE>>>
Do not use Markdown fences. Do not emit patches or ellipses. If no change is needed,
return exactly <<<NO_CHANGES>>>.
"""


def emit(event: str, **payload: Any) -> None:
    record = {"event": event, "timestamp": time.time(), **payload}
    print("[MAGE_TELEMETRY] " + json.dumps(record, ensure_ascii=False), flush=True)


def _post_chat(messages: list[dict[str, str]], max_tokens: int, role: str) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
        # SGLang/Qwen-compatible switch. A harmless closing </think> marker may
        # remain in the response and is ignored by the file-block parser.
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking": {"type": "disabled"},
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    request = urllib.request.Request(
        API_BASE_URL + "/chat/completions", data=body, headers=headers, method="POST"
    )
    last_error: Exception | None = None
    for attempt in range(1, API_RETRIES + 1):
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
                raw = response.read().decode("utf-8")
            decoded = json.loads(raw)
            content = decoded["choices"][0]["message"].get("content") or ""
            usage = decoded.get("usage") or {}
            finish_reason = decoded["choices"][0].get("finish_reason")
            response_path = RUNTIME_ROOT / f"{role}.response.txt"
            response_path.write_text(content, encoding="utf-8")
            emit(
                "llm_call",
                role=role,
                model=decoded.get("model", MODEL),
                endpoint=API_BASE_URL,
                attempt=attempt,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                finish_reason=finish_reason,
                latency_seconds=round(time.monotonic() - started, 3),
                response_chars=len(content),
            )
            return content
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError,
                ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            detail = str(exc)
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail += ": " + exc.read().decode("utf-8", errors="replace")[:2000]
                except Exception:
                    pass
            emit(
                "llm_call_error",
                role=role,
                attempt=attempt,
                latency_seconds=round(time.monotonic() - started, 3),
                error=detail,
            )
            if attempt < API_RETRIES:
                time.sleep(min(10, 2 ** attempt))
    raise RuntimeError(f"{role} failed after {API_RETRIES} attempts: {last_error}")


def _priority(path: Path) -> tuple[int, str]:
    relative = path.relative_to(CODE_ROOT)
    first = relative.parts[0] if relative.parts else ""
    rank = {"rtl": 0, "verif": 1, "docs": 2}.get(first, 3)
    return rank, str(relative)


def _eligible(path: Path) -> bool:
    if not path.is_file() or path.name == "prompt.json":
        return False
    relative = path.relative_to(CODE_ROOT)
    if any(part in EXCLUDED_PARTS or part.startswith(".") for part in relative.parts):
        return False
    return path.suffix.lower() in TEXT_SUFFIXES


def _clip_file(text: str, allowance: int) -> tuple[str, bool]:
    if len(text) <= allowance:
        return text, False
    if allowance < 400:
        return text[:allowance], True
    head = allowance * 2 // 3
    tail = allowance - head
    marker = "\n/* ... public file truncated for model context ... */\n"
    return text[: max(0, head - len(marker))] + marker + text[-tail:], True


def workspace_snapshot() -> tuple[str, list[str]]:
    paths = sorted((path for path in CODE_ROOT.rglob("*") if _eligible(path)), key=_priority)
    if not paths:
        return "(no public text files)", []
    per_file_cap = max(2500, min(16000, SNAPSHOT_CHAR_BUDGET // len(paths) * 2))
    remaining = SNAPSHOT_CHAR_BUDGET
    sections: list[str] = []
    truncated: list[str] = []
    for path in paths:
        relative = path.relative_to(CODE_ROOT).as_posix()
        header = f"\n<public_file path=\"{relative}\">\n"
        footer = "\n</public_file>\n"
        if remaining <= len(header) + len(footer) + 80:
            truncated.append(relative)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        allowance = min(per_file_cap, remaining - len(header) - len(footer))
        clipped, was_truncated = _clip_file(text, allowance)
        if was_truncated:
            truncated.append(relative)
        section = header + clipped + footer
        sections.append(section)
        remaining -= len(section)
    manifest = "Public files: " + ", ".join(
        path.relative_to(CODE_ROOT).as_posix() for path in paths
    )
    result = manifest + "\n" + "".join(sections)
    return result[:SNAPSHOT_CHAR_BUDGET], truncated


FILE_BLOCK_RE = re.compile(
    r"<<<FILE:\s*([^>\r\n]+?)\s*>>>\s*\r?\n?(.*?)<<<END_FILE>>>",
    re.DOTALL,
)


def _safe_rtl_path(raw: str) -> Path | None:
    cleaned = raw.strip().replace("\\", "/")
    pure = PurePosixPath(cleaned)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        return None
    if pure.parts[0] != "rtl" or pure.suffix.lower() not in RTL_SUFFIXES:
        return None
    path = (CODE_ROOT / Path(*pure.parts)).resolve()
    try:
        path.relative_to(CODE_ROOT.resolve())
    except ValueError:
        return None
    return path


def _single_target_from_task(task: str) -> Path | None:
    candidates: list[Path] = []
    for match in re.findall(r"(?:/code/)?(rtl/[A-Za-z0-9_./$-]+\.(?:sv|v|svh|vh))", task):
        path = _safe_rtl_path(match)
        if path is not None and path not in candidates:
            candidates.append(path)
    empty_rtl = sorted(
        path for path in (CODE_ROOT / "rtl").glob("**/*")
        if path.is_file() and path.suffix.lower() in RTL_SUFFIXES and path.stat().st_size == 0
    ) if (CODE_ROOT / "rtl").is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    if len(empty_rtl) == 1:
        return empty_rtl[0].resolve()
    return None


def apply_file_blocks(response: str, task: str, role: str) -> list[str]:
    writes: list[str] = []
    rejected: list[str] = []
    for raw_path, content in FILE_BLOCK_RE.findall(response):
        target = _safe_rtl_path(raw_path)
        if target is None:
            rejected.append(raw_path.strip())
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        normalized = content.strip("\r\n") + "\n"
        target.write_text(normalized, encoding="utf-8")
        writes.append(target.relative_to(CODE_ROOT).as_posix())

    # Bounded fallback for a model that ignored the marker but returned one
    # fenced RTL file for an unambiguous target.
    if not writes:
        target = _single_target_from_task(task)
        fenced = re.search(
            r"```(?:systemverilog|verilog|sv)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE
        )
        if target is not None and fenced:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(fenced.group(1).strip("\r\n") + "\n", encoding="utf-8")
            writes.append(target.relative_to(CODE_ROOT).as_posix())

    emit("file_application", role=role, written=writes, rejected=rejected)
    return writes


def run_public_validation() -> str:
    rtl_files = sorted(
        path for path in (CODE_ROOT / "rtl").glob("**/*")
        if path.is_file() and path.suffix.lower() in RTL_SUFFIXES and path.stat().st_size
    ) if (CODE_ROOT / "rtl").is_dir() else []
    verif_files = sorted(
        path for path in (CODE_ROOT / "verif").glob("**/*")
        if path.is_file() and path.suffix.lower() in RTL_SUFFIXES and path.stat().st_size
    ) if (CODE_ROOT / "verif").is_dir() else []
    if not rtl_files:
        return "No non-empty RTL file exists after generation."

    output = RUNTIME_ROOT / "public_smoke.vvp"
    if verif_files:
        command = ["iverilog", "-g2012", "-o", str(output)] + [
            str(path) for path in rtl_files + verif_files
        ]
    else:
        command = ["iverilog", "-g2012", "-t", "null"] + [str(path) for path in rtl_files]
    records: list[str] = ["COMMAND: " + " ".join(command)]
    try:
        compiled = subprocess.run(
            command, cwd=CODE_ROOT, text=True, capture_output=True, timeout=120
        )
        records.extend([
            f"COMPILE_RETURN_CODE: {compiled.returncode}",
            "COMPILE_STDOUT:\n" + compiled.stdout,
            "COMPILE_STDERR:\n" + compiled.stderr,
        ])
        if compiled.returncode == 0 and verif_files and output.is_file():
            simulated = subprocess.run(
                ["vvp", "-n", str(output)],
                cwd=CODE_ROOT,
                text=True,
                capture_output=True,
                timeout=90,
            )
            records.extend([
                f"SIM_RETURN_CODE: {simulated.returncode}",
                "SIM_STDOUT:\n" + simulated.stdout,
                "SIM_STDERR:\n" + simulated.stderr,
            ])
    except subprocess.TimeoutExpired as exc:
        records.append(f"PUBLIC_VALIDATION_TIMEOUT: {exc}")
    result = "\n".join(records)
    emit("public_validation", output_chars=len(result), rtl_files=len(rtl_files), verif_files=len(verif_files))
    return result[-16000:]


def main() -> int:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    prompt_path = CODE_ROOT / "prompt.json"
    if not prompt_path.is_file():
        raise RuntimeError("/code/prompt.json is missing")
    task = json.loads(prompt_path.read_text(encoding="utf-8")).get("prompt", "").strip()
    if not task:
        raise RuntimeError("prompt.json contains no prompt")

    snapshot, truncated = workspace_snapshot()
    emit(
        "run_start",
        model=MODEL,
        endpoint=API_BASE_URL,
        snapshot_chars=len(snapshot),
        truncated_public_files=truncated,
    )

    plan = _post_chat(
        [
            {
                "role": "system",
                "content": "You are MAGE's testbench/interface planning agent and an expert RTL verifier.\n" + WORKSPACE_RULES,
            },
            {
                "role": "user",
                "content": (
                    "Derive the exact interface, cycle behavior, reset semantics, corner cases, and a concise validation plan. "
                    "Do not output RTL and do not modify files.\n\nTASK:\n" + task + "\n\nPUBLIC WORKSPACE:\n" + snapshot
                ),
            },
        ],
        min(2048, MAX_OUTPUT_TOKENS),
        "interface_planner",
    )
    (RUNTIME_ROOT / "interface-and-test-plan.md").write_text(plan, encoding="utf-8")

    generator_response = _post_chat(
        [
            {
                "role": "system",
                "content": "You are MAGE's RTL generator and an expert hardware designer.\n" + WORKSPACE_RULES,
            },
            {
                "role": "user",
                "content": (
                    "Implement the requested RTL completely. Respect existing interfaces and public verification.\n\n"
                    + FILE_PROTOCOL
                    + "\nTASK:\n" + task
                    + "\n\nINTERFACE/TEST PLAN:\n" + plan[-12000:]
                    + "\n\nPUBLIC WORKSPACE:\n" + snapshot
                ),
            },
        ],
        MAX_OUTPUT_TOKENS,
        "rtl_generator",
    )
    generator_writes = apply_file_blocks(generator_response, task, "rtl_generator")
    diagnostics = run_public_validation()

    reviewed_snapshot, review_truncated = workspace_snapshot()
    reviewer_response = _post_chat(
        [
            {
                "role": "system",
                "content": "You are MAGE's simulation reviewer and RTL editor.\n" + WORKSPACE_RULES,
            },
            {
                "role": "user",
                "content": (
                    "Review the generated RTL against the task, plan, current public workspace, and validation diagnostics. "
                    "Correct all identifiable issues. Return complete contents only for files that need changes.\n\n"
                    + FILE_PROTOCOL
                    + "\nTASK:\n" + task
                    + "\n\nINTERFACE/TEST PLAN:\n" + plan[-10000:]
                    + "\n\nPUBLIC VALIDATION:\n" + diagnostics
                    + "\n\nCURRENT PUBLIC WORKSPACE:\n" + reviewed_snapshot
                ),
            },
        ],
        MAX_OUTPUT_TOKENS,
        "simulation_reviewer",
    )
    reviewer_writes = apply_file_blocks(reviewer_response, task, "simulation_reviewer")
    final_diagnostics = run_public_validation()
    emit(
        "run_complete",
        model=MODEL,
        generator_writes=generator_writes,
        reviewer_writes=reviewer_writes,
        review_truncated_public_files=review_truncated,
        final_validation_excerpt=final_diagnostics[-2000:],
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit("fatal", error=f"{type(exc).__name__}: {exc}")
        print(f"[MAGE] fatal: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

