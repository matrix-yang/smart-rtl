# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CVDP integration implementation mixins."""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cvdp_integration_support import *


class CVDPReportsMixin:
    def _read_report_capped(self, report_path: Path) -> str:
        """
        Read a harness report file with a hard size cap.

        A hung testbench can produce a multi-hundred-MB report. Reading it fully
        would OOM the agent, fill the LLM context, and the file itself wastes
        disk. This reads only the head and tail, summarizes runaway repetition,
        and rewrites the on-disk file to the capped version so disk is reclaimed.
        """
        try:
            file_size = report_path.stat().st_size
        except OSError:
            file_size = 0

        # Small reports: read normally.
        if file_size <= MAX_REPORT_BYTES:
            with open(report_path, 'r', errors='replace') as f:
                return f.read()

        # Oversized report: read bounded head + tail only.
        self.logger.warning(
            f"Report {report_path} is oversized ({file_size / (1024*1024):.0f} MB) - "
            "likely a hung/spinning testbench. Capping and truncating on disk."
        )
        with open(report_path, 'r', errors='replace') as f:
            head_text = f.read(REPORT_HEAD_BYTES)
            try:
                f.seek(max(0, file_size - REPORT_TAIL_BYTES))
                tail_text = f.read()
            except OSError:
                tail_text = ""

        capped = _summarize_runaway_report(head_text, tail_text, file_size)

        # Rewrite the on-disk file to the capped version to reclaim disk space.
        try:
            with open(report_path, 'w', errors='replace') as f:
                f.write(capped)
        except OSError as e:
            self.logger.warning(f"Could not truncate oversized report on disk: {e}")

        return capped

    def _is_compilation_failure(self, report_text: str) -> bool:
        """
        Decide whether a failure is a COMPILATION failure (vs a functional one).

        A compilation failure is when iverilog itself returns non-zero (the design
        never elaborates/simulates). A functional failure is when the design
        compiled, the simulation ran, and a testbench assertion/check failed.

        We only want to surface the raw compiler log (sim.log) for compilation
        failures - functional failures must keep their original report behavior.
        """
        if not report_text:
            return False
        t = report_text.lower()

        # The pytest runner reports a failed iverilog invocation as a
        # CalledProcessError whose command array begins with 'iverilog'.
        if re.search(r"command\s+'\[\s*['\"]?iverilog", t):
            return True
        # The cocotb Icarus runner reports a compile failure as
        # "Process 'iverilog' terminated with error N" (error 2 = compile error).
        if re.search(r"process\s+'iverilog'\s+terminated", t):
            return True
        if "iverilog" in t and "terminated with error" in t:
            return True
        # Direct compiler error signatures (file:line: error / syntax error).
        if re.search(r"\.s?v:\d+:\s*(error|syntax error)", t):
            return True
        if "error: " in t and ("iverilog" in t or "elaborat" in t):
            return True
        return False

    def _collect_compiler_logs(self) -> str:
        """
        Collect the actual compiler error output (sim.log) from the harness.

        The cocotb runner redirects iverilog's stdout/stderr to sim.log, so the
        pytest report only shows a generic 'CalledProcessError ... non-zero exit
        status N' with NO compiler message. Without this, the debugging LLM
        cannot see real errors like "foo.sv:42: error: ... already been declared"
        and loops forever on a compilation failure it can't diagnose.

        NOTE: this is only invoked for compilation failures (see
        _is_compilation_failure), so we extract genuine compiler error lines.
        """
        if not hasattr(self, 'base_path') or not self.base_path:
            return ""

        snippets: List[str] = []
        seen: set = set()
        try:
            for sim_log in Path(self.base_path).rglob("sim.log"):
                try:
                    if sim_log.stat().st_size == 0:
                        continue
                    with open(sim_log, 'r', errors='replace') as f:
                        text = f.read(64 * 1024)  # compiler logs are small; cap anyway
                except OSError:
                    continue
                # Keep lines that are real compiler error diagnostics. Icarus
                # also emits warning-only lines with a file:line prefix (for
                # example "unique case ignored"); those are noisy for functional
                # cocotb failures and should not be labeled as compiler errors.
                diag_re = re.compile(
                    r"(syntax error|malformed statement|elaboration failed|"
                    r"\berrors?\b|no such file or directory|cannot (?:open|find)|"
                    r"unable to (?:open|find)|not found|failed with \d+\s+errors?)",
                    re.IGNORECASE,
                )
                diag_lines = [ln for ln in text.splitlines() if diag_re.search(ln)]
                if not diag_lines:
                    continue
                blob = "\n".join(diag_lines[:80])
                if blob in seen:
                    continue
                seen.add(blob)
                snippets.append(blob)
        except Exception as e:
            self.logger.warning(f"Failed collecting compiler logs: {e}")

        if not snippets:
            return ""

        return (
            "\n\n================ ACTUAL COMPILER OUTPUT (sim.log) ================\n"
            "These are the real iverilog compiler errors (the pytest report only\n"
            "shows a generic non-zero exit status). FIX the RTL to resolve these:\n\n"
            + "\n---\n".join(snippets)
            + "\n==================================================================\n"
        )

    def _read_native_rundir_log_excerpt(self, log_path: Path) -> str:
        """Read a bounded native harness log excerpt."""
        try:
            file_size = log_path.stat().st_size
        except OSError:
            return ""
        if file_size <= 0:
            return ""

        try:
            with open(log_path, "r", errors="replace") as f:
                if file_size <= MAX_NATIVE_RUNDIR_LOG_BYTES:
                    return f.read()
                head_bytes = MAX_NATIVE_RUNDIR_LOG_BYTES // 2
                tail_bytes = MAX_NATIVE_RUNDIR_LOG_BYTES - head_bytes
                head = f.read(head_bytes)
                try:
                    f.seek(max(0, file_size - tail_bytes))
                    tail = f.read()
                except OSError:
                    tail = ""
        except OSError:
            return ""

        return (
            head
            + f"\n... [{log_path.name} truncated from {file_size} bytes] ...\n"
            + tail
        )

    def _collect_native_rundir_logs(self) -> str:
        """
        Append Docker-free native harness logs that pytest redirects away.

        Optimization/lint tasks often fail with only a generic
        pytest assertion in the report, while Verilator/Yosys diagnostics live
        under harness/<id>/rundir/*.log. Without those diagnostics the repair
        loop guesses. Keep this restricted to the active native rundir and cap
        each file so commercial verification reports stay bounded.
        """
        harness_dir = getattr(self, "harness_dir", None)
        if not harness_dir:
            return ""

        rundir = Path(harness_dir) / "rundir"
        if not rundir.exists():
            return ""

        sections: List[str] = []
        for name in NATIVE_RUNDIR_LOG_NAMES:
            log_path = rundir / name
            if not log_path.exists() or not log_path.is_file():
                continue
            text = self._read_native_rundir_log_excerpt(log_path).strip()
            if not text:
                continue
            sections.append(
                f"\n--- {name} ({log_path}) ---\n{text}\n"
            )

        if not sections:
            return ""

        return (
            "\n\n================ NATIVE CVDP RUNDIR DIAGNOSTICS ================\n"
            "The pytest report may only show a generic non-zero status. These are\n"
            "the redirected Verilator/Yosys/simulation diagnostics from the active\n"
            "CVDP native harness rundir:\n"
            + "".join(sections)
            + "=================================================================\n"
        )

    def _maybe_append_compiler_logs(self, report_text: str) -> str:
        """Append raw harness logs when pytest hid the useful diagnostics."""
        appended = report_text
        if self._is_compilation_failure(report_text):
            appended += self._collect_compiler_logs()
        native_logs = self._collect_native_rundir_logs()
        if native_logs and native_logs not in appended:
            appended += native_logs
        return appended

    def _report_service_rank(self, report_path: Path) -> int:
        """Prefer the earliest failing CVDP service: lint, then sanity, then synth."""
        name = report_path.stem.lower()
        if "lint" in name:
            return 0
        if "sanity" in name or "test" in name or "sim" in name:
            return 1
        if "synth" in name or "yosys" in name:
            return 2
        return 3

    def _is_failing_report_text(self, report_text: str) -> bool:
        """Return True if a pytest/CVDP text report contains an actual failure."""
        if not report_text:
            return False

        failure_patterns = (
            r"(?m)^FAILED\b",
            r"(?m)^ERROR\b",
            r"\bFAILED\b",
            r"\bAssertionError\b",
            r"\bSystemExit\b",
            r"\bTraceback \(most recent call last\)",
            r"\bterminated with error\b",
            r"\breturned error\b",
            r"\breturn errors\b",
            r"\bOptimization failed\b",
            r"\bSynthesis failed\b",
            r"\bNo upgrades in synthesis\b",
            r"\bTimeout for .+ expired\b",
            r"\btimeout expired\b",
            r"\btimed out\b",
        )
        return any(re.search(pattern, report_text, flags=re.IGNORECASE) for pattern in failure_patterns)

    def _result_value_is_nonzero(self, value: Any) -> bool:
        """Return True for native CVDP result fields that encode a failing exit status."""
        if isinstance(value, bool) or value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            stripped = value.strip().lower()
            if not stripped:
                return False
            if stripped.isdigit():
                return int(stripped) != 0
            return stripped in {"fail", "failed", "error", "errored", "timeout", "timed out"}
        return False

    def _format_result_payload(self, payload: Any, limit: int = 4096) -> str:
        """Format a CVDP result fragment without letting it dominate the prompt."""
        try:
            text = json.dumps(payload, default=str, sort_keys=True)
        except Exception:
            text = str(payload)
        if len(text) > limit:
            return text[:limit] + "\n... [CVDP result fragment truncated] ..."
        return text

    def _result_failure_summary(self, result: Any) -> str:
        """
        Extract failure evidence from the structured CVDP result.

        Native CVDP can leave a benign-looking report file from pytest while the
        wrapper result still records a nonzero command/test status, especially
        for simulator timeouts. Surface that structured failure before falling
        back to the first report file so the repair loop sees the real signal.
        """
        if not result:
            return ""

        snippets: List[str] = []

        def add(label: str, payload: Any) -> None:
            if len(snippets) >= 12:
                return
            snippets.append(f"{label}:\n{self._format_result_payload(payload)}")

        def has_timeout_text(text: str) -> bool:
            return any(
                re.search(pattern, text, flags=re.IGNORECASE)
                for pattern in (
                    r"\bTimeout for .+ expired\b",
                    r"\btimeout expired\b",
                    r"\btimed out\b",
                )
            )

        def has_strong_failure_text(text: str) -> bool:
            return any(
                re.search(pattern, text, flags=re.IGNORECASE)
                for pattern in (
                    r"\bTraceback \(most recent call last\)",
                    r"\bAssertionError\b",
                    r"\bSystemExit\b",
                    r"\bterminated with error\b",
                    r"\breturned error\b",
                    r"\bfatal error\b",
                )
            ) or has_timeout_text(text)

        def walk(obj: Any, path: str, depth: int) -> None:
            if depth > 5 or len(snippets) >= 12:
                return

            if isinstance(obj, dict):
                errors = obj.get("errors")
                if errors:
                    add(f"{path}.errors", errors)

                for key in ("result", "returncode", "return_code", "exit_code", "exitcode", "rc"):
                    if key in obj and self._result_value_is_nonzero(obj.get(key)):
                        add(f"Nonzero CVDP status at {path}.{key}", obj)
                        break

                for key in ("status", "outcome"):
                    value = obj.get(key)
                    if isinstance(value, str) and self._result_value_is_nonzero(value):
                        add(f"Failing CVDP status at {path}.{key}", obj)
                        break

                tests = obj.get("tests")
                if isinstance(tests, list):
                    for index, test in enumerate(tests):
                        if not isinstance(test, dict):
                            continue
                        value = test.get("result")
                        if self._result_value_is_nonzero(value):
                            add(f"Nonzero CVDP test result at {path}.tests[{index}]", test)
                        status = test.get("status") or test.get("outcome")
                        if isinstance(status, str) and self._result_value_is_nonzero(status):
                            add(f"Failing CVDP test status at {path}.tests[{index}]", test)

                for key, value in obj.items():
                    if isinstance(value, str):
                        if has_strong_failure_text(value):
                            add(f"Failure text at {path}.{key}", value)
                    elif isinstance(value, (dict, list)):
                        walk(value, f"{path}.{key}", depth + 1)
                return

            if isinstance(obj, list):
                for index, value in enumerate(obj[:32]):
                    walk(value, f"{path}[{index}]", depth + 1)
                return

            if isinstance(obj, str) and has_strong_failure_text(obj):
                add(f"Failure text at {path}", obj)

        walk(result, "result", 0)
        if not snippets:
            return ""

        return (
            "================ STRUCTURED CVDP RESULT FAILURE ================\n"
            "The native CVDP wrapper reported a failing command/test result even if\n"
            "the selected pytest report below looks passing. Repair the generated\n"
            "code against this failure first.\n\n"
            + "\n\n".join(snippets)
            + "\n=================================================================\n"
        )

    def _ordered_report_files(self) -> List[Path]:
        """Order CVDP reports so current-issue reports are inspected first."""
        if not hasattr(self, 'reports_dir') or not self.reports_dir.exists():
            return []

        all_reports = sorted(self.reports_dir.glob("*.txt"), key=lambda f: f.name)
        issue_id = self._current_issue_id()

        def is_current_issue(report_path: Path) -> bool:
            if not issue_id:
                return False
            return report_path.stem == issue_id or report_path.stem.startswith(f"{issue_id}_")

        return sorted(
            all_reports,
            key=lambda p: (
                0 if is_current_issue(p) else 1,
                self._report_service_rank(p),
                p.name,
            ),
        )

    def _extract_error_report_from_natural_structure(self, result: Dict) -> str:
        """Extract error report from natural CVDP structure."""
        result_failure = self._result_failure_summary(result)

        # First try to find report files
        report_files = self._ordered_report_files()
        if report_files:
            report_contents = []
            for report_file in report_files:
                try:
                    content = self._read_report_capped(report_file)
                except Exception as e:
                    self.logger.warning(f"Failed to read report file {report_file}: {str(e)}")
                    continue

                report_contents.append((report_file, content))
                if self._is_failing_report_text(content):
                    content = self._maybe_append_compiler_logs(content)
                    self.logger.info(
                        f"Extracted failing error report from natural structure: {report_file} ({len(content)} chars)"
                    )
                    return content

            if report_contents:
                report_file, content = report_contents[0]
                content = self._maybe_append_compiler_logs(content)
                if result_failure:
                    content = result_failure + "\n\n================ FIRST CURRENT CVDP REPORT ================\n" + content
                    self.logger.info(
                        f"Extracted structured CVDP failure plus first current report: {report_file} ({len(content)} chars)"
                    )
                    return content
                self.logger.info(
                    f"No failing report detected; extracted first current report from natural structure: {report_file} ({len(content)} chars)"
                )
                return content

        # Fallback to result structure
        if result_failure:
            return self._maybe_append_compiler_logs(result_failure)
        if result and 'errors' in result and result['errors']:
            return self._maybe_append_compiler_logs(str(result['errors']))
        elif result and 'tests' in result:
            tests = result['tests']
            if isinstance(tests, list):
                error_info = []
                for test in tests:
                    if test.get('result', 0) != 0:
                        error_info.append(f"Test failed: {test}")
                if error_info:
                    return self._maybe_append_compiler_logs('\n'.join(error_info))

        return "No detailed error report available"
