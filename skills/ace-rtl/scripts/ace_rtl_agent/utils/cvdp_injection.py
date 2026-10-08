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


class CVDPInjectionMixin:
    def _inject_rtl_into_natural_structure(self, rtl_code: str):
        """Inject corrected RTL into the natural CVDP structure."""
        code_dir = getattr(self, "generated_code_dir", None) or getattr(self, "rtl_dir", None)
        if not code_dir:
            self.logger.error("Generated code directory not found in natural CVDP structure")
            return

        self.generated_code_files = []
        self.generated_code_file_contents = {}
        if self._is_plain_rtl_generation_task() and self._inject_marked_target_sections(code_dir, rtl_code):
            return
        if self._inject_multifile_rtl_into_natural_structure(rtl_code):
            return

        rtl_code = self._strip_generated_target_markers(rtl_code)

        # Extract module name from the generated RTL code
        module_name = self._extract_module_name_from_rtl(rtl_code)
        declared_targets = self._declared_generated_code_targets(code_dir)

        # Find existing RTL files
        rtl_files = self._list_code_files(code_dir)

        # Prefer the file selected by the agent for multi-file CVDP verif
        # deliverables. Module-name matching can pick a helper when multiple
        # testbench files define related modules.
        rtl_file = None
        selected_file = getattr(self, "generated_code_file", None)
        if selected_file:
            selected_path = Path(selected_file)
            try:
                selected_path.resolve().relative_to(code_dir.resolve())
                rtl_file = selected_path
                rtl_file.parent.mkdir(parents=True, exist_ok=True)
                if selected_path.exists():
                    self.logger.info(f"Using selected generated-code file for injection: {rtl_file}")
                else:
                    self.logger.info(f"Creating selected generated-code file for injection: {rtl_file}")
            except ValueError:
                self.logger.warning(f"Selected generated-code file is outside target dir: {selected_path}")

        if self._is_plain_rtl_generation_task() and module_name and declared_targets:
            declared_match = self._declared_target_for_module(module_name, declared_targets)
            if declared_match:
                if rtl_file is not None:
                    try:
                        same_target = rtl_file.resolve() == declared_match.resolve()
                    except OSError:
                        same_target = rtl_file == declared_match
                    if not same_target:
                        self.logger.warning(
                            "Routing single-module RTL by actual module name "
                            f"`{module_name}`: {rtl_file.name} -> {declared_match.name}"
                        )
                rtl_file = declared_match
                rtl_file.parent.mkdir(parents=True, exist_ok=True)

        # Try to find a file that matches the module name
        if module_name:
            if rtl_file is None:
                for file_path in rtl_files:
                    if module_name.lower() in file_path.stem.lower():
                        rtl_file = file_path
                        self.logger.info(f"Found matching file for module '{module_name}': {rtl_file}")
                        break

        # Fallback to first file or create new file
        if not rtl_file:
            if rtl_files:
                rtl_file = rtl_files[0]
                self.logger.info(f"No module-specific file found, using first available: {rtl_file}")
            else:
                # Create with expected filename
                expected_filename = self._get_expected_rtl_filename(module_name)
                rtl_file = code_dir / expected_filename
                self.logger.info(f"Creating new RTL file: {rtl_file}")

        if (
            self._is_plain_rtl_generation_task()
            and module_name
            and declared_targets
            and rtl_file.name.lower() in {path.name.lower() for path in declared_targets}
            and rtl_file.stem.lower() != module_name.lower()
        ):
            exact_target = self._declared_target_for_module(module_name, declared_targets)
            if exact_target:
                self.logger.warning(
                    "Correcting generated RTL target mismatch before write: "
                    f"module `{module_name}` belongs in {exact_target.name}, not {rtl_file.name}"
                )
                rtl_file = exact_target

        try:
            with open(rtl_file, 'w') as f:
                f.write(rtl_code)
            self.expected_rtl_filename = rtl_file.name
            self.generated_code_file = rtl_file
            self.generated_code_files = [rtl_file]
            self.generated_code_file_contents = {rtl_file: rtl_code}
            self.logger.info(f"Injected corrected generated code into natural structure: {rtl_file}")

            # Verify injection succeeded by checking file content
            with open(rtl_file, 'r') as f:
                written_content = f.read()
            if len(written_content) != len(rtl_code):
                self.logger.warning(f"RTL injection size mismatch: wrote {len(rtl_code)} chars, found {len(written_content)} chars")
            elif written_content != rtl_code:
                self.logger.error("RTL injection content mismatch - file was modified after write!")
            else:
                self.logger.info(f"Generated-code injection verified: {len(rtl_code)} chars correctly written")

        except Exception as e:
            self.logger.error(f"Failed to inject RTL: {str(e)}")
            raise

    def _sync_generated_code_to_cvdp_files(self, datapoint_name: str, code_file: Path, rtl_code: str) -> bool:
        """
        Mirror the agent-edited file into CVDP's prepared file cache.

        CVDP's execute_single() calls prepare() on each run. prepare() restores
        files from self.wrapper.repo.files[id], so updating only the on-disk
        harness file is not enough: the next retest can overwrite it with the
        original LLM response. Keeping the cache in sync makes CVDP restore the
        candidate the agent just produced.
        """
        if not code_file:
            self.logger.warning("Cannot sync generated code: no code file recorded")
            return False

        harness_dir = getattr(self, "harness_dir", None)
        if not harness_dir:
            self.logger.warning("Cannot sync generated code: harness directory unknown")
            return False

        try:
            rel_path = code_file.resolve().relative_to(harness_dir.resolve()).as_posix()
        except ValueError:
            self.logger.warning(f"Cannot sync generated code outside harness dir: {code_file}")
            return False

        files_cache = getattr(getattr(self.wrapper, "repo", None), "files", None)
        if not isinstance(files_cache, dict):
            self.logger.warning("Cannot sync generated code: CVDP files cache unavailable")
            return False

        if datapoint_name not in files_cache or not isinstance(files_cache[datapoint_name], dict):
            files_cache[datapoint_name] = {}

        files_cache[datapoint_name][rel_path] = rtl_code
        self.logger.info(f"Synced generated code into CVDP files cache: {datapoint_name}:{rel_path}")
        return True

    def _sync_injected_generated_code_to_cvdp_files(self, datapoint_name: str) -> bool:
        """Sync all injected generated-code files into CVDP's prepared file cache."""
        contents = getattr(self, "generated_code_file_contents", None)
        if isinstance(contents, dict) and contents:
            ok = True
            for code_file, file_code in contents.items():
                ok = self._sync_generated_code_to_cvdp_files(datapoint_name, Path(code_file), file_code) and ok
            return ok

        code_file = getattr(self, "generated_code_file", None)
        if code_file:
            try:
                file_code = Path(code_file).read_text(encoding="utf-8", errors="replace")
            except OSError:
                file_code = ""
            return self._sync_generated_code_to_cvdp_files(datapoint_name, Path(code_file), file_code)
        return False

    def _verify_injected_generated_code(self, fallback_code: str) -> Optional[Path]:
        """Verify every injected generated-code file is still present and unchanged."""
        contents = getattr(self, "generated_code_file_contents", None)
        if isinstance(contents, dict) and contents:
            primary = getattr(self, "generated_code_file", None)
            for code_file, expected in contents.items():
                path = Path(code_file)
                if not path.exists():
                    self.logger.error(f"RTL injection failed - missing file {path}")
                    return None
                actual = path.read_text(encoding="utf-8", errors="replace")
                if actual != expected:
                    self.logger.error(f"RTL injection failed - content mismatch in file {path}")
                    return None
            self.logger.info(f"RTL injection verified across {len(contents)} generated file(s)")
            return Path(primary) if primary else next(iter(contents.keys()))

        rtl_file = getattr(self, "generated_code_file", None)
        if rtl_file and Path(rtl_file).exists():
            actual = Path(rtl_file).read_text(encoding="utf-8", errors="replace")
            if actual == fallback_code:
                self.logger.info(f"RTL injection verified: {len(fallback_code)} chars correctly written")
                return Path(rtl_file)
            self.logger.error(f"RTL injection failed - content mismatch in file {rtl_file}")
            return None
        return None

    def _extract_module_name_from_rtl(self, rtl_code: str) -> str:
        """Extract the module name from RTL code."""
        import re

        # Remove block comments first to avoid matching "module" in comments
        code_no_comments = re.sub(r'/\*.*?\*/', '', rtl_code, flags=re.DOTALL)

        # Remove line comments
        code_no_comments = re.sub(r'//.*', '', code_no_comments)

        # Look for module declaration pattern (first occurrence after removing comments)
        module_match = re.search(r'^\s*module\s+([a-zA-Z_][a-zA-Z0-9_]*)', code_no_comments, re.MULTILINE)
        if module_match:
            module_name = module_match.group(1)
            self.logger.debug(f"Extracted module name: {module_name}")
            return module_name

        self.logger.warning("Could not extract module name from RTL code")
        return None

    def _get_expected_rtl_filename(self, module_name: str = None) -> str:
        """Get the expected RTL filename based on module name or datapoint name."""
        if getattr(self, "expected_rtl_filename", None):
            return self.expected_rtl_filename
        if module_name:
            return f"{module_name}.sv"

        # Fallback to datapoint-based naming
        if hasattr(self, 'current_datapoint_name'):
            # Handle CVDP datapoint names with a two-token dataset prefix.
            base_name = self.current_datapoint_name.split('_')
            if len(base_name) >= 3:
                # Remove dataset prefix and numeric suffix.
                module_name = '_'.join(base_name[2:-1])  # e.g., "binary_to_BCD"
                return f"{module_name.lower()}.sv"

        # No fallback - raise error to catch configuration issues
        raise ValueError("Cannot determine RTL filename: no module name found and no valid datapoint name available")

    def _is_commercial_verification_harness(self) -> bool:
        """
        Detect CVDP commercial verification-generation harnesses.

        These tasks score generated files in verif/ while src/ and sometimes
        rtl/ hold immutable DUT/helper code. The .env scoring knobs are the most
        stable local signal: coverage tasks have TARGET, while bug-detection tasks use
        NUM_BUGS.
        """
        harness_dir = getattr(self, "harness_dir", None)
        if not harness_dir:
            return False

        env_path = Path(harness_dir) / "src" / ".env"
        if not env_path.exists():
            return False

        try:
            env_text = env_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

        return (
            re.search(r"\bTARGET\s*=", env_text) is not None
            or re.search(r"\bNUM_BUGS\s*=", env_text) is not None
        )

    def _read_harness_env(self) -> Dict[str, str]:
        """Read simple KEY=VALUE entries from the active CVDP harness .env."""
        harness_dir = getattr(self, "harness_dir", None)
        if not harness_dir:
            return {}

        env_path = Path(harness_dir) / "src" / ".env"
        if not env_path.exists():
            return {}

        env: Dict[str, str] = {}
        try:
            for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"').strip("'")
        except OSError:
            return {}
        return env

    def _is_coverage_testbench_task(self) -> bool:
        """Return True for generated testbench coverage tasks."""
        harness_dir = getattr(self, "harness_dir", None)
        generated_dir = getattr(self, "generated_code_dir", None)
        if not harness_dir or not generated_dir:
            return False

        try:
            rel_generated = generated_dir.resolve().relative_to(harness_dir.resolve())
        except ValueError:
            return False
        if not str(rel_generated).startswith("verif"):
            return False

        env_path = Path(harness_dir) / "src" / ".env"
        if not env_path.exists():
            return False
        env_text = env_path.read_text(encoding="utf-8", errors="replace")
        has_target = re.search(r"\bTARGET\s*=", env_text) is not None
        has_num_bugs = re.search(r"\bNUM_BUGS\s*=", env_text) is not None
        return has_target and not has_num_bugs

    def _is_bug_detection_testbench_task(self) -> bool:
        """Return True for checker/bug-detection testbench tasks."""
        harness_dir = getattr(self, "harness_dir", None)
        generated_dir = getattr(self, "generated_code_dir", None)
        if not harness_dir or not generated_dir:
            return False

        try:
            rel_generated = generated_dir.resolve().relative_to(harness_dir.resolve())
        except ValueError:
            return False
        if not str(rel_generated).startswith("verif"):
            return False

        env_path = Path(harness_dir) / "src" / ".env"
        if not env_path.exists():
            return False
        try:
            env_text = env_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return re.search(r"\bNUM_BUGS\s*=", env_text) is not None

    def _strip_sv_comments_for_guard(self, code: str) -> str:
        """Remove SV comments for no-cheat construct detection."""
        without_block = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        return re.sub(r"//.*", "", without_block)

    def _is_assertion_insertion_task(self) -> bool:
        """Return True for generated assertion tasks."""
        env = self._read_harness_env()
        env_text = " ".join(f"{k}={v}" for k, v in env.items()).lower()
        return "assertion" in env_text and "num_bugs" not in env_text

    def _is_plain_rtl_generation_task(self) -> bool:
        """Return True when the generated deliverable should be synthesizable RTL."""
        if self._is_assertion_insertion_task():
            return False

        harness_dir = getattr(self, "harness_dir", None)
        generated_dir = getattr(self, "generated_code_dir", None)
        if not harness_dir or not generated_dir:
            return False
        try:
            rel_generated = generated_dir.resolve().relative_to(harness_dir.resolve())
        except ValueError:
            return False
        return str(rel_generated).startswith("rtl")

    def _collect_no_cheat_violations(self, scan_text: str, checks) -> List[str]:
        violations: List[str] = []
        lines = scan_text.splitlines()
        for reason, pattern in checks:
            for match in pattern.finditer(scan_text):
                line_no = scan_text.count("\n", 0, match.start()) + 1
                line = lines[line_no - 1].strip() if 0 <= line_no - 1 < len(lines) else ""
                violations.append(f"line {line_no}: {reason}: {line[:180]}")
                if len(violations) >= 12:
                    break
            if len(violations) >= 12:
                break
        return violations

    def _format_no_cheat_report(self, task_name: str, rule_summary: str, violations: List[str], fix: str) -> Optional[str]:
        if not violations:
            return None
        return (
            f"CVDP_NO_CHEAT_GUARD: {task_name} {rule_summary}.\n"
            + "\n".join(f"- {item}" for item in violations)
            + f"\nFix: {fix}"
        )

    def _verification_no_cheat_violation_report(self, code: str) -> Optional[str]:
        """
        Reject verification-code candidates that interact with the evaluator,
        manipulate immutable DUT internals, or rely on external artifacts.
        """
        is_checker_task = self._is_bug_detection_testbench_task()
        is_assertion_task = self._is_assertion_insertion_task()
        if not is_checker_task and not is_assertion_task:
            return None

        scan_text = self._strip_sv_comments_for_guard(code or "")
        checks = [
            ("force/release is not legal generated verification behavior", re.compile(r"(?m)(?:^|;|\bbegin\b)\s*(?:force|release)\b")),
            (
                "shell/system calls are forbidden",
                re.compile(r"\$(?:system)\b|\bsystem\s*\(", re.IGNORECASE),
            ),
            (
                "file writes are forbidden in generated verification code",
                re.compile(r"\$(?:fopen|fwrite|fdisplay|fclose)\b", re.IGNORECASE),
            ),
            (
                "evaluator artifact references are forbidden",
                re.compile(
                    r"coverage\.log|cov_work|\bimc\b|\bxrun\b|\birun\b|raw_result|report\.json|pytest|subprocess|docker|/home/|/code\b|/src\b",
                    re.IGNORECASE,
                ),
            ),
        ]
        if is_checker_task:
            checks.insert(1, ("bind is not legal checker behavior", re.compile(r"(?m)^\s*bind\b")))

        task_name = "checker testbenches" if is_checker_task else "assertion code"
        fix = (
            "observe public design behavior with bounded stimulus/checks only; do not force DUT state, "
            "write evaluator artifacts, call shell tools, or depend on harness result files."
        )
        return self._format_no_cheat_report(
            task_name,
            "must not manipulate immutable DUT internals or evaluator artifacts",
            self._collect_no_cheat_violations(scan_text, checks),
            fix,
        )

    def _rtl_no_cheat_violation_report(self, code: str) -> Optional[str]:
        """Reject RTL candidates that include testbench-only or evaluator-control constructs."""
        if not self._is_plain_rtl_generation_task():
            return None

        scan_text = self._strip_sv_comments_for_guard(code or "")
        checks = [
            ("force/release/bind is not legal synthesizable RTL", re.compile(r"(?m)(?:^|;|\bbegin\b)\s*(?:force|release|bind)\b")),
            ("initial blocks are not legal for these RTL-generation tasks", re.compile(r"(?m)^\s*initial\b")),
            (
                "simulation delays are not legal synthesizable RTL",
                re.compile(
                    r"(?m)(?:^|;|\bbegin\b|\bfork\b)\s*#\s*(?:\d|\()"
                    r"|(?:<=|=)\s*#\s*(?:\d|\()"
                    r"|\bassign\s*#\s*(?:\d|\()",
                    re.IGNORECASE,
                ),
            ),
            (
                "simulation/testbench system tasks are forbidden in generated RTL",
                re.compile(
                    r"\$(?:display|write|monitor|strobe|error|fatal|finish|stop|random|urandom|fopen|fwrite|fdisplay|fclose)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "assertion/coverage constructs are not legal for these RTL-generation tasks",
                re.compile(r"\b(?:assert|assume|cover)\s+(?:property|\()", re.IGNORECASE),
            ),
            (
                "testbench/class/DPI constructs are forbidden in generated RTL",
                re.compile(r"\b(?:program|class|covergroup|import\s+\"DPI|export\s+\"DPI)\b", re.IGNORECASE),
            ),
            (
                "evaluator artifact references are forbidden",
                re.compile(
                    r"coverage\.log|cov_work|\bimc\b|\bxrun\b|\birun\b|raw_result|report\.json|pytest|subprocess|docker|/home/|/code\b|/src\b",
                    re.IGNORECASE,
                ),
            ),
        ]
        return self._format_no_cheat_report(
            "RTL-generation tasks",
            "must return only synthesizable target RTL and must not control the evaluator",
            self._collect_no_cheat_violations(scan_text, checks),
            "return the complete target RTL file only, using synthesizable logic that preserves the specified public interface and behavior.",
        )

    def _generated_code_no_cheat_violation_report(self, code: str) -> Optional[str]:
        """Apply no-cheat checks for the active CVDP task shape."""
        for check in (
            self._coverage_no_cheat_violation_report,
            self._verification_no_cheat_violation_report,
            self._rtl_no_cheat_violation_report,
        ):
            report = check(code)
            if report:
                return report
        return None

    def _coverage_no_cheat_violation_report(self, code: str) -> Optional[str]:
        """
        Reject coverage-stimulus candidates that manipulate the DUT or
        score path instead of exercising legal public inputs.
        """
        if not self._is_coverage_testbench_task():
            return None

        scan_text = self._strip_sv_comments_for_guard(code or "")
        checks = [
            ("force/release is not legal coverage stimulus", re.compile(r"(?m)(?:^|;|\bbegin\b)\s*(?:force|release)\b")),
            ("bind is not legal coverage stimulus", re.compile(r"(?m)^\s*bind\b")),
            (
                "coverage system tasks are forbidden",
                re.compile(r"\$(?:coverage_control|coverage_save|coverage_get|set_coverage_db_name)\b", re.IGNORECASE),
            ),
            (
                "shell/system calls are forbidden",
                re.compile(r"\$(?:system)\b|\bsystem\s*\(", re.IGNORECASE),
            ),
            (
                "file writes are forbidden in generated coverage testbenches",
                re.compile(r"\$(?:fopen|fwrite|fdisplay|fclose)\b", re.IGNORECASE),
            ),
            (
                "score/coverage artifact references are forbidden",
                re.compile(
                    r"coverage\.log|cov_work|\bimc\b|\bxrun\b|\birun\b|raw_result|report\.json|pytest|subprocess|docker|/home/|/code\b|/src\b",
                    re.IGNORECASE,
                ),
            ),
        ]

        return self._format_no_cheat_report(
            "coverage testbenches",
            "must only drive legal DUT inputs and must not manipulate DUT internals, coverage/report artifacts, or external tools",
            self._collect_no_cheat_violations(scan_text, checks),
            "remove the forbidden construct and close coverage using reset, public ports, legal protocol sequences, and bounded directed stimulus only.",
        )

    def _read_coverage_target(self) -> Optional[float]:
        harness_dir = getattr(self, "harness_dir", None)
        if not harness_dir:
            return None

        env_path = Path(harness_dir) / "src" / ".env"
        if not env_path.exists():
            return None

        try:
            env_text = env_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        match = re.search(r"\bTARGET\s*=\s*([0-9]+(?:\.[0-9]+)?)", env_text)
        if not match:
            return None
        return float(match.group(1))

    def _read_num_bugs(self) -> Optional[int]:
        harness_dir = getattr(self, "harness_dir", None)
        if not harness_dir:
            return None

        env_path = Path(harness_dir) / "src" / ".env"
        if not env_path.exists():
            return None

        try:
            env_text = env_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        match = re.search(r"\bNUM_BUGS\s*=\s*([0-9]+)", env_text)
        if not match:
            return None
        return int(match.group(1))

    def _extract_coverage_gap_centipct(self) -> Optional[int]:
        """
        Return a lower-is-better coverage gap in centipercent units.

        CVDP's coverage.log reports one or more scope rows. A task passes only
        when all relevant scopes clear TARGET, so summing positive gaps gives
        the existing coordinator a useful monotonic signal.
        """
        if not self._is_coverage_testbench_task():
            return None

        target = self._read_coverage_target()
        if target is None:
            return None

        coverage_path = Path(self.harness_dir) / "rundir" / "coverage.log"
        if not coverage_path.exists():
            return None

        try:
            coverage_text = coverage_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        gaps = []
        for line in coverage_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("-") or stripped.lower().startswith("name"):
                continue
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", stripped)
            if not match:
                continue
            pct = float(match.group(1))
            gaps.append(max(0.0, target - pct))

        if not gaps:
            return None
        return int(round(sum(gaps) * 100))

    def _extract_error_penalty(self, error_report: str) -> int:
        if not error_report:
            return 0

        compiler_errors = self._extract_compiler_error_count(error_report)
        if compiler_errors:
            # Compilation/structural failures should never look like useful
            # functional progress. Keep them worse than ordinary cocotb
            # assertion failures so the best-code tracker does not revert
            # toward uncompilable RTL.
            return 1_500_000 + compiler_errors * 10_000

        test_errors = 0
        for match in re.findall(r"TEST FAILED with\s+([0-9]+)\s+errors?", error_report, re.IGNORECASE):
            test_errors += int(match)

        if test_errors == 0:
            test_errors = len(re.findall(r"\$error\b|error:", error_report, re.IGNORECASE))

        return compiler_errors * 100000 + test_errors * 1000

    def _extract_compiler_error_count(self, error_report: str) -> int:
        if not error_report:
            return 0

        compiler_errors = len(re.findall(r"\bncvlog:\s+\*E,", error_report))
        compiler_errors += len(re.findall(r"\bxmvlog:\s+\*E,", error_report))
        compiler_errors += len(re.findall(r"\bxmelab:\s+\*E,", error_report))
        compiler_errors += len(re.findall(r"\birun:\s+\*E,", error_report))
        compiler_errors += len(re.findall(r"\bxrun:\s+\*E,", error_report))
        compiler_errors += len(re.findall(r"\bsyntax error\b", error_report, re.IGNORECASE))
        compiler_errors += len(re.findall(r"\bmalformed statement\b", error_report, re.IGNORECASE))
        compiler_errors += len(re.findall(r"\bProcess\s+'(?:iverilog|vvp|xrun|irun)'\s+terminated\s+with\s+error\b", error_report, re.IGNORECASE))
        compiler_errors += len(re.findall(r"\belaboration failed\b", error_report, re.IGNORECASE))
        return compiler_errors

    def _extract_bug_detection_details(self, error_report: str) -> Optional[Dict[str, Any]]:
        """
        Return lower-is-better checker/bug-detection progress details.

        The task passes only when the generated checker testbench is clean on the
        unmodified DUT and makes every injected BUG_i simulation exit nonzero.
        When BUG_i is missed, the pytest loop stops at the first miss, so the
        first missed bug is a useful monotonic estimate of progress.
        """
        if not self._is_bug_detection_testbench_task():
            return None

        num_bugs = self._read_num_bugs() or 1
        text = error_report or ""
        if self._pytest_report_all_tests_passed(text):
            return {
                "progress_metric": "checker_pytest_passed",
                "num_bugs": num_bugs,
                "mismatches": 0,
                "failure_stage": "passed",
            }
        lower = text.lower()
        compiler_errors = self._extract_compiler_error_count(text)

        details: Dict[str, Any] = {
            "progress_metric": "checker_clean_fail_or_undetected_bug_count",
            "num_bugs": num_bugs,
        }
        if compiler_errors:
            details["compiler_errors"] = compiler_errors
            details["mismatches"] = 1_000_000 + compiler_errors * 10_000
            details["failure_stage"] = "compile_or_elaboration"
            return details

        sanity_failed = (
            "test_sanity" in lower
            and (
                "simulation ended with error" in lower
                or "assertionerror" in lower
                or re.search(r"failed\s+.*test_sanity", lower) is not None
            )
        )
        if sanity_failed:
            test_errors = 0
            for match in re.findall(r"test failed with\s+([0-9]+)\s+errors?", text, re.IGNORECASE):
                test_errors += int(match)
            if test_errors == 0:
                test_errors = len(re.findall(r"\$(?:error|fatal)\b|test failed|some tests failed", text, re.IGNORECASE))
            details["mismatches"] = 500_000 + min(test_errors, 999) * 100
            details["failure_stage"] = "clean_dut_sanity"
            details["clean_sanity_failed"] = True
            if test_errors:
                details["clean_test_error_count"] = test_errors
            return details

        missed_ids = []
        for groups in re.findall(
            r"BUG_([0-9]+)\s+simulation ended without error|injected bug\s+([0-9]+)",
            text,
            flags=re.IGNORECASE,
        ):
            for group in groups:
                if group != "":
                    missed_ids.append(int(group))
        if missed_ids:
            first_missed = min(missed_ids)
            undetected_estimate = max(1, num_bugs - first_missed)
            details["mismatches"] = undetected_estimate
            details["failure_stage"] = "bug_not_detected"
            details["first_undetected_bug"] = first_missed
            details["detected_bugs_before_first_miss"] = max(0, min(first_missed, num_bugs))
            details["undetected_bug_count_estimate"] = undetected_estimate
            return details

        if "simulation ended without error" in lower or "did not detect injected bug" in lower:
            details["mismatches"] = num_bugs
            details["failure_stage"] = "bug_not_detected"
            details["undetected_bug_count_estimate"] = num_bugs
            return details

        # Unknown checker failure: keep it worse than a pure missed-bug case but
        # better than a compile error so the loop can still notice improvements.
        details["mismatches"] = 100_000
        details["failure_stage"] = "unknown_checker_failure"
        return details

    def _build_progress_details(self, result: Dict, error_report: str) -> Dict:
        details = dict(result) if isinstance(result, dict) else {}

        if details.get("mismatches", -1) != -1:
            return details

        bug_details = self._extract_bug_detection_details(error_report)
        if bug_details is not None:
            details.update(bug_details)
            return details

        coverage_gap = self._extract_coverage_gap_centipct()
        error_penalty = self._extract_error_penalty(error_report)
        if coverage_gap is None and error_penalty <= 0:
            return details

        metric = error_penalty + (coverage_gap or 0)
        details["mismatches"] = metric
        details["progress_metric"] = "coverage_error_penalty_plus_gap_centipct"
        if coverage_gap is not None:
            details["coverage_gap_centipct"] = coverage_gap
        if error_penalty:
            details["error_penalty"] = error_penalty
        return details
