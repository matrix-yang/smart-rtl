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


class CVDPTestingMixin:
    def _strip_timescale_directives_for_native_verif(self, code: str) -> str:
        """
        Native Xcelium rejects mixed timescale state when generated SV
        testbenches include `timescale but immutable Verilog DUT files do not.
        Let the harness use its default time unit instead of spending iterations
        on simulator elaboration fatals.
        """
        if not self._is_commercial_verification_harness():
            return code
        return re.sub(r"(?m)^[ \t]*`timescale[^\n]*(?:\n|$)", "", code)

    def _strip_markdown_fence_lines(self, code: str) -> str:
        """Remove stray LLM markdown fences that occasionally survive extraction."""
        return re.sub(r"(?m)^[ \t]*```[A-Za-z0-9_-]*[ \t]*(?:\n|$)", "", code)

    def _expected_verif_toplevel_name(self) -> Optional[str]:
        """
        Return the expected generated verifier module name when the harness
        explicitly makes the generated verif file the simulation top.

        Assertion-module tasks often set TOPLEVEL to the verif file stem.
        Coverage/checker tasks usually set TOPLEVEL to the DUT, so this intentionally
        requires the generated file stem to match TOPLEVEL before normalizing.
        """
        harness_dir = getattr(self, "harness_dir", None)
        generated_dir = getattr(self, "generated_code_dir", None)
        if not harness_dir or not generated_dir:
            return None

        try:
            rel_generated = generated_dir.resolve().relative_to(harness_dir.resolve())
        except ValueError:
            return None
        if not str(rel_generated).startswith("verif"):
            return None

        toplevel = self._read_harness_env().get("TOPLEVEL", "")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_$]*$", toplevel):
            return None

        candidate_names = []
        selected_file = getattr(self, "generated_code_file", None)
        if selected_file:
            candidate_names.append(Path(selected_file).stem)
        expected_filename = getattr(self, "expected_rtl_filename", None)
        if expected_filename:
            candidate_names.append(Path(expected_filename).stem)
        files = self._list_code_files(generated_dir)
        if len(files) == 1:
            candidate_names.append(files[0].stem)

        if any(name == toplevel for name in candidate_names):
            return toplevel
        return None

    def _normalize_verif_toplevel_module_name(self, code: str) -> str:
        expected = self._expected_verif_toplevel_name()
        if not expected:
            return code

        module_match = re.search(r"(?m)^(\s*module\s+)([A-Za-z_][A-Za-z0-9_$]*)\b", code)
        if not module_match:
            return code

        actual = module_match.group(2)
        if actual == expected:
            return code

        normalized = (
            code[: module_match.start(2)]
            + expected
            + code[module_match.end(2) :]
        )
        normalized = re.sub(
            rf"(?m)^(\s*endmodule\s*:\s*){re.escape(actual)}\b",
            rf"\1{expected}",
            normalized,
            count=1,
        )
        self.logger.info(
            f"Normalized generated verif module name from {actual} to harness TOPLEVEL {expected}"
        )
        return normalized

    def _prepare_generated_code_for_current_task(self, code: str) -> str:
        sanitized = self._strip_markdown_fence_lines(code)
        removed_fence = sanitized != code
        before_target_markers = sanitized
        preserve_target_markers = self._is_plain_rtl_generation_task()
        if not preserve_target_markers:
            sanitized = self._strip_generated_target_markers(sanitized)
        removed_target_markers = sanitized != before_target_markers
        before_timescale = sanitized
        sanitized = self._strip_timescale_directives_for_native_verif(sanitized)
        removed_timescale = sanitized != before_timescale
        before_top_normalize = sanitized
        sanitized = self._normalize_verif_toplevel_module_name(sanitized)
        normalized_toplevel = sanitized != before_top_normalize
        if not self._is_coverage_testbench_task():
            notes = []
            if removed_fence:
                notes.append("stray markdown fences")
            if removed_target_markers:
                notes.append("agent target-file markers")
            if removed_timescale:
                notes.append("`timescale directive")
            if normalized_toplevel:
                notes.append("verif module name")
            if notes:
                self.logger.info(f"Sanitized generated code before CVDP retest: {', '.join(notes)}")
            return sanitized

        if sanitized != code:
            message = "Sanitized generated code before CVDP retest"
            if removed_fence:
                message += ", removed stray markdown fences"
            if removed_target_markers:
                message += ", removed agent target-file markers"
            if removed_timescale:
                message += ", removed `timescale directive"
            if normalized_toplevel:
                message += ", normalized verif module name"
            self.logger.info(message)
        return sanitized

    def _cleanup_native_coverage_artifacts_before_test(self) -> None:
        """Remove stale coverage outputs that can poison iterative coverage reruns."""
        harness_dir = getattr(self, "harness_dir", None)
        if not harness_dir:
            return
        rundir = Path(harness_dir) / "rundir"
        if not rundir.exists():
            return
        for name in (
            "cov_work",
            "coverage_html",
            "coverage.log",
            "imc.log",
            "mdv.log",
        ):
            path = rundir / name
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            except OSError as e:
                self.logger.warning(f"Could not remove stale coverage artifact {path}: {e}")

    def _cleanup_native_sim_artifacts_before_test(self) -> None:
        """Remove stale Xcelium artifacts that can leak BUG_i defines."""
        harness_dir = getattr(self, "harness_dir", None)
        if not harness_dir:
            return
        rundir = Path(harness_dir) / "rundir"
        if not rundir.exists():
            return
        for name in (
            "xcelium.d",
            "INCA_libs",
            ".simvision",
            "xrun.history",
            "xrun.key",
        ):
            path = rundir / name
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            except OSError as e:
                self.logger.warning(f"Could not remove stale simulator artifact {path}: {e}")

    def _cleanup_native_diagnostic_logs_before_test(self) -> None:
        """Remove stale native redirected logs before a new CVDP retest."""
        harness_dir = getattr(self, "harness_dir", None)
        if not harness_dir:
            return
        rundir = Path(harness_dir) / "rundir"
        if not rundir.exists():
            return
        for name in (
            "lint_results.log",
            "synth.log",
            "sim.log",
        ):
            path = rundir / name
            try:
                if path.exists():
                    path.unlink()
            except OSError as e:
                self.logger.warning(f"Could not remove stale native diagnostic log {path}: {e}")

    def _protected_rtl_injection_and_test(self, rtl_code: str, datapoint_name: str) -> bool:
        """
        Protected RTL injection that ensures the test actually runs against the injected code.

        This fixes the critical pipeline bug where CVDP test process overwrites injected RTL.
        """
        rtl_code = self._prepare_generated_code_for_current_task(rtl_code)
        rtl_code = self._lift_late_sv_procedural_declarations(rtl_code)
        rtl_code = self._widen_overflowing_sv_enums(rtl_code)
        rtl_code = self._cast_enum_ternary_assignments(rtl_code)
        self._last_contract_violation_report = None
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Protected RTL injection attempt {attempt + 1}/{max_attempts}")

                # Step 1: Inject the RTL
                self._inject_rtl_into_natural_structure(rtl_code)

                # Step 2: Verify injection before test
                # Use the same file selection logic as injection to ensure consistency
                rtl_file = self._verify_injected_generated_code(rtl_code)
                if not rtl_file:
                    self.logger.error(f"RTL injection verification failed on attempt {attempt + 1}")
                    continue

                self._sync_injected_generated_code_to_cvdp_files(datapoint_name)

                contract_report = self._generated_code_contract_violation_report()
                if contract_report:
                    self.logger.warning(contract_report.splitlines()[0])
                    self._last_contract_violation_report = contract_report
                    self._last_test_result = {"tests": [{"result": 1}], "contract_guard": True}
                    return True

                semantic_report = self._generated_code_sv_semantic_guard_report(rtl_code)
                if semantic_report:
                    self.logger.warning(semantic_report.splitlines()[0])
                    self._last_contract_violation_report = semantic_report
                    self._last_test_result = {"tests": [{"result": 1}], "semantic_guard": True}
                    return True

                # Step 3: Run test through CVDP using the updated file cache.
                self.logger.info("Running CVDP test with updated generated-code cache")
                if self.native_harness_enabled():
                    self._cleanup_native_diagnostic_logs_before_test()
                if self._is_coverage_testbench_task():
                    self._cleanup_native_coverage_artifacts_before_test()
                if self._is_bug_detection_testbench_task():
                    self._cleanup_native_sim_artifacts_before_test()
                try:
                    result = self.wrapper.execute_single(datapoint_name)

                    # Verify RTL was preserved (should be same as what we injected)
                    if self._verify_injected_generated_code(rtl_code):
                        self.logger.info("SUCCESS: generated-code preservation successful after CVDP retest")
                    else:
                        self.logger.warning("Generated-code preservation could not be fully verified after CVDP retest")

                    self._last_test_result = result

                except Exception as test_error:
                    self.logger.error(f"CVDP test execution failed: {str(test_error)}")
                    continue

                # If we get here, the test ran successfully with our RTL
                # Result is already stored in _last_test_result by persistent execution
                return True

            except Exception as e:
                self.logger.error(f"Protected RTL injection attempt {attempt + 1} failed: {str(e)}")
                continue

        self.logger.error(f"All {max_attempts} protected RTL injection attempts failed")
        return False



    def _load_datapoint_data(self, datapoint_name: str) -> Optional[Dict]:
        """Load datapoint data from JSONL files."""
        # Look in workspace-local CVDP JSONL files; the caller's requested CID
        # determines which rows are relevant.
        jsonl_paths = sorted(self.cvdp_path.glob("cid*.jsonl"))
        jsonl_paths.extend(sorted(self.cvdp_path.glob("cvdp_*.jsonl")))

        for jsonl_path in jsonl_paths:
            if jsonl_path.suffix == ".jsonl" and jsonl_path.exists():
                try:
                    with open(jsonl_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                if data.get("id") == datapoint_name:
                                    self.logger.info(f"Found datapoint {datapoint_name} in {jsonl_path}")
                                    return data
                except Exception as e:
                    self.logger.warning(f"Error reading JSONL file {jsonl_path}: {str(e)}")

        # Try glob pattern for other JSONL files
        for jsonl_file in self.cvdp_path.glob("cid*.jsonl"):
            try:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if data.get("id") == datapoint_name:
                                self.logger.info(f"Found datapoint {datapoint_name} in {jsonl_file}")
                                return data
            except Exception as e:
                self.logger.warning(f"Error reading JSONL file {jsonl_file}: {str(e)}")

        return None

    def _find_jsonl_for_datapoint(self, datapoint_name: str) -> str:
        """Find the appropriate JSONL file for a given datapoint."""
        # Prefer the JSONL file supplied by the CLI batch flow.
        preferred = getattr(self, "preferred_jsonl_file", None) or os.environ.get("ACE_RTL_CVDP_JSONL")
        candidate_files: List[Path] = []
        if preferred:
            candidate_files.append(Path(preferred))

        # Then search CVDP-local and workspace-local dataset locations.
        candidate_files.extend(sorted(self.cvdp_path.glob("cid*.jsonl")))
        candidate_files.extend(sorted(self.cvdp_path.glob("cvdp_*.jsonl")))
        workspace = Path(os.environ.get("ACE_RTL_WORKSPACE", Path.cwd())).resolve()
        dataset_roots = [
            workspace / "datasets" / "cvdp-benchmark-dataset",
        ]
        for env_key in ("CVDP_DATASET_ROOT", "CVDP_DATASET_DIR"):
            env_value = os.environ.get(env_key)
            if env_value:
                env_path = Path(env_value).resolve()
                try:
                    env_path.relative_to(workspace)
                    dataset_roots.append(env_path)
                except ValueError:
                    self.logger.warning(f"Ignoring non-workspace {env_key}: {env_path}")
        for dataset_dir in dataset_roots:
            if dataset_dir.exists():
                candidate_files.extend(sorted(dataset_dir.glob("cvdp_*.jsonl")))

        seen = set()
        for jsonl_file in candidate_files:
            jsonl_file = jsonl_file.resolve()
            if jsonl_file in seen or not jsonl_file.exists():
                continue
            seen.add(jsonl_file)
            try:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if data.get("id") == datapoint_name:
                                self.logger.info(f"Found datapoint {datapoint_name} in {jsonl_file}")
                                return str(jsonl_file)
            except Exception as e:
                self.logger.warning(f"Error reading JSONL file {jsonl_file}: {str(e)}")

        raise FileNotFoundError(f"Datapoint {datapoint_name} not found in workspace-local CVDP JSONL files")

    def _determine_test_success(self, result: Dict, error_report: str = "") -> bool:
        """Determine if tests passed from CVDP result structure."""
        if not result:
            return False

        if 'tests' in result:
            tests = result['tests']
            if isinstance(tests, list):
                # All tests must have result == 0 to pass, and the generated
                # report must not contain simulator/pytest failure evidence.
                if (
                    self._is_bug_detection_testbench_task()
                    and all(test.get('result', 1) == 0 for test in tests)
                    and self._pytest_report_all_tests_passed(error_report)
                ):
                    return True
                return (
                    all(test.get('result', 1) == 0 for test in tests)
                    and not self._error_report_indicates_failure(error_report)
                )

        return False

    def _pytest_report_all_tests_passed(self, error_report: str) -> bool:
        """Detect pytest's final all-passed summary in a concatenated simulator log."""
        if not error_report:
            return False
        lower = str(error_report).lower()
        if not re.search(r"=+\s*\d+\s+passed\s+in\s+[\d.]+s\s*=+", lower):
            return False
        pytest_failure_patterns = [
            r"=+\s*failures\s*=+",
            r"=+\s*short test summary info\s*=+",
            r"\b\d+\s+failed(?:,|\s)",
            r"\bfailed,\s*\d+\s+passed\b",
            r"\berror\s+collecting\b",
        ]
        return not any(re.search(pattern, lower) for pattern in pytest_failure_patterns)

    def _error_report_indicates_failure(self, error_report: str) -> bool:
        """Return True when a CVDP text report shows failed tests despite a zero wrapper status."""
        if not error_report:
            return False

        text = str(error_report)
        lower = text.lower()

        if self._is_bug_detection_testbench_task() and self._pytest_report_all_tests_passed(text):
            return False

        # CVDP reports sometimes concatenate simulator output where an early
        # success banner is followed by one or more real failing tests. Treat
        # any explicit partial-pass summary as a failure.
        for passed, total in re.findall(r"\b(\d+)\s*/\s*(\d+)\s+tests\s+passed\b", lower):
            try:
                if int(passed) < int(total):
                    return True
            except ValueError:
                continue

        regex_failures = [
            r"\b[1-9]\d*\s+failed\b",
            r"\bfailed,\s*[1-9]\d*\s+passed\b",
            r"\btest\s+failed\s+with\s+[1-9]\d*\s+errors?\b",
            r"\b[1-9]\d*\s+errors?\s+found\b",
            r"\bTimeout for .+ expired\b",
            r"\btimeout expired\b",
            r"\btimed out\b",
        ]
        if any(re.search(pattern, lower) for pattern in regex_failures):
            return True

        failure_markers = [
            "assertionerror",
            "== failures ==",
            "= failures =",
            "systemexit: process 'xrun' terminated with error",
            "systemexit: process 'irun' terminated with error",
            "process 'xrun' terminated with error",
            "process 'irun' terminated with error",
            "$fatal",
            "fatal error",
            "test failed",
            "structured cvdp result failure",
            "nonzero cvdp test result",
            "nonzero cvdp status",
        ]
        return any(marker in lower for marker in failure_markers)

    def _determine_compilation_success(self, result: Dict) -> bool:
        """Determine if compilation succeeded from CVDP result structure."""
        if not result:
            return False

        # Check for compilation errors in the test results
        if 'errors' in result and result['errors']:
            error_str = str(result['errors']).lower()
            if any(keyword in error_str for keyword in ['compilation', 'syntax', 'parse error']):
                return False

        return True

    def cleanup_test_environment(self, datapoint_name: str):
        """Clean up any temporary test environments."""
        self.logger.info(f"Cleaning up test environment for {datapoint_name}")

        # Cleanup Docker resources for this datapoint
        self._cleanup_docker_resources(datapoint_name)

    def cleanup_global_resources(self):
        """Clean up global resources like Docker networks."""
        if self.native_harness_enabled():
            return
        try:
            # Only cleanup network if we created it
            if hasattr(self, '_created_networks'):
                for network_name in self._created_networks:
                    self.logger.info(f"Cleaning up Docker network: {network_name}")
                    try:
                        subprocess.run(
                            ["docker", "network", "rm", network_name],
                            capture_output=True,
                            text=True
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to remove network {network_name}: {str(e)}")
        except Exception as e:
            self.logger.warning(f"Error during global cleanup: {str(e)}")

    def _cleanup_docker_resources(self, datapoint_name: str):
        """Clean up Docker containers and resources for a specific datapoint."""
        if self.native_harness_enabled():
            return
        try:
            cleanup_commands = [
                # Stop any running containers for this datapoint
                ["docker", "ps", "-a", "--filter", f"name={datapoint_name}", "--format", "{{.ID}}"],
                # Remove stopped containers
                ["docker", "container", "prune", "-f"]
            ]

            for cmd in cleanup_commands:
                try:
                    subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                except Exception:
                    # Ignore cleanup errors
                    pass

        except Exception as e:
            self.logger.warning(f"Docker cleanup failed: {str(e)}")

    def _cleanup_unwanted_cvdp_files(self):
        """Clean up unwanted JSON files generated by CVDP."""
        if not hasattr(self, 'wrapper') or not self.wrapper:
            return

        # Get CVDP prefix where files are generated
        prefix_path = Path(self.wrapper.repo.prefix)

        # Files to remove
        unwanted_patterns = ["raw_result.json", "report.json"]

        # Check multiple directory levels where CVDP might create these files.
        levels_to_check = [prefix_path]
        for cid_dir in sorted(prefix_path.glob("cid*")):
            if cid_dir.is_dir():
                levels_to_check.append(cid_dir)
                levels_to_check.extend(path for path in cid_dir.iterdir() if path.is_dir())

        # Also check if base_path exists and is different
        if getattr(self, 'base_path', None):
            levels_to_check.append(Path(self.base_path))
            levels_to_check.append(Path(self.base_path).parent)

        for level_dir in levels_to_check:
            if level_dir.exists():
                for pattern in unwanted_patterns:
                    unwanted_file = level_dir / pattern
                    if unwanted_file.exists():
                        try:
                            unwanted_file.unlink()
                            self.logger.info(f"Removed unwanted file: {unwanted_file}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove {unwanted_file}: {str(e)}")

        # Also scan the entire prefix tree for any stray JSON files
        try:
            if prefix_path.exists():
                for json_file in prefix_path.rglob("raw_result.json"):
                    try:
                        json_file.unlink()
                        self.logger.info(f"Removed stray raw_result.json: {json_file}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove stray {json_file}: {str(e)}")

                for json_file in prefix_path.rglob("report.json"):
                    try:
                        json_file.unlink()
                        self.logger.info(f"Removed stray report.json: {json_file}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove stray {json_file}: {str(e)}")
        except Exception as e:
            self.logger.warning(f"Error during comprehensive JSON cleanup: {str(e)}")
