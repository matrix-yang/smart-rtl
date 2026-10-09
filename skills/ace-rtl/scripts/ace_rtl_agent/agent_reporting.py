# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACE-RTL agent implementation mixins."""

import json
import multiprocessing
import os
import re
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .agent_types import AgentConfig, IterationResult


class AgentReportingMixin:
    def _generate_final_report(self, datapoint_name: str, total_time: float, iteration_results: List[IterationResult]) -> Dict[str, Any]:
        """Generate a comprehensive report of all iterations and final result."""
        if not iteration_results:
            return {
                "datapoint_name": datapoint_name,
                "success": False,
                "total_iterations": 0,
                "total_time": total_time,
                "error": "No iterations completed"
            }

        final_iteration = iteration_results[-1]
        successful_iterations = [r for r in iteration_results if r.test_passed]

        # Get systematic debugging summary if available
        debug_summary = self.rtl_generator.get_debugging_summary()

        # Extract debug strategies used across iterations
        debug_strategies = []
        for result in iteration_results:
            if (hasattr(result, 'test_results') and result.test_results and
                isinstance(result.test_results, dict) and 'debug_metadata' in result.test_results):
                metadata = result.test_results['debug_metadata']
                if metadata and 'strategy_used' in metadata:
                    debug_strategies.append({
                        'iteration': result.iteration,
                        'strategy': metadata['strategy_used'],
                        'passed': result.test_passed
                    })

        report = {
            "datapoint_name": datapoint_name,
            "success": final_iteration.test_passed,
            "total_iterations": len(iteration_results),
            "successful_iterations": len(successful_iterations),
            "total_time": total_time,
            "average_iteration_time": sum(r.execution_time for r in iteration_results) / len(iteration_results),
            "final_rtl_code": final_iteration.rtl_code,
            "final_generated_code": final_iteration.rtl_code,
            "final_error": final_iteration.error_message,
            "iteration_history": [asdict(r) for r in iteration_results],
            "config": asdict(self.config),
            "systematic_debugging": {
                "enabled": self.rtl_generator.use_systematic_debugging,
                "debug_strategies_used": debug_strategies,
                "debug_summary": self._safe_serialize_debug_summary(debug_summary) if debug_summary else None
            }
        }

        return report

    def _extract_assertion_details(self, error_report: str) -> Optional[str]:
        """Extract assertion error details from error report for display."""
        if not error_report:
            return None

        # Some simulator reports are very large. Avoid DOTALL regexes across
        # the whole report here; this is only a console summary and should never
        # be allowed to stall the experiment.
        chunk_size = 12000
        if len(error_report) > chunk_size * 2:
            search_text = error_report[:chunk_size] + "\n" + error_report[-chunk_size:]
        else:
            search_text = error_report

        for line in search_text.splitlines():
            normalized = re.sub(r'\s+', ' ', line.strip())
            if not normalized:
                continue
            lower = normalized.lower()
            has_assertion_context = "assert" in lower or "fail" in lower or "error" in lower
            has_comparison = (
                "expected" in lower and
                ("got" in lower or "actual" in lower or "received" in lower)
            )
            if has_assertion_context and has_comparison:
                return normalized[:500]

        for line in search_text.splitlines():
            normalized = re.sub(r'\s+', ' ', line.strip())
            lower = normalized.lower()
            if "expected" in lower and ("got" in lower or "actual" in lower or "received" in lower):
                return normalized[:500]

        return None

    def _save_iteration_result(self, output_dir: Path, result: IterationResult):
        """Save individual iteration result to file."""
        iteration_dir = output_dir / f"iteration_{result.iteration}"
        iteration_dir.mkdir(exist_ok=True)

        # Save RTL code
        if result.rtl_code:
            rtl_file = iteration_dir / "generated.sv"
            with open(rtl_file, 'w') as f:
                f.write(result.rtl_code)

        # Save iteration metadata
        metadata = asdict(result)
        metadata_file = iteration_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _save_iteration_result_to_cvdp_structure(self, result: IterationResult, cvdp_int=None):
        """Save iteration RTL script and error report to iterations/ subdirectory."""
        cvdp_integration = cvdp_int or self.cvdp_integration

        if not getattr(cvdp_integration, 'base_path', None):
            self.logger.warning("CVDP base_path not available, cannot save iteration result")
            return

        # Create iterations directory under CVDP base path
        iterations_dir = cvdp_integration.base_path / "iterations"
        iterations_dir.mkdir(exist_ok=True)

        # Create iteration-specific subdirectory
        iteration_dir = iterations_dir / f"iteration_{result.iteration}"
        iteration_dir.mkdir(exist_ok=True)

        # Save RTL code
        if result.rtl_code:
            # Use expected RTL filename from CVDP integration
            rtl_filename = getattr(cvdp_integration, 'expected_rtl_filename', 'generated.sv')
            rtl_file = iteration_dir / rtl_filename
            with open(rtl_file, 'w') as f:
                f.write(result.rtl_code)
            self.logger.info(f"Saved iteration {result.iteration} RTL to {rtl_file}")

        # Save error report (always save, even if empty) - use full test results, not just error_message
        if result.test_passed:
            # For successful tests, try to get full test output instead of just success message
            if hasattr(result, 'test_results') and result.test_results:
                if isinstance(result.test_results, dict) and 'error_report' in result.test_results:
                    error_content = result.test_results['error_report']
                elif isinstance(result.test_results, str):
                    error_content = result.test_results
                else:
                    error_content = str(result.test_results)
            else:
                error_content = "Test passed - no errors"
        else:
            # Use detailed test results if available, otherwise fall back to error_message
            if hasattr(result, 'test_results') and result.test_results:
                if isinstance(result.test_results, dict) and 'error_report' in result.test_results:
                    error_content = result.test_results['error_report']
                elif isinstance(result.test_results, str):
                    error_content = result.test_results
                else:
                    error_content = str(result.test_results)
            else:
                error_content = result.error_message if result.error_message else "Test failed - no detailed error report available"

            # If still generic, try to read from reports directory
            if error_content == "Test failed - see error_report for details" and hasattr(cvdp_integration, 'reports_dir'):
                try:
                    report_files = list(cvdp_integration.reports_dir.glob("*.txt"))
                    if report_files:
                        if hasattr(cvdp_integration, '_read_report_capped'):
                            full_report = cvdp_integration._read_report_capped(report_files[0])
                        else:
                            with open(report_files[0], 'r', errors='replace') as f:
                                full_report = f.read()
                        if full_report and len(full_report) > 100:  # Only use if substantial content
                            error_content = full_report
                            self.logger.info(f"Read full error report from {report_files[0]} ({len(full_report)} chars)")
                except Exception as e:
                    self.logger.warning(f"Failed to read full error report for iteration: {str(e)}")

        error_file = iteration_dir / "error_report.txt"
        with open(error_file, 'w') as f:
            # Handle case where error_content is None (test passed with no error report)
            if error_content is None:
                f.write("Test passed - no errors")
            else:
                f.write(error_content)
        content_len = len(error_content) if error_content else 0
        self.logger.info(f"Saved iteration {result.iteration} error report to {error_file} ({content_len} chars)")

        # Save prompt content
        if result.prompt_content:
            prompt_file = iteration_dir / "prompt.md"
            with open(prompt_file, 'w') as f:
                f.write(result.prompt_content)
            self.logger.info(f"Saved iteration {result.iteration} prompt to {prompt_file} ({len(result.prompt_content)} chars)")
        else:
            self.logger.warning(f"No prompt content available for iteration {result.iteration}")

        # Save response content
        if result.response_content:
            response_file = iteration_dir / "response.md"
            with open(response_file, 'w') as f:
                f.write(result.response_content)
            self.logger.info(f"Saved iteration {result.iteration} response to {response_file} ({len(result.response_content)} chars)")
        else:
            self.logger.debug(f"No response content available for iteration {result.iteration}")

        # Save debug metadata if available from systematic debugging
        if (hasattr(result, 'test_results') and result.test_results and
            isinstance(result.test_results, dict) and 'debug_metadata' in result.test_results):
            debug_metadata = result.test_results['debug_metadata']
            if debug_metadata:
                debug_file = iteration_dir / "debug_metadata.json"
                with open(debug_file, 'w') as f:
                    import json
                    try:
                        # Try safe JSON serialization first
                        serializable_metadata = self._make_json_serializable(debug_metadata)
                        json.dump(serializable_metadata, f, indent=2)
                        self.logger.info(f"Saved iteration {result.iteration} debug metadata to {debug_file}")
                    except Exception as e:
                        # Fallback: save just the essential information
                        self.logger.warning(f"Full debug metadata serialization failed: {e}")
                        safe_metadata = {
                            "strategy_used": debug_metadata.get("strategy_used", "unknown"),
                            "reasoning": debug_metadata.get("reasoning", "not available"),
                            "serialization_note": "Full metadata could not be serialized due to complexity"
                        }
                        json.dump(safe_metadata, f, indent=2)
                        self.logger.info(f"Saved simplified debug metadata to {debug_file}")

        # Save debugging summary for final iteration
        if result.test_passed:
            debug_summary = self.rtl_generator.get_debugging_summary()
            if debug_summary:
                summary_file = iteration_dir / "debugging_summary.json"
                with open(summary_file, 'w') as f:
                    import json
                    try:
                        # Convert any remaining dataclass objects to dicts for JSON serialization
                        serializable_summary = self._make_json_serializable(debug_summary)
                        json.dump(serializable_summary, f, indent=2)
                        self.logger.info(f"Saved debugging summary to {summary_file}")
                    except Exception as e:
                        # Fallback: save just the essential summary information
                        self.logger.warning(f"Full debugging summary serialization failed: {e}")
                        safe_summary = {
                            "total_iterations": debug_summary.get("total_iterations", 0),
                            "strategies_used": debug_summary.get("strategies_used", []),
                            "successful_iterations": debug_summary.get("successful_iterations", []),
                            "serialization_note": "Full summary could not be serialized due to complexity"
                        }
                        json.dump(safe_summary, f, indent=2)
                        self.logger.info(f"Saved simplified debugging summary to {summary_file}")

    def _save_initial_rtl_as_iteration_0(self, initial_rtl: str, initial_error_report: str = "", initial_prompt: str = "", cvdp_int=None):
        """Save initial RTL code and initial test report as iteration_0."""
        cvdp_integration = cvdp_int or self.cvdp_integration

        try:
            if not getattr(cvdp_integration, 'base_path', None):
                self.logger.warning("CVDP base_path not available, cannot save initial RTL")
                return

            # Create iterations directory under CVDP base path
            iterations_dir = cvdp_integration.base_path / "iterations"
            iterations_dir.mkdir(exist_ok=True)
            self.logger.info(f"Created iterations directory: {iterations_dir}")

            # Create iteration_0 subdirectory
            iteration_0_dir = iterations_dir / "iteration_0"
            iteration_0_dir.mkdir(exist_ok=True)
            self.logger.info(f"Created iteration_0 directory: {iteration_0_dir}")

            # Save initial RTL code
            rtl_filename = getattr(cvdp_integration, 'expected_rtl_filename', 'generated.sv')
            rtl_file = iteration_0_dir / rtl_filename
            with open(rtl_file, 'w') as f:
                f.write(initial_rtl)
            self.logger.info(f"Saved initial RTL as iteration_0 to {rtl_file}")

            # Save initial error report - try to get full content from reports directory
            error_content = initial_error_report if initial_error_report else "Initial RTL passed tests - no errors"

            # If error report is generic or too short, try to read the full report from reports directory
            should_read_full_report = (
                error_content == "Test failed - see error_report for details" or
                len(error_content) < 100 or
                error_content in ["Initial RTL - no error report available yet", "Initial RTL passed tests - no errors"]
            )

            if should_read_full_report and hasattr(cvdp_integration, 'reports_dir'):
                try:
                    report_files = list(cvdp_integration.reports_dir.glob("*.txt"))
                    if report_files:
                        # Sort by modification time, get most recent
                        report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        if hasattr(cvdp_integration, '_read_report_capped'):
                            full_report = cvdp_integration._read_report_capped(report_files[0])
                        else:
                            with open(report_files[0], 'r', errors='replace') as f:
                                full_report = f.read()
                        if full_report and len(full_report) > 100:  # Only use if substantial content
                            error_content = full_report
                            self.logger.info(f"Read full error report from {report_files[0]} ({len(full_report)} chars)")
                        elif full_report:
                            error_content = full_report
                            self.logger.info(f"Read short error report from {report_files[0]} ({len(full_report)} chars)")
                except Exception as e:
                    self.logger.warning(f"Failed to read full error report: {str(e)}")

            error_file = iteration_0_dir / "error_report.txt"
            with open(error_file, 'w') as f:
                # Handle case where error_content is None (test passed with no error report)
                if error_content is None:
                    f.write("Test passed - no errors")
                else:
                    f.write(error_content)
            content_len = len(error_content) if error_content else 0
            self.logger.info(f"Saved initial error report to {error_file} ({content_len} chars)")

            # Save initial prompt content
            if initial_prompt:
                prompt_file = iteration_0_dir / "prompt.md"
                with open(prompt_file, 'w') as f:
                    f.write(initial_prompt)
                self.logger.info(f"Saved initial prompt to {prompt_file} ({len(initial_prompt)} chars)")
            else:
                self.logger.warning("No initial prompt content available for iteration_0")

        except Exception as e:
            self.logger.error(f"Failed to save initial RTL as iteration_0: {str(e)}")
            # Don't raise the exception - we don't want this to break the main flow

    def _load_generated_target_bundle(
        self,
        code_dir: Path,
        target_file_names: List[str],
        primary_target: Optional[str] = None,
    ) -> str:
        """
        Load all declared generated target files as the editable payload.

        The older loop handed the LLM only one guessed "buggy" file. For CVDP
        adapter-driven multi-file RTL, the actual bug is often a wrapper/core API
        mismatch, so the LLM needs the complete generated target set while still
        treating immutable helper RTL as context only.
        """
        names: List[str] = []
        seen: Set[str] = set()
        for raw_name in target_file_names or []:
            name = Path(str(raw_name)).name
            if not name or name in seen:
                continue
            names.append(name)
            seen.add(name)
        if primary_target:
            primary_name = Path(str(primary_target)).name
            if primary_name and primary_name not in seen:
                names.insert(0, primary_name)
                seen.add(primary_name)
        if not names and primary_target:
            names = [Path(str(primary_target)).name]

        if not names:
            return ""

        def read_or_placeholder(name: str) -> str:
            path = code_dir / name
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    self.logger.warning(f"Could not read generated target {path}: {e}")
            return (
                f"// Target file {name} was missing from the initial CVDP generation.\n"
                "// Generate the complete synthesizable RTL for this file from the specification and context.\n"
            )

        if len(names) == 1:
            return read_or_placeholder(names[0])

        parts = []
        for name in names:
            content = read_or_placeholder(name).rstrip()
            parts.append(f"// TARGET_FILE: {name}\n{content}\n")
        bundle = "\n".join(parts)
        self.logger.info(
            "Loaded generated target bundle for repair: "
            + ", ".join(f"{name}({len(read_or_placeholder(name))} chars)" for name in names)
        )
        return bundle

    def _generated_code_target_names_from_datapoint(
        self,
        datapoint: Dict[str, Any],
        target_dir_name: Optional[str] = None,
    ) -> List[str]:
        """
        Return CVDP-declared generated RTL filenames.

        Non-agentic CVDP rows declare generated files in output.context. Agentic
        rows declare generated/modified deliverables as patch paths. For native
        Docker-free agentic runs we transform rows to generation format for initial
        generation, but the agent still loads the original row, so both sources
        must be understood here.
        """
        if not isinstance(datapoint, dict):
            return []

        rel_paths: List[str] = []
        output_context = datapoint.get("output", {}).get("context", {})
        if isinstance(output_context, dict):
            rel_paths.extend(str(path) for path in output_context)

        patch_context = datapoint.get("patch", {})
        if isinstance(patch_context, dict):
            rel_paths.extend(str(path) for path in patch_context)

        target_names: List[str] = []
        seen: Set[str] = set()
        target_dir_name = target_dir_name.lower() if target_dir_name else None
        for rel_path in rel_paths:
            path = Path(rel_path)
            if path.suffix.lower() not in {".sv", ".v"}:
                continue
            parts_l = [part.lower() for part in path.parts]
            if path.parent.name not in {"rtl", "verif"} and "rtl" not in parts_l and "verif" not in parts_l:
                continue
            if target_dir_name and target_dir_name in {"rtl", "verif"} and target_dir_name not in parts_l:
                continue
            filename = path.name
            if filename and filename not in seen:
                target_names.append(filename)
                seen.add(filename)
        return target_names

    def _select_primary_target_name(
        self,
        target_names: List[str],
        rtl_files: list,
        error_report: str = "",
        spec: str = "",
    ) -> Optional[str]:
        """Choose the generated target most likely to be the harness top/current failure."""
        if not target_names:
            return None

        def has_top_token(filename: str) -> bool:
            stem = Path(filename).stem.lower()
            return bool(re.search(r"(?:^|_)top(?:_|$)", stem))

        text = f"{error_report}\n{spec}"
        top_candidates: List[str] = []
        for pattern in [
            r'root module\s+"([^"]+)"',
            r"-s\s+([A-Za-z_][A-Za-z0-9_$]*)",
            r"hdl_toplevel\s*=\s*['\"]?([A-Za-z_][A-Za-z0-9_$]*)",
            r"TOPLEVEL\s*=\s*([A-Za-z_][A-Za-z0-9_$]*)",
            r"not found in `([A-Za-z_][A-Za-z0-9_$]*)`",
            r"top-level module (?:called|named)?\s*`?([A-Za-z_][A-Za-z0-9_$]*)`?",
            r"module (?:called|named)?\s*`([A-Za-z_][A-Za-z0-9_$]*)`",
        ]:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                name = match.group(1)
                if name not in top_candidates:
                    top_candidates.append(name)

        target_by_stem = {Path(name).stem.lower(): name for name in target_names}
        for module_name in top_candidates:
            filename = target_by_stem.get(module_name.lower())
            if filename:
                self.logger.info(f"Selected generated target from harness top/module error: {filename}")
                return filename

        existing_names = {rtl_file.name for rtl_file in rtl_files}
        for name in target_names:
            if name in existing_names and has_top_token(name):
                self.logger.info(f"Selected existing top-like generated target: {name}")
                return name
        for name in target_names:
            if name in existing_names:
                self.logger.info(f"Selected existing generated target: {name}")
                return name
        for name in target_names:
            if has_top_token(name):
                self.logger.info(f"Selected missing top-like generated target: {name}")
                return name
        return target_names[0]

    def _identify_generated_file_from_datapoint_output(
        self,
        datapoint: Dict[str, Any],
        rtl_files: list,
        error_report: str = "",
        spec: str = "",
        target_dir_name: Optional[str] = None,
    ) -> Optional[str]:
        """Prefer CVDP-declared generated file names for multi-file artifacts."""
        rtl_by_name = {rtl_file.name: rtl_file for rtl_file in rtl_files}
        target_names = self._generated_code_target_names_from_datapoint(
            datapoint,
            target_dir_name=target_dir_name,
        )
        preferred = self._select_primary_target_name(
            target_names,
            rtl_files,
            error_report=error_report,
            spec=spec,
        )
        if preferred:
            if preferred in rtl_by_name:
                self.logger.info(f"Identified generated file from CVDP target metadata: {preferred}")
            else:
                self.logger.info(
                    f"Identified missing generated target from CVDP metadata; will create it: {preferred}"
                )
            return preferred
        if target_names:
            filename = target_names[0]
            self.logger.info(
                f"Identified missing generated target from CVDP metadata; will create it: {filename}"
            )
            return filename
        return None

    def _identify_buggy_file_from_initial_prompt(self, rtl_files: list, spec: str) -> str:
        """
        Identify which file is the buggy one by analyzing the initial prompt/spec.

        CVDP format:
        - "Modify the RTL code for the **XXX**" - indicates which module to modify
        - "Save your RTL implementation as `filename.sv`" - explicit save instruction
        - "Consider the following content:" appears multiple times - one for buggy, others for context

        Returns the filename (not full path) of the buggy file.
        """
        import re

        # Strategy 1: Look for explicit save instruction (most reliable)
        # "Save your RTL implementation as `elevator_control_system.sv`"
        save_match = re.search(r'Save your RTL implementation as `([^`]+\.sv)`', spec)
        if save_match:
            filename = save_match.group(1)
            # Check if this file exists in rtl_files
            for rtl_file in rtl_files:
                if rtl_file.name == filename:
                    self.logger.info(f"Identified buggy file from save instruction: {filename}")
                    return filename

        # Strategy 2: Look for "Modify the RTL code for the **XXX**"
        modify_match = re.search(r'Modify the RTL code for (?:the )?\*\*([^*]+)\*\*', spec)
        if modify_match:
            module_desc = modify_match.group(1).lower()
            # Match description to filename
            for rtl_file in rtl_files:
                filename_parts = rtl_file.name.replace('.sv', '').replace('_', ' ').lower()
                if module_desc in filename_parts or filename_parts in module_desc:
                    self.logger.info(f"Identified buggy file from modify instruction: {rtl_file.name}")
                    return rtl_file.name

        # Strategy 3: First "Consider the following content:" is usually the buggy file
        # (assuming CVDP shows incomplete file first, complete files later)

        # Strategy 4: Fallback - use naming heuristics
        for rtl_file in rtl_files:
            if any(kw in rtl_file.name.lower() for kw in ['system', 'control', 'main', 'top']):
                self.logger.info(f"Identified buggy file by naming: {rtl_file.name}")
                return rtl_file.name

        # Last resort: return first file
        self.logger.warning(f"Could not identify buggy file, using first: {rtl_files[0].name}")
        return rtl_files[0].name

    def _initialize_iteration_log(self, datapoint_name: str, cvdp_int=None):
        """Initialize iteration log file with header."""
        cvdp_integration = cvdp_int or self.cvdp_integration

        if getattr(cvdp_integration, 'base_path', None):
            # Debug: Show exactly where base_path points
            self.logger.info(f"Base path for iteration log: {cvdp_integration.base_path}")

            # Create iterations directory first
            iterations_dir = cvdp_integration.base_path / "iterations"
            iterations_dir.mkdir(exist_ok=True)

            # Create iteration log file inside iterations directory
            log_file = iterations_dir / "iterations_log.txt"

            # Debug: Show exact log file path
            self.logger.info(f"Creating iteration log at: {log_file.resolve()}")

            # Clear any existing log and write header
            try:
                with open(log_file, 'w') as f:
                    f.write(f"Iteration Log for {datapoint_name}\n")
                    f.write(f"Model: {self.config.model_name}\n")
                    f.write(f"Base Path: {cvdp_integration.base_path}\n")
                    f.write(f"=" * 50 + "\n")
                self.logger.info(f"Initialized iteration log: {log_file}")
            except Exception as e:
                self.logger.error(f"Failed to initialize iteration log: {str(e)}")
        else:
            self.logger.warning("base_path not available for iteration log initialization")

    def _log_iteration_result(self, datapoint_name: str, iteration: int, passed: bool, description: str, cvdp_int=None):
        """Log iteration result to simple text file inside iterations directory."""
        cvdp_integration = cvdp_int or self.cvdp_integration

        if getattr(cvdp_integration, 'base_path', None):
            # Create iterations directory if it doesn't exist
            iterations_dir = cvdp_integration.base_path / "iterations"
            iterations_dir.mkdir(exist_ok=True)

            # Create iteration log file inside iterations directory
            log_file = iterations_dir / "iterations_log.txt"

            # Prepare log entry
            status = "PASS" if passed else "FAIL"
            log_entry = f"iteration_{iteration}: {status} - {description}\n"

            try:
                # Append to log file
                with open(log_file, 'a') as f:
                    f.write(log_entry)
                self.logger.info(f"Logged iteration {iteration} to: {log_file}")
            except Exception as e:
                self.logger.warning(f"Failed to log iteration result: {str(e)}")
        else:
            self.logger.warning(f"base_path not available for logging iteration {iteration}")

    def _save_iteration_count(self, output_dir: Path, final_report: Dict[str, Any]):
        """Save iteration count to a text file in the output directory."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Find the iteration where tests first passed
            iteration_count = None
            if final_report.get('success', False):
                iteration_history = final_report.get('iteration_history', [])
                for iteration in iteration_history:
                    if iteration.get('test_passed', False):
                        iteration_count = iteration.get('iteration', None)
                        break

            # If no pass found, report the last completed iteration. This keeps
            # initial-generation/API failures at iteration 0 instead of making
            # them look like they consumed the full repair budget.
            if iteration_count is None:
                iteration_history = final_report.get('iteration_history', [])
                if iteration_history:
                    iteration_count = max(iteration.get('iteration', 0) for iteration in iteration_history)
                else:
                    iteration_count = self.config.max_iterations

            # Save to text file
            iteration_count_file = output_dir / "iteration_count.txt"
            with open(iteration_count_file, 'w') as f:
                if final_report.get('success', False):
                    f.write(f"Iteration Count to Pass: {iteration_count}\n")
                    f.write(f"Status: SUCCESS\n")
                else:
                    f.write(f"Iteration Count (Max Reached): {iteration_count}\n")
                    f.write(f"Status: FAILED\n")
                f.write(f"Total Iterations: {final_report.get('total_iterations', 0)}\n")
                f.write(f"Datapoint: {final_report.get('datapoint_name', 'Unknown')}\n")

            self.logger.info(f"Saved iteration count to: {iteration_count_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save iteration count: {str(e)}")

    def _append_to_summary_log(self, final_report: Dict[str, Any]):
        """Append datapoint results to summary log file at top level of output directory."""
        try:
            # Extract information from final report
            datapoint_name = final_report.get('datapoint_name', 'Unknown')
            experiment_id = final_report.get('experiment_id', 0)

            # Get CID from final report (extracted from datapoint categories)
            cid = final_report.get('cid', 'Unknown')

            # Find the iteration where tests first passed
            iteration_count = None
            if final_report.get('success', False):
                iteration_history = final_report.get('iteration_history', [])
                for iteration in iteration_history:
                    if iteration.get('test_passed', False):
                        iteration_count = iteration.get('iteration', None)
                        break

            # If no pass found, report the last completed iteration. This keeps
            # initial-generation/API failures at iteration 0 instead of making
            # them look like they consumed the full repair budget.
            if iteration_count is None:
                iteration_history = final_report.get('iteration_history', [])
                if iteration_history:
                    iteration_count = max(iteration.get('iteration', 0) for iteration in iteration_history)
                else:
                    iteration_count = self.config.max_iterations

            # Determine summary log path (at top level of output_dir)
            summary_log_path = Path(self.config.output_dir) / "results_summary.log"

            # Create header if file doesn't exist
            file_exists = summary_log_path.exists()

            # Append to summary log
            with open(summary_log_path, 'a') as f:
                if not file_exists:
                    # Write header
                    f.write(f"{'CID':<10} {'Datapoint':<50} {'ExperimentID':<15} {'IterationCount':<15} {'Status':<10}\n")
                    f.write(f"{'-'*10} {'-'*50} {'-'*15} {'-'*15} {'-'*10}\n")

                # Write data row
                status = "SUCCESS" if final_report.get('success', False) else "FAILED"
                f.write(f"{cid:<10} {datapoint_name:<50} {experiment_id:<15} {iteration_count:<15} {status:<10}\n")

            self.logger.info(f"Appended results to summary log: {summary_log_path}")
        except Exception as e:
            self.logger.warning(f"Failed to append to summary log: {str(e)}")

    def _save_results(self, output_dir: Path, final_report: Dict[str, Any]):
        """Save final results to CVDP structure (no JSON files at top level)."""
        benchmark_integration = self.benchmark_integration
        # Don't save any JSON files at the top level as requested
        # All iteration tracking is done in iterations/ subdirectories
        self.logger.info(f"Final report summary: Success={final_report['success']}, Iterations={final_report['total_iterations']}")

        # Save iteration count to text file
        self._save_iteration_count(output_dir, final_report)

        # Optionally save final successful RTL in CVDP structure
        if getattr(benchmark_integration, 'base_path', None) and final_report["success"] and final_report["final_rtl_code"]:
            cvdp_base = benchmark_integration.base_path
            generated_name = getattr(benchmark_integration, 'expected_rtl_filename', 'generated.sv')
            rtl_file = cvdp_base / f"final_{generated_name}"
            with open(rtl_file, 'w') as f:
                f.write(final_report["final_rtl_code"])
            self.logger.info(f"Saved final successful generated code to: {rtl_file}")

        # Clean up unwanted files like raw_result.json
        self._cleanup_unwanted_files()

        # Also call CVDP-specific cleanup to be extra sure
        if hasattr(benchmark_integration, '_cleanup_unwanted_cvdp_files'):
            try:
                benchmark_integration._cleanup_unwanted_cvdp_files()
            except Exception as e:
                self.logger.warning(f"CVDP file cleanup failed while saving results: {e}")

        # Log that results are available in iterations/ subdirectories
        if getattr(benchmark_integration, 'base_path', None):
            iterations_dir = benchmark_integration.base_path / "iterations"
            if iterations_dir.exists():
                self.logger.info(f"All iteration results available in: {iterations_dir}")
            else:
                self.logger.warning("No iterations directory found")

    def _cleanup_unwanted_files(self):
        """Clean up unwanted files like raw_result.json generated by CVDP."""
        benchmark_integration = self.benchmark_integration
        if hasattr(benchmark_integration, 'wrapper') and benchmark_integration.wrapper:
            # Look for and remove raw_result.json at various levels
            prefix_path = Path(benchmark_integration.wrapper.repo.prefix)

            # Check multiple locations where CVDP might create raw_result.json
            possible_locations = [
                prefix_path / "raw_result.json",
                prefix_path.parent / "raw_result.json",
                prefix_path.parent.parent / "raw_result.json",  # Check higher levels
            ]
            if getattr(benchmark_integration, 'base_path', None):
                possible_locations.append(benchmark_integration.base_path / "raw_result.json")

            # Also check for report.json and other unwanted JSON files
            unwanted_patterns = ["raw_result.json", "report.json", "final_report.json"]

            for pattern in unwanted_patterns:
                for level in [prefix_path, prefix_path.parent, prefix_path.parent.parent]:
                    json_file = level / pattern
                    if json_file.exists():
                        try:
                            json_file.unlink()
                            self.logger.info(f"Removed unwanted file: {json_file}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove {json_file}: {str(e)}")

            # Also clean up in base_path if different
            if getattr(benchmark_integration, 'base_path', None):
                base_path = Path(benchmark_integration.base_path)
                for pattern in unwanted_patterns:
                    json_file = base_path / pattern
                    if json_file.exists():
                        try:
                            json_file.unlink()
                            self.logger.info(f"Removed unwanted file from base_path: {json_file}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove {json_file}: {str(e)}")

    def _make_json_serializable(self, obj: Any, _seen: Optional[Set[int]] = None, _depth: int = 0) -> Any:
        """Convert dataclass objects and other non-serializable objects to JSON-serializable format."""
        # Protection against infinite recursion
        if _depth > 10:  # Max recursion depth
            return f"<max_depth_exceeded_{type(obj).__name__}>"

        if _seen is None:
            _seen = set()

        # Check for circular references
        obj_id = id(obj)
        if obj_id in _seen:
            return f"<circular_reference_{type(obj).__name__}>"

        # Handle basic JSON-serializable types first
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        # Add to seen set for complex objects
        _seen.add(obj_id)

        try:
            if is_dataclass(obj):
                result = asdict(obj)
                return self._make_json_serializable(result, _seen, _depth + 1)
            elif isinstance(obj, dict):
                return {key: self._make_json_serializable(value, _seen, _depth + 1)
                       for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [self._make_json_serializable(item, _seen, _depth + 1) for item in obj]
            elif hasattr(obj, 'value') and hasattr(obj, 'name'):  # Handle Enums
                return obj.value
            elif hasattr(obj, '__dict__'):
                # Handle objects with __dict__ attribute, but limit what we serialize
                try:
                    simple_dict = {}
                    for key, value in obj.__dict__.items():
                        if not key.startswith('_'):  # Skip private attributes
                            simple_dict[key] = self._make_json_serializable(value, _seen, _depth + 1)
                    return simple_dict
                except Exception:
                    return str(obj)
            else:
                # For other objects, convert to string representation
                return str(obj)
        except Exception as e:
            # If anything goes wrong, return string representation
            return f"<serialization_error_{type(obj).__name__}: {str(e)}>"
        finally:
            # Remove from seen set when done processing this object
            _seen.discard(obj_id)

    def _safe_serialize_debug_summary(self, debug_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Safely serialize debug summary with fallback for complex objects."""
        try:
            return self._make_json_serializable(debug_summary)
        except Exception as e:
            self.logger.warning(f"Debug summary serialization failed: {e}")
            # Return just the essential information
            return {
                "total_iterations": debug_summary.get("total_iterations", 0),
                "strategies_used": debug_summary.get("strategies_used", []),
                "successful_iterations": debug_summary.get("successful_iterations", []),
                "serialization_note": f"Full summary could not be serialized: {str(e)}"
            }

    def _is_transient_iteration_error(self, error: Exception) -> bool:
        """Check if an iteration error is transient and worth retrying."""
        error_text = str(error).lower()
        transient_indicators = [
            "llm call failed", "api call failed", "timeout", "429", "500", "502", "503", "504",
            "connection", "network", "rate limit", "temporarily unavailable",
            "authentication error", "bedrock", "litellm", "unauthorized"
        ]
        return any(indicator in error_text for indicator in transient_indicators)

    def cleanup(self):
        """Clean up agent resources including Docker networks."""
        self.logger.info("Cleaning up agent resources")
        try:
            self.benchmark_integration.cleanup_global_resources()
        except Exception as e:
            self.logger.warning(f"Agent cleanup failed: {str(e)}")
