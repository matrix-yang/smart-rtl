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


class AgentLoopMixin:
    def _iterative_improvement_loop(self, datapoint_name: str, rtl_gen, feedback_an, dp_proc, benchmark_int, iteration_results: List[IterationResult], current_datapoint: Dict) -> IterationResult:
        """
        Main iterative improvement loop that generates initial RTL and
        corrects it based on error reports until success or max iterations.

        Args:
            iteration_results: Local list to store iteration results (thread-safe)
            current_datapoint: Local datapoint specification (thread-safe)
        """
        self.logger.info(f"Starting iterative improvement loop for {datapoint_name}")

        # Get the specification for error correction
        spec = dp_proc.extract_specification(current_datapoint) if self.benchmark_mode == "cvdp" else benchmark_int.get_problem_prompt(datapoint_name)

        # Generate and test initial RTL
        self.logger.info(f"Generating initial RTL using {self.benchmark_mode} mode")
        if self.benchmark_mode in ("verilogeval", "chipbench", "chipverilog"):
            # For direct benchmark integrations, generate RTL here and test it
            # behind the integration's evaluator boundary.
            initial_rtl, _ = rtl_gen.generate_fresh_rtl(spec, "")  # No previous context
            initial_test_result = benchmark_int.test_rtl_code(datapoint_name, initial_rtl)
            initial_test_result["rtl_code"] = initial_rtl
        else:
            # For CVDP, use the benchmark infrastructure
            try:
                initial_test_result = benchmark_int.generate_and_test_initial_rtl(datapoint_name)
            except Exception as e:
                self.logger.error(f"Initial CVDP generation/testing failed for {datapoint_name}: {e}")
                failed_initial = IterationResult(
                    iteration=0,
                    rtl_code="",
                    test_passed=False,
                    error_message=f"Initial CVDP generation/testing failed: {str(e)}",
                    execution_time=0.0,
                    compilation_success=False,
                    test_results={"error_report": str(e)},
                    prompt_content=None,
                    response_content=None,
                )
                iteration_results.append(failed_initial)
                return failed_initial

        if (
            self.benchmark_mode == "cvdp"
            and getattr(benchmark_int, "base_path", None)
            and not initial_test_result.get("rtl_code")
        ):
            code_dir = getattr(benchmark_int, 'generated_code_dir', getattr(benchmark_int, 'rtl_dir', None))
            target_file_names = []
            if code_dir:
                target_file_names = self._generated_code_target_names_from_datapoint(
                    current_datapoint,
                    target_dir_name=Path(code_dir).name,
                )
            if target_file_names and code_dir:
                initial_error_text = (
                    initial_test_result.get("error_report")
                    or initial_test_result.get("error_message")
                    or ""
                )
                rtl_files = sorted(list(code_dir.glob("*.sv")) + list(code_dir.glob("*.v")))
                target_name = self._select_primary_target_name(
                    target_file_names,
                    rtl_files,
                    error_report=initial_error_text,
                    spec=spec,
                ) or target_file_names[0]
                target_path = code_dir / target_name
                initial_test_result["rtl_code"] = self._load_generated_target_bundle(
                    code_dir=Path(code_dir),
                    target_file_names=target_file_names,
                    primary_target=target_name,
                )
                setattr(benchmark_int, "expected_rtl_filename", target_name)
                setattr(benchmark_int, "generated_code_file", target_path)
                setattr(benchmark_int, "generated_code_target_files", target_file_names)
                self.logger.warning(
                    f"Initial CVDP generation produced no RTL, but declared target {target_name}; continuing with repair loop"
                )

        if self.benchmark_mode == "cvdp" and (
            not getattr(benchmark_int, "base_path", None)
            or not initial_test_result.get("rtl_code")
        ):
            error_text = (
                initial_test_result.get("error_report")
                or initial_test_result.get("error_message")
                or "Initial CVDP generation did not produce a harness or generated code"
            )
            self.logger.error(
                f"Initial CVDP generation/testing produced no usable harness for {datapoint_name}: {error_text}"
            )
            failed_initial = IterationResult(
                iteration=0,
                rtl_code=initial_test_result.get("rtl_code", ""),
                test_passed=False,
                error_message=f"Initial CVDP generation/testing failed: {error_text}",
                execution_time=initial_test_result.get("execution_time", 0.0),
                compilation_success=initial_test_result.get("compilation_success", False),
                test_results={"error_report": error_text, "details": initial_test_result.get("details")},
                prompt_content=None,
                response_content=None,
            )
            iteration_results.append(failed_initial)
            return failed_initial

        # Initialize iteration log AFTER structure discovery
        self._initialize_iteration_log(datapoint_name, benchmark_int)

        # Log initial result to iteration log
        self._log_iteration_result(datapoint_name, 0, initial_test_result["passed"], f"Initial {self.benchmark_mode} generation", benchmark_int)

        # Save initial RTL as iteration_0 - ALWAYS create iterations folder for tracking
        current_rtl = initial_test_result.get("rtl_code", "")
        self.logger.info(f"Initial RTL code length: {len(current_rtl) if current_rtl else 0}")

        # Step 1: Detect multi-file designs (local to this experiment)
        multi_file_info = None

        code_dir = getattr(benchmark_int, 'generated_code_dir', getattr(benchmark_int, 'rtl_dir', None))
        if code_dir:
            try:
                rtl_files = sorted(list(code_dir.glob("*.sv")) + list(code_dir.glob("*.v")))
                target_file_names = self._generated_code_target_names_from_datapoint(
                    current_datapoint,
                    target_dir_name=Path(code_dir).name,
                )

                if len(rtl_files) > 1:
                    # Step 1: Multi-file design detected
                    self.logger.info(f"Step 1: Detected multi-file design with {len(rtl_files)} files: {[f.name for f in rtl_files]}")

                    # Step 2: Identify the generated file to repair. CVDP
                    # verification tasks list generated deliverables in
                    # output.context; input.context often contains immutable
                    # helpers and must not be selected as the file under repair.
                    initial_error_text = (
                        initial_test_result.get("error_report")
                        or initial_test_result.get("error_message")
                        or ""
                    )
                    buggy_file_name = (
                        self._identify_generated_file_from_datapoint_output(
                            current_datapoint,
                            rtl_files,
                            error_report=initial_error_text,
                            spec=spec,
                            target_dir_name=Path(code_dir).name,
                        )
                        or self._identify_buggy_file_from_initial_prompt(rtl_files, spec)
                    )
                    self.logger.info(f"Step 2: Identified generated file to repair: {buggy_file_name}")

                    target_file_set = set(target_file_names)
                    if target_file_set:
                        context_file_names = [f.name for f in rtl_files if f.name not in target_file_set]
                    else:
                        context_file_names = [f.name for f in rtl_files if f.name != buggy_file_name]

                    # Store multi-file metadata (local to this experiment)
                    multi_file_info = {
                        'buggy_file': buggy_file_name,
                        'context_files': context_file_names,
                        'target_files': target_file_names,
                        'additional_target_files': [name for name in target_file_names if name != buggy_file_name],
                        'rtl_dir': code_dir,
                    }
                    if target_file_names:
                        self.logger.info(f"CVDP declared generated target file(s): {target_file_names}")
                    self.logger.info(f"Context files (won't be modified): {multi_file_info['context_files']}")

                    # Always load the selected generated file. The initial
                    # extractor may have picked the largest file, which is not
                    # necessarily the requested output file in generated verification target.
                    buggy_file_path = code_dir / buggy_file_name
                    current_rtl = self._load_generated_target_bundle(
                        code_dir=Path(code_dir),
                        target_file_names=target_file_names or [buggy_file_name],
                        primary_target=buggy_file_name,
                    )
                    if not buggy_file_path.exists():
                        self.logger.warning(
                            f"Selected generated target file does not exist yet; will create during repair: {buggy_file_path}"
                        )
                    setattr(benchmark_int, "expected_rtl_filename", buggy_file_name)
                    setattr(benchmark_int, "generated_code_file", buggy_file_path)
                    setattr(benchmark_int, "generated_code_target_files", target_file_names)
                    self.logger.info(f"Loaded selected generated file for repair: {buggy_file_name}")

                elif target_file_names:
                    initial_error_text = (
                        initial_test_result.get("error_report")
                        or initial_test_result.get("error_message")
                        or ""
                    )
                    buggy_file_name = self._select_primary_target_name(
                        target_file_names,
                        rtl_files,
                        error_report=initial_error_text,
                        spec=spec,
                    ) or target_file_names[0]
                    target_file_set = set(target_file_names)
                    multi_file_info = {
                        'buggy_file': buggy_file_name,
                        'context_files': [f.name for f in rtl_files if f.name not in target_file_set],
                        'target_files': target_file_names,
                        'additional_target_files': [name for name in target_file_names if name != buggy_file_name],
                        'rtl_dir': code_dir,
                    }
                    buggy_file_path = code_dir / buggy_file_name
                    current_rtl = self._load_generated_target_bundle(
                        code_dir=Path(code_dir),
                        target_file_names=target_file_names or [buggy_file_name],
                        primary_target=buggy_file_name,
                    )
                    if buggy_file_path.exists():
                        self.logger.info(f"Loaded selected generated file for repair: {buggy_file_name}")
                    else:
                        self.logger.warning(
                            f"Selected generated target file does not exist yet; will create during repair: {buggy_file_path}"
                        )
                    setattr(benchmark_int, "expected_rtl_filename", buggy_file_name)
                    setattr(benchmark_int, "generated_code_file", buggy_file_path)
                    setattr(benchmark_int, "generated_code_target_files", target_file_names)

                elif len(rtl_files) == 1 and not current_rtl:
                    # Single file - read it
                    with open(rtl_files[0], 'r') as f:
                        current_rtl = f.read()
                    self.logger.info(f"Read RTL code from: {rtl_files[0].name}")

            except Exception as e:
                self.logger.warning(f"Failed to process RTL files: {str(e)}")

        if current_rtl:
            # Get full error report content, not just the generic error_message
            if initial_test_result["passed"]:
                initial_error_report = "Initial test passed successfully"
            else:
                # Use the detailed error_report if available, otherwise fall back to error_message
                initial_error_report = initial_test_result.get("error_report",
                                                             initial_test_result.get("error_message", "No detailed error report available"))

            self.logger.info(f"Calling _save_initial_rtl_as_iteration_0 for datapoint: {datapoint_name}")
            self.logger.info(f"Error report length: {len(initial_error_report) if initial_error_report else 0}")
            self._save_initial_rtl_as_iteration_0(current_rtl, initial_error_report, spec, benchmark_int)
        else:
            self.logger.warning(f"No RTL code available for iteration_0 save - initial_test_result keys: {list(initial_test_result.keys())}")
            # Create empty iterations folder anyway for consistency
            self._save_initial_rtl_as_iteration_0("// RTL code not available", "Initial RTL code not captured", spec, benchmark_int)

        if initial_test_result["passed"]:
            self.logger.info(
                "SUCCESS: Initial RTL from %s passed tests", self.benchmark_mode
            )
            # Create success result for initial iteration
            initial_success_result = IterationResult(
                iteration=0,  # This is iteration 0 (initial generation)
                rtl_code=initial_test_result.get("rtl_code", ""),
                test_passed=True,
                execution_time=initial_test_result.get("execution_time", 0.0),
                compilation_success=True,
                test_results={
                    "error_report": initial_test_result.get("error_report", "Test passed successfully"),
                    "details": initial_test_result.get("details")
                },
                prompt_content=spec,  # The initial prompt is just the specification
                response_content=None  # No LLM response for initial generation
            )

            # Add the successful initial result to iteration_results so it's tracked correctly
            iteration_results.append(initial_success_result)

            return initial_success_result

        current_error_report = initial_test_result.get("error_report", initial_test_result.get("error_message", ""))

        # Best-code preservation: track the best-performing RTL to avoid regression
        initial_mismatch = -1
        current_per_signal = None
        if initial_test_result.get("details") and isinstance(initial_test_result["details"], dict):
            initial_mismatch = initial_test_result["details"].get("mismatches", -1)
            current_per_signal = initial_test_result["details"].get("per_signal_mismatches")
        best_rtl = current_rtl
        best_mismatch_count = initial_mismatch
        best_error_report = current_error_report
        best_per_signal = current_per_signal
        consecutive_reverts = 0  # Track how many times we revert in a row

        # Set harness directory for testbench analysis.
        if hasattr(benchmark_int, 'harness_dir'):
            rtl_gen.set_harness_directory(benchmark_int.harness_dir)

        # Direct integrations may provide sanitized public testbench context.
        # ChipVerilog intentionally returns None to keep its golden TB private.
        if self.benchmark_mode in ("verilogeval", "chipbench", "chipverilog") and hasattr(benchmark_int, 'get_sanitized_testbench'):
            sanitized_tb = benchmark_int.get_sanitized_testbench()
            if sanitized_tb:
                rtl_gen.set_testbench_context(sanitized_tb)

        # Track whether next debug iteration is immediately after a fresh start
        pending_restart_for_next_iteration = False

        try:
            for iteration in range(self.config.max_iterations):
                # Note: In multiprocessing mode, processes are killed by parent when another succeeds
                # No need to check for success event within the process (complete isolation)

                iteration_wall_start = time.time()
                print(f"\nITERATION {iteration + 1}/{self.config.max_iterations}")
                self.logger.info(f"Iteration {iteration + 1} BEGIN at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                # Removed verbose iteration start logging

                # Initialize iteration success flag for this iteration
                iteration_succeeded = False
                goto_next_iteration = False

                # Debug: Check why iteration might be skipped
                if not current_rtl:
                    print(f"WARNING: No RTL code available for iteration {iteration + 1} - skipping")
                    goto_next_iteration = True
                if not current_error_report:
                    print(f"WARNING: No error report available for iteration {iteration + 1} - skipping")
                    goto_next_iteration = True
                if goto_next_iteration:
                    continue

                # Debug: Show that we're proceeding to debugging work
                print(f"Starting debugging work for iteration {iteration + 1}")

                # Check if we need a fresh start before this iteration
                if iteration >= 5:  # Only check fresh start after several iterations
                    print(f"Checking if fresh start needed for iteration {iteration + 1}")
                    should_restart, restart_reason = rtl_gen.check_need_fresh_start(iteration + 1)
                    if should_restart:
                        print(f"FRESH START triggered: {restart_reason}")
                        print(f"Using Fresh Start Coordinator (generates new RTL from scratch)")

                        # Reset rule advisor session for fresh start to enable rule retrieval
                        if hasattr(feedback_an, 'comprehensive_error_analyzer') and \
                           hasattr(feedback_an.comprehensive_error_analyzer, 'rtl_rule_advisor'):
                            feedback_an.comprehensive_error_analyzer.rtl_rule_advisor.reset_session()
                            print(f"DEBUG: Reset rule advisor session for fresh start")
                        # Also reset the analyzer used by the LLM-centric debugger
                        if hasattr(rtl_gen, 'llm_centric_debugger') and \
                           hasattr(rtl_gen.llm_centric_debugger, 'error_analyzer') and \
                           hasattr(rtl_gen.llm_centric_debugger.error_analyzer, 'rtl_rule_advisor'):
                            rtl_gen.llm_centric_debugger.error_analyzer.rtl_rule_advisor.reset_session()
                            print(f"DEBUG: Reset LLMCentricDebugger rule advisor for fresh start")

                        # Removed verbose fresh start logging

                        # Generate fresh RTL from scratch
                        try:
                            fresh_start_time = time.time()  # Track fresh start timing
                            fresh_rtl, fresh_prompt = rtl_gen.generate_fresh_rtl(
                                specification=spec,
                                iteration_context=restart_reason,
                                current_iteration=iteration + 1,
                            )

                            print(f"Generated fresh RTL code ({len(fresh_rtl)} chars)")
                            # Removed verbose fresh RTL generation logging

                            # Test fresh RTL directly without debugging contamination
                            print(f"Testing fresh RTL directly")
                            fresh_test_result = benchmark_int.test_rtl_code(datapoint_name, fresh_rtl)
                            fresh_mismatch_count = -1
                            fresh_per_signal = None
                            if fresh_test_result.get("details") and isinstance(fresh_test_result["details"], dict):
                                fresh_mismatch_count = fresh_test_result["details"].get("mismatches", -1)
                                fresh_per_signal = fresh_test_result["details"].get("per_signal_mismatches")

                            # Create fresh start iteration result
                            fresh_iteration_result = IterationResult(
                                iteration=iteration + 1,
                                rtl_code=fresh_rtl,
                                test_passed=fresh_test_result["passed"],
                                error_message=fresh_test_result.get("error_message"),
                                feedback=f"Fresh start generated and tested: {restart_reason}",
                                execution_time=time.time() - fresh_start_time,
                                compilation_success=fresh_test_result.get("compilation_success", False),
                                test_results={
                                    "error_report": fresh_test_result.get("error_report", ""),
                                    "details": fresh_test_result.get("details"),
                                    "fresh_start": True
                                },
                                prompt_content=fresh_prompt,
                                response_content=None  # Fresh start uses CVDP generation, no LLM response
                            )

                            # Update fresh start coordinator with test result
                            rtl_gen.update_iteration_result(
                                iteration=iteration + 1,
                                test_passed=fresh_test_result["passed"],
                                attempted_fix=fresh_rtl,
                                reasoning="Fresh start generation",
                                code_before="",
                                code_after=fresh_rtl,
                                error_msg=fresh_test_result.get("error_report", fresh_test_result.get("error_message", "")),
                                mismatch_count=fresh_mismatch_count,
                            )

                            iteration_results.append(fresh_iteration_result)

                            # Save fresh start result
                            if self.config.save_intermediate:
                                self._save_iteration_result_to_cvdp_structure(fresh_iteration_result, benchmark_int)

                            # Log iteration result
                            self._log_iteration_result(datapoint_name, iteration + 1, fresh_test_result["passed"],
                                                     f"Fresh start - restart #{rtl_gen.fresh_start_coordinator.fresh_start_count}", benchmark_int)

                            # If fresh RTL passes, we're done!
                            if fresh_test_result["passed"]:
                                self.logger.info(f"SUCCESS: Fresh RTL passed tests in iteration {iteration + 1}")
                                return fresh_iteration_result

                            # If fresh RTL fails, set up for debugging in next iteration
                            current_rtl = fresh_rtl
                            current_error_report = fresh_test_result.get("error_report", fresh_test_result.get("error_message", ""))
                            current_per_signal = fresh_per_signal
                            if fresh_mismatch_count >= 0 and (
                                best_mismatch_count < 0 or fresh_mismatch_count < best_mismatch_count
                            ):
                                best_rtl = fresh_rtl
                                best_mismatch_count = fresh_mismatch_count
                                best_error_report = current_error_report
                                best_per_signal = fresh_per_signal
                            # Mark that next iteration is a restart-driven debug step
                            pending_restart_for_next_iteration = True
                            iteration_succeeded = True  # Fresh start completed (even though test failed)
                            continue  # Skip the debugging for this iteration since we already tested

                        except Exception as e:
                            self.logger.error(f"Fresh start generation failed: {e}")
                            # Continue with current approach if fresh start fails

                # Debug: Show that we're entering retry loop
                print(f"Entering retry loop for iteration {iteration + 1}")

                # Retry logic for this specific iteration
                for attempt in range(self.config.iteration_retries + 1):  # +1 for initial attempt
                    if attempt > 0:
                        print(f"Retrying iteration {iteration + 1}, attempt {attempt + 1}")
                        # Removed verbose retry logging

                    print(f"Attempt {attempt + 1} for iteration {iteration + 1}")

                    iteration_start_time = time.time()
                    # Enforce per-iteration wall-clock timeout before starting expensive steps (only if > 0)
                    if self.config.timeout_per_iteration > 0 and (time.time() - iteration_wall_start > self.config.timeout_per_iteration):
                        timeout_msg = (f"Iteration {iteration + 1} exceeded wall-clock timeout "
                                       f"({self.config.timeout_per_iteration}s). Skipping to next iteration.")
                        print(f"TIMEOUT: {timeout_msg}")
                        self._log_iteration_result(datapoint_name, iteration + 1, False, timeout_msg, benchmark_int)
                        iteration_result = IterationResult(
                            iteration=iteration + 1,
                            rtl_code="",
                            test_passed=False,
                            error_message=timeout_msg,
                            execution_time=time.time() - iteration_wall_start,
                            prompt_content=None,
                            response_content=None
                        )
                        iteration_results.append(iteration_result)
                        iteration_succeeded = True
                        break

                    try:
                        # Show component usage plan for this iteration
                        print(f"COMPONENTS TO USE:")
                        print(f"   RTL Generator (orchestrates debugging)")
                        print(f"   LLM-Centric Debugger -> Comprehensive Error Analyzer -> Regression Tracker")
                        print(f"   CVDP Integration (testing & code preservation)")

                        # Generate corrected RTL code using systematic debugging
                        llm_gen_start = time.time()
                        corrected_rtl, correction_prompt, correction_response, debug_metadata = rtl_gen.correct_rtl_code(
                            spec, current_rtl, current_error_report, iteration + 1,
                            is_restart=pending_restart_for_next_iteration,
                            multi_file_info=multi_file_info,
                            per_signal_mismatches=current_per_signal
                        )
                        self.logger.info(f"LLM generation time (iteration {iteration + 1}, attempt {attempt + 1}): {time.time() - llm_gen_start:.2f}s")
                        # Reset the pending flag after use to avoid repeated restart signaling
                        pending_restart_for_next_iteration = False

                        # Skip RTL validation - CVDP testbench will catch any syntax errors

                        # Test the corrected RTL
                        # Enforce per-iteration timeout again before running tests (only if > 0)
                        if self.config.timeout_per_iteration > 0 and (time.time() - iteration_wall_start > self.config.timeout_per_iteration):
                            timeout_msg = (f"Iteration {iteration + 1} exceeded wall-clock timeout "
                                           f"({self.config.timeout_per_iteration}s) before testing. Skipping.")
                            print(f"TIMEOUT: {timeout_msg}")
                            self._log_iteration_result(datapoint_name, iteration + 1, False, timeout_msg, benchmark_int)
                            iteration_result = IterationResult(
                                iteration=iteration + 1,
                                rtl_code=corrected_rtl,
                                test_passed=False,
                                error_message=timeout_msg,
                                execution_time=time.time() - iteration_wall_start,
                                prompt_content=correction_prompt,
                                response_content=correction_response
                            )
                            iteration_results.append(iteration_result)
                            iteration_succeeded = True
                            break

                        test_run_start = time.time()
                        test_result = benchmark_int.test_rtl_code(datapoint_name, corrected_rtl)
                        self.logger.info(f"Test runtime (iteration {iteration + 1}, attempt {attempt + 1}): {time.time() - test_run_start:.2f}s")

                        # Update the RTL generator with test results for debugging history and learning
                        # Use full error_report (not error_message summary) for retrospective analysis
                        full_error_report = test_result.get("error_report", test_result.get("error_message", ""))

                        # Extract numeric mismatch count for objective progress tracking
                        iter_mismatch_count = -1
                        if test_result.get("details") and isinstance(test_result["details"], dict):
                            iter_mismatch_count = test_result["details"].get("mismatches", -1)

                        rtl_gen.update_iteration_result(
                            iteration=iteration + 1,
                            test_passed=test_result["passed"],
                            attempted_fix=corrected_rtl,
                            reasoning=debug_metadata.get("reasoning", "No reasoning extracted"),
                            code_before=current_rtl,
                            code_after=corrected_rtl,
                            error_msg=full_error_report,
                            debug_metadata=debug_metadata,
                            mismatch_count=iter_mismatch_count
                        )

                        execution_time = time.time() - iteration_start_time

                        # Create iteration result with debug metadata
                        iteration_result = IterationResult(
                            iteration=iteration + 1,
                            rtl_code=corrected_rtl,
                            test_passed=test_result["passed"],
                            error_message=test_result.get("error_message"),
                            execution_time=execution_time,
                            compilation_success=test_result.get("compilation_success", False),
                            test_results={
                                "error_report": test_result.get("error_report", test_result.get("error_message", "")),
                                "details": test_result.get("details"),
                                "debug_metadata": debug_metadata  # Include systematic debugging metadata
                            },
                            prompt_content=correction_prompt,
                            response_content=correction_response
                        )

                        iteration_results.append(iteration_result)

                        # Save intermediate results if configured
                        if self.config.save_intermediate:
                            print(f"Saving iteration result to preserve generated code")
                            self._save_iteration_result_to_cvdp_structure(iteration_result, benchmark_int)

                        # Log iteration result
                        self._log_iteration_result(datapoint_name, iteration + 1, test_result["passed"],
                                                 f"LLM correction attempt {iteration + 1}", benchmark_int)

                        # Check if test passed
                        if test_result["passed"]:
                            print(f"SUCCESS: RTL passed tests in iteration {iteration + 1}")
                            print(f"Code preserved for final result")
                            # Removed verbose success logging
                            self.logger.info(f"Iteration {iteration + 1} END (PASS)")
                            return iteration_result
                        else:
                            # Extract and display assertion error details
                            error_msg = test_result.get('error_message', 'Unknown error')
                            error_report = test_result.get('error_report', '')

                            # Look for assertion details in error report
                            assertion_details = self._extract_assertion_details(error_report)
                            if assertion_details:
                                print(f"ERROR: Iteration {iteration + 1} failed - {assertion_details}")
                            else:
                                print(f"ERROR: Iteration {iteration + 1} failed - {error_msg}")
                            # Best-code preservation: revert to best if regression is significant
                            if iter_mismatch_count >= 0 and best_mismatch_count >= 0:
                                if iter_mismatch_count < best_mismatch_count:
                                    prev_best = best_mismatch_count
                                    best_rtl = corrected_rtl
                                    best_mismatch_count = iter_mismatch_count
                                    best_error_report = test_result.get("error_report", test_result.get("error_message", ""))
                                    if test_result.get("details") and isinstance(test_result["details"], dict):
                                        best_per_signal = test_result["details"].get("per_signal_mismatches")
                                    consecutive_reverts = 0
                                    print(f"   New best: {iter_mismatch_count} mismatches (improved from {prev_best})")
                                elif iter_mismatch_count > best_mismatch_count * 1.1 and best_mismatch_count > 0:
                                    consecutive_reverts += 1
                                    failure_stage = ""
                                    if test_result.get("details") and isinstance(test_result["details"], dict):
                                        failure_stage = str(test_result["details"].get("failure_stage", ""))
                                    structural_regression = (
                                        not test_result.get("compilation_success", False)
                                        or failure_stage in {
                                            "cvdp_no_cheat_guard",
                                            "cvdp_contract_guard",
                                            "cvdp_semantic_guard",
                                        }
                                        or "CVDP_RTL_SEMANTIC_GUARD" in error_report
                                        or "CVDP_CONTRACT_GUARD" in error_report
                                        or "CVDP_NO_CHEAT_GUARD" in error_report
                                    )
                                    # Functional exploration can occasionally leave the local optimum.
                                    # Structurally invalid RTL cannot teach the next iteration useful CVDP
                                    # behavior, so always keep the best runnable candidate in that case.
                                    acceptable_functional_detour = (
                                        best_mismatch_count > 0
                                        and iter_mismatch_count <= max(best_mismatch_count * 2, best_mismatch_count + 5000)
                                        and iter_mismatch_count < 100_000
                                    )
                                    if consecutive_reverts >= 3 and not structural_regression and acceptable_functional_detour:
                                        print(f"   Regression: {iter_mismatch_count} mismatches (best was {best_mismatch_count}). "
                                              f"3+ consecutive reverts - accepting to try different path.")
                                        consecutive_reverts = 0
                                        pending_restart_for_next_iteration = True
                                    else:
                                        print(f"   Regression: {iter_mismatch_count} mismatches (best was {best_mismatch_count}). Reverting to best code.")
                                        current_rtl = best_rtl
                                        current_error_report = best_error_report
                                        current_per_signal = best_per_signal
                                        iteration_succeeded = True
                                        self.logger.info(f"Iteration {iteration + 1} END (REGRESSED - reverted to best)")
                                        break
                                else:
                                    consecutive_reverts = 0

                            current_rtl = corrected_rtl
                            current_error_report = test_result.get("error_report", test_result.get("error_message", ""))
                            # Update per-signal mismatch data for next iteration
                            if test_result.get("details") and isinstance(test_result["details"], dict):
                                current_per_signal = test_result["details"].get("per_signal_mismatches")
                            iteration_succeeded = True
                            self.logger.info(f"Iteration {iteration + 1} END (FAIL - proceed)")
                            break  # Exit retry loop for this iteration

                    except Exception as e:
                        print(f"ERROR: Exception in iteration {iteration + 1}, attempt {attempt + 1}: {str(e)}")

                        # Check if this is a transient error worth retrying
                        if self._is_transient_iteration_error(e) and attempt < self.config.iteration_retries:
                            print(f"Transient error - will retry")
                            continue  # Retry this iteration
                        else:
                            # Non-transient error or max retries reached
                            print(f"ERROR: Iteration {iteration + 1} failed after {attempt + 1} attempts")

                            # Log failed iteration
                            self._log_iteration_result(datapoint_name, iteration + 1, False, f"Exception after {attempt + 1} attempts: {str(e)}", benchmark_int)

                            # Create failure result
                            iteration_result = IterationResult(
                                iteration=iteration + 1,
                                rtl_code="",
                                test_passed=False,
                                error_message=f"Failed after {attempt + 1} attempts: {str(e)}",
                                execution_time=time.time() - iteration_start_time,
                                prompt_content=None,
                                response_content=None
                            )
                            iteration_results.append(iteration_result)
                            iteration_succeeded = True  # Mark iteration as completed (even though it failed)
                            self.logger.info(f"Iteration {iteration + 1} END (EXCEPTION - proceed)")
                            break  # Exit retry loop for this iteration

                # If iteration didn't succeed after all retries, continue to next iteration
                if not iteration_succeeded:
                    continue
                self.logger.info(f"Advancing to iteration {iteration + 2}")
        except Exception as e:
            # Catch any unexpected error so we don't exit without a summary
            self.logger.error(f"Unexpected termination during iterative loop: {e}")
            try:
                self._log_iteration_result(datapoint_name, len(iteration_results) + 1, False,
                                           f"Unexpected termination: {str(e)}", benchmark_int)
            except Exception:
                pass

        # If we reach here, all iterations failed
        actual_iterations = len(iteration_results)
        print(f"FINAL FAILURE: Completed {actual_iterations}/{self.config.max_iterations} iterations without success")
        self.logger.error(f"FAILURE: Completed {actual_iterations}/{self.config.max_iterations} iterations without success")
        return iteration_results[-1] if iteration_results else IterationResult(
            iteration=0, rtl_code="", test_passed=False, error_message="No iterations completed",
            prompt_content=None, response_content=None
        )
