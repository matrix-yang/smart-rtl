# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RTL Generator - Uses LLM LLM APIs to generate SystemVerilog RTL code with systematic debugging
"""

import json
import os
import re
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path

from .utils.llm_client import LLMClient
from .systematic_rtl_debugger import SystematicRTLDebugger
from .minimal_change_coordinator import MinimalChangeCoordinator
from .fresh_start_coordinator import FreshStartCoordinator
from .failure_context_analyzer import FailureContextAnalyzer
from .progressive_debugging_strategy import ProgressiveDebuggingStrategy, DebuggingApproach
from .enhanced_rtl_debugger import EnhancedRTLDebugger
from .llm_centric_debugger import LLMCentricDebugger



class RTLGenerator:
    """
    Handles RTL code correction using LLM LLM APIs with systematic debugging.

    This class provides both traditional correction and advanced systematic debugging
    approaches. Initial RTL generation is handled by cvdp_benchmark scripts.
    """

    def __init__(self, model_name: str, logger, api_key: Optional[str] = None,
                 max_retries: int = 3, initial_backoff: float = 5.0,
                 use_systematic_debugging: bool = True, llm_context_limit: int = 300000,
                 ace_rtl_stages: set = None, ace_rtl_temperature: float = 0.2,
                 disable_testbench_access: bool = False):
        """
        Initialize the RTL Generator.

        Args:
            model_name: Name of the LLM model to use (for example, "nvidia/nemotron-3-ultra-550b-a55b")
            logger: Logger instance
            api_key: Optional LLM API key (will use environment variable if not provided)
            max_retries: Maximum number of retries for API calls (default: 3)
            initial_backoff: Initial backoff time in seconds (default: 5.0)
            use_systematic_debugging: Enable systematic debugging approach (default: True)
            llm_context_limit: Context limit for LLM error analysis in characters (default: 300000)
            ace_rtl_stages: Set of stages to use ACE-RTL model (0=initial, 1=error analysis, 2=correction)
            ace_rtl_temperature: Temperature for ACE-RTL model API calls (default: 0.2)
        """
        self.model_name = model_name
        self.logger = logger
        self.use_systematic_debugging = use_systematic_debugging
        self.harness_dir = None  # Will be set via set_harness_directory()
        self.testbench_context = None  # Sanitized testbench for VerilogEval/ChipBench
        self.llm_context_limit = llm_context_limit
        self.ace_rtl_stages = ace_rtl_stages if ace_rtl_stages is not None else set()
        self.ace_rtl_temperature = ace_rtl_temperature
        # When True, never expose the testbench/test file to the LLM (it only
        # sees the simulation error report). Prevents hardcoding to the testbench.
        self.disable_testbench_access = disable_testbench_access
        if disable_testbench_access:
            logger.info("Testbench access DISABLED - LLM will only see simulation error reports")

        # Stagnation detection: when the model returns identical RTL several
        # iterations in a row (no code change), debugging is stuck repeating the
        # same answer. After this many consecutive no-change iterations, force a
        # fresh start (regenerate from scratch) instead of burning more iterations.
        self._consecutive_no_change = 0
        self._no_change_restart_threshold = 3

        # Initialize robust LLM client
        self.client = LLMClient(
            model=model_name,
            api_key=api_key,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            logger=logger
        )

        # Initialize minimal change coordinator (based on successful debugging insights)
        self.minimal_change_coordinator = MinimalChangeCoordinator(
            model_name=model_name,
            logger=logger,
            api_key=api_key,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            llm_context_limit=llm_context_limit
        )

        # Initialize fresh start coordinator for detecting hopeless iterations
        self.fresh_start_coordinator = FreshStartCoordinator(
            model_name=model_name,
            logger=logger,
            api_key=api_key,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            ace_rtl_stages=self.ace_rtl_stages,
            ace_rtl_temperature=self.ace_rtl_temperature
        )

        # Initialize failure context analyzer for deep understanding of test failures
        self.failure_analyzer = FailureContextAnalyzer(
            model_name=model_name,
            logger=logger,
            api_key=api_key,
            max_retries=max_retries,
            initial_backoff=initial_backoff
        )

        # Initialize enhanced RTL debugger (NEW - integrates all debugging improvements)
        self.enhanced_debugger = EnhancedRTLDebugger(
            model_name=model_name,
            logger=logger,
            api_key=api_key,
            max_retries=max_retries,
            initial_backoff=initial_backoff
        )

        # Initialize LLM-centric debugger (ALTERNATIVE - simpler, more general approach)
        self.llm_centric_debugger = LLMCentricDebugger(
            model_name=model_name,
            logger=logger,
            api_key=api_key,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            llm_context_limit=llm_context_limit,
            ace_rtl_stages=self.ace_rtl_stages,
            ace_rtl_temperature=self.ace_rtl_temperature
        )

        # Link focused debugger to fresh start coordinator for history trajectory access
        self.fresh_start_coordinator.focused_debugger = self.llm_centric_debugger.focused_debugger

        # Initialize enhanced debugging strategy for traditional correction path
        self.debugging_strategy = ProgressiveDebuggingStrategy(logger)

        # Initialize systematic debugger if enabled (fallback for complex cases)
        if use_systematic_debugging:
            self.systematic_debugger = SystematicRTLDebugger(
                model_name=model_name,
                logger=logger,
                api_key=api_key,
                max_retries=max_retries,
                initial_backoff=initial_backoff
            )
            self.logger.info("Minimal change coordinator + systematic debugging enabled")
        else:
            self.systematic_debugger = None
            self.logger.info("Using minimal change coordinator only")

        self.logger.info("LLM-centric debugger is now the primary approach (simpler, more general)")

    def reset_for_new_datapoint(self) -> None:
        """Clear per-datapoint debugging state before starting a new CVDP item."""
        self._consecutive_no_change = 0
        self.harness_dir = None
        self.testbench_context = None

        if hasattr(self, "llm_centric_debugger") and self.llm_centric_debugger:
            self.llm_centric_debugger.clear_history_for_restart()
            error_analyzer = getattr(self.llm_centric_debugger, "error_analyzer", None)
            rule_advisor = getattr(error_analyzer, "rtl_rule_advisor", None)
            if rule_advisor:
                rule_advisor.reset_session()

        fresh = getattr(self, "fresh_start_coordinator", None)
        if fresh:
            fresh.iteration_history = []
            fresh.fresh_start_count = 0
            fresh.last_fresh_start_iteration = 0
            fresh.harness_dir = None

        if self.systematic_debugger and hasattr(self.systematic_debugger, "debugging_history"):
            self.systematic_debugger.debugging_history = []

        enhanced = getattr(self, "enhanced_debugger", None)
        if enhanced:
            regression = getattr(enhanced, "regression_prevention", None)
            if regression:
                regression.passing_assertions = {}
                if hasattr(regression, "test_phase_progress"):
                    for phase in regression.test_phase_progress:
                        regression.test_phase_progress[phase] = False
            fix_coordinator = getattr(enhanced, "fix_coordinator", None)
            if fix_coordinator:
                fix_coordinator.applied_fixes = {}
                fix_coordinator.fix_history = []

        self.logger.info("Reset RTL generator debugging state for new datapoint")

    def correct_rtl_code(self, specification: str, buggy_code: str, error_report: str,
                        iteration: int = 1, is_restart: bool = False,
                        multi_file_info: dict = None,
                        per_signal_mismatches: list = None) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        Generate corrected RTL code based on error report using focused debugging.

        Args:
            specification: Original problem specification
            buggy_code: RTL code that failed testing
            error_report: Error report from test failure
            iteration: Current iteration number
            per_signal_mismatches: Per-signal mismatch breakdown from simulation

        Returns:
            Tuple of (corrected_rtl_code, prompt_used, response_content, debug_metadata)
        """
        # When testbench access is disabled, withhold the harness dir and the
        # sanitized testbench context so the LLM only sees the error report.
        effective_harness_dir = None if self.disable_testbench_access else self.harness_dir
        effective_tb_context = None if self.disable_testbench_access else self.testbench_context

        # Use LLM-centric debugger as primary approach (SIMPLER - relies on LLM reasoning)
        try:
            print(f"Using LLM-Centric Debugger (primary approach)")
            corrected_rtl, prompt, response, debug_metadata = self.llm_centric_debugger.debug_rtl_with_llm_reasoning(
                specification=specification,
                current_code=buggy_code,
                error_report=error_report,
                iteration=iteration,
                harness_dir=effective_harness_dir,
                is_restart=is_restart,
                multi_file_info=multi_file_info,
                testbench_context=effective_tb_context,
                per_signal_mismatches=per_signal_mismatches
            )

            print(f"LLM-centric debugging succeeded")
            # Removed verbose success logging
            return corrected_rtl, prompt, response, debug_metadata

        except Exception as e:
            self.logger.warning(f"LLM-centric debugging failed: {e}, falling back to enhanced debugging")

            # Fall back to minimal change approach
            print(f"ERROR: LLM-centric failed - {str(e)}")
            try:
                print(f"Using Minimal Change Coordinator (fallback - focuses on primary failure)")
                corrected_rtl, prompt, _, debug_metadata = self.minimal_change_coordinator.debug_rtl_with_minimal_changes(
                    specification=specification,
                    current_code=buggy_code,
                    error_report=error_report,
                    iteration=iteration
                )

                print(f"Minimal change debugging succeeded")
                # Removed verbose success logging
                return corrected_rtl, prompt, "", debug_metadata  # Empty response for fallback methods

            except Exception as e2:
                self.logger.warning(f"Minimal change approach also failed: {e2}, falling back to systematic debugging")

                # Fall back to systematic debugging for complex cases
                print(f"ERROR: Minimal change failed - {str(e2)}")
                if self.use_systematic_debugging and self.systematic_debugger:
                    print(f"Using Systematic RTL Debugger (fallback - forensic analysis)")
                    return self._systematic_correction(specification, buggy_code, error_report, iteration)
                else:
                    print(f"Using Traditional Correction (final fallback)")
                    return self._traditional_correction(specification, buggy_code, error_report)

    def record_iteration_results(self, error_report: str, iteration: int,
                               test_passed: bool, debug_metadata: Dict = None,
                               code_before: str = "", code_after: str = "",
                               mismatch_count: int = -1):
        """
        Record iteration results for history tracking.

        This must be called AFTER testing to record what happened.
        """
        try:
            if debug_metadata and debug_metadata.get('approach') == 'focused':
                self.llm_centric_debugger.record_iteration_result(
                    iteration=iteration,
                    error_report=error_report,
                    metadata=debug_metadata,
                    test_passed=test_passed,
                    code_before=code_before,
                    code_after=code_after,
                    mismatch_count=mismatch_count
                )
                self.logger.info(f"Recorded iteration {iteration} results in focused debugger (mismatches: {mismatch_count})")
        except Exception as e:
            self.logger.warning(f"Failed to record iteration results: {e}")

    def _systematic_correction(self, specification: str, buggy_code: str,
                              error_report: str, iteration: int) -> Tuple[str, str, Dict[str, Any]]:
        """Generate corrected RTL code using systematic debugging approach."""
        self.logger.info(f"Using systematic debugging approach for iteration {iteration}")

        try:
            corrected_rtl, prompt_used, _, debug_metadata = self.systematic_debugger.debug_rtl_systematically(
                specification, buggy_code, error_report, iteration
            )

            self.logger.info(f"Systematic debugging completed: {debug_metadata['strategy_used']}")
            return corrected_rtl, prompt_used, "", debug_metadata  # Empty response for fallback methods

        except Exception as e:
            self.logger.error(f"Systematic debugging failed, falling back to traditional: {str(e)}")
            # Fallback to traditional approach
            rtl_code, prompt, _, debug_metadata = self._traditional_correction(specification, buggy_code, error_report)
            debug_metadata = {
                "strategy_used": "fallback_traditional",
                "reason": f"Systematic debugging failed: {str(e)}"
            }
            return rtl_code, prompt, "", debug_metadata  # Empty response for fallback methods

    def _traditional_correction(self, specification: str, buggy_code: str,
                               error_report: str) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """Generate corrected RTL code using traditional approach."""
        self.logger.info("Using traditional correction approach")

        try:
            # Generate the prompt using enhanced debugging strategy
            prompt = self.debugging_strategy.create_approach_prompt(
                DebuggingApproach.SYSTEMATIC_DEBUGGING,
                specification=specification,
                current_code=buggy_code,
                error_report=error_report,
                enhanced_context="**TRADITIONAL CORRECTION**: Using systematic analysis for targeted debugging"
            )

            # Call LLM with the prompt
            response = self.client.call_llm(prompt)
            if response is None:
                raise ValueError("Failed to generate corrected RTL code - no response from LLM")

            # Extract and validate LLM reasoning
            reasoning_analysis = self.debugging_strategy.extract_llm_reasoning(response)
            self.logger.info(f"Traditional correction - LLM reasoning quality: {reasoning_analysis['reasoning_quality']}")
            if not reasoning_analysis['followed_framework']:
                self.logger.warning("LLM did not follow structured reasoning framework in traditional correction")

            rtl_code = self._extract_rtl_code(response)

            # Prepare debug metadata with reasoning analysis
            debug_metadata = {
                "strategy_used": "traditional_correction",
                "llm_reasoning_analysis": reasoning_analysis,
                "reasoning_compliance": {
                    "followed_framework": reasoning_analysis['followed_framework'],
                    "reasoning_quality": reasoning_analysis['reasoning_quality'],
                    "root_cause_identified": bool(reasoning_analysis['identified_root_cause']),
                    "steps_completed": len(reasoning_analysis['completed_steps'])
                }
            }

            self.logger.info(f"Generated corrected RTL code ({len(rtl_code)} chars)")
            return rtl_code, prompt, "", debug_metadata  # Empty response for fallback methods

        except Exception as e:
            self.logger.error(f"Failed to generate corrected RTL: {str(e)}")
            raise

    def set_harness_directory(self, harness_dir: Path):
        """Set harness directory for testbench analysis."""
        self.harness_dir = harness_dir
        if hasattr(self, "fresh_start_coordinator") and self.fresh_start_coordinator:
            if self.disable_testbench_access:
                self.fresh_start_coordinator.set_harness_directory(None)
            else:
                self.fresh_start_coordinator.set_harness_directory(harness_dir)
        self.logger.info(f"Set harness directory: {harness_dir}")

    def set_testbench_context(self, testbench_context: str):
        """Set sanitized testbench context for VerilogEval/ChipBench debugging."""
        self.testbench_context = testbench_context
        self.logger.info(f"Set testbench context: {len(testbench_context)} chars")

    def update_iteration_result(self, iteration: int, test_passed: bool,
                               attempted_fix: str = "", reasoning: str = "",
                               code_before: str = "", code_after: str = "", error_msg: str = "",
                               debug_metadata: Dict = None, mismatch_count: int = -1):
        """Update test result for learning from attempts."""

        # Track stagnation: count consecutive iterations where the model returned
        # identical RTL (no actual change). A passing test resets it.
        if test_passed:
            self._consecutive_no_change = 0
        elif code_before.strip() and code_after.strip() and code_before.strip() == code_after.strip():
            self._consecutive_no_change += 1
            self.logger.warning(
                f"No code change at iteration {iteration} "
                f"(consecutive={self._consecutive_no_change}/{self._no_change_restart_threshold})"
            )
        else:
            self._consecutive_no_change = 0

        # Update fresh start coordinator with iteration result
        self.fresh_start_coordinator.add_iteration_result(
            iteration=iteration,
            test_passed=test_passed,
            error_report=error_msg,
            rtl_code=code_after,
            mismatch_count=mismatch_count,
        )

        # Record in focused debugger (NEW)
        if debug_metadata and debug_metadata.get('approach') == 'focused':
            self.record_iteration_results(
                error_report=error_msg,
                iteration=iteration,
                test_passed=test_passed,
                debug_metadata=debug_metadata,
                code_before=code_before,
                code_after=code_after,
                mismatch_count=mismatch_count
            )

        # Update systematic debugger if available (fallback only)
        if self.systematic_debugger:
            self.systematic_debugger.update_iteration_result(iteration, test_passed)

    def get_debugging_summary(self) -> Optional[Dict[str, Any]]:
        """Get debugging summary."""
        summary = {
            "minimal_change_approach": "Used progressive debugging with minimal changes",
            "fresh_start_analysis": self.fresh_start_coordinator.get_restart_summary()
        }

        if self.systematic_debugger:
            summary["systematic_debugging"] = self.systematic_debugger.get_debugging_summary()

        return summary

    def check_need_fresh_start(self, current_iteration: int) -> Tuple[bool, str]:
        """
        Check if current debugging approach needs fresh start.

        Args:
            current_iteration: Current iteration number

        Returns:
            Tuple of (should_restart, reason)
        """
        # Stagnation override: if the model has returned identical RTL for
        # several consecutive iterations, it is stuck repeating the same answer.
        # Force a fresh start regardless of the conservative restart policy.
        if self._consecutive_no_change >= self._no_change_restart_threshold:
            reason = (
                f"Stagnation: model returned identical RTL for "
                f"{self._consecutive_no_change} consecutive iterations - "
                f"forcing fresh start to break the loop"
            )
            self._consecutive_no_change = 0  # reset so we don't re-trigger immediately
            return True, reason
        return self.fresh_start_coordinator.should_fresh_start(current_iteration)

    def generate_fresh_rtl(self, specification: str, iteration_context: str = "",
                           current_iteration: int = None) -> Tuple[str, str]:
        """
        Generate completely fresh RTL code from scratch.

        Args:
            specification: Original design specification
            iteration_context: Context about what went wrong

        Returns:
            Tuple of (fresh_rtl_code, generation_prompt)
        """
        # Clear debugging history when doing fresh start
        self.llm_centric_debugger.clear_history_for_restart()

        return self.fresh_start_coordinator.generate_fresh_rtl(
            specification,
            iteration_context,
            current_iteration=current_iteration,
        )

# Old structured reasoning method removed - now handled by the coordinator.

    def _extract_rtl_code(self, response: str) -> str:
        """
        Extract RTL code from LLM response using simple token-based approach.

        The LLM is instructed to wrap RTL code with <code>...</code> tokens.
        This is much more reliable than complex regex patterns.

        Args:
            response: Raw response from LLM

        Returns:
            Extracted RTL code

        Raises:
            ValueError: If no valid RTL code can be extracted
        """
        self.logger.debug("Starting token-based RTL extraction from LLM response")

        # Method 1: Extract using <code>...</code> tokens (primary method)
        extracted_code = self._extract_with_code_tokens(response)
        if extracted_code:
            self.logger.debug("Successfully extracted RTL using <code> tokens")
            return extracted_code

        # Method 2: Fallback to markdown code blocks (for backward compatibility)
        extracted_code = self._extract_from_markdown_blocks(response)
        if extracted_code:
            self.logger.debug("Successfully extracted RTL from markdown code blocks")
            return extracted_code

        # Method 3: Check if entire response is valid RTL (last resort)
        if self._is_valid_rtl_structure(response.strip()):
            self.logger.debug("Entire response appears to be valid RTL")
            return response.strip()

        # If all methods fail, this indicates LLM didn't follow instructions
        self.logger.error("Failed to extract RTL code - LLM may not have followed format instructions")
        self.logger.debug("Failed to extract RTL code from LLM response - full response logged separately")
        # Log full response to a separate debug log if needed, don't truncate
        raise ValueError("LLM response does not contain properly formatted RTL code. Please retry the LLM call.")

    def _extract_with_code_tokens(self, response: str) -> Optional[str]:
        """Extract RTL code using <code>...</code> tokens."""
        # Look for <code>...</code> tokens (case insensitive)
        pattern = r'<code>\s*(.*?)\s*</code>'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            code = match.strip()
            if self._is_valid_rtl_structure(code):
                return code

        # The configured reasoning model occasionally returns an opening <code> token without the
        # closing token. Keep this recovery path bounded to complete modules.
        orphan_match = re.search(r'<code\b[^>]*>\s*(.*)', response, re.DOTALL | re.IGNORECASE)
        if orphan_match:
            code = orphan_match.group(1).strip()
            endmodule_matches = list(re.finditer(r'\bendmodule\b', code, re.IGNORECASE))
            if endmodule_matches:
                code = code[:endmodule_matches[-1].end()].strip()
            if self._is_valid_rtl_structure(code):
                self.logger.info("Recovered RTL from unterminated <code> block")
                return code

        return None

    def _extract_from_markdown_blocks(self, response: str) -> Optional[str]:
        """Fallback: Extract RTL from markdown code blocks."""
        # Try common RTL language tags
        language_tags = ['systemverilog', 'verilog', 'sv', 'v']

        for tag in language_tags:
            pattern = rf'```{tag}\s*\n(.*?)\n```'
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                code = match.strip()
                if self._is_valid_rtl_structure(code):
                    return code

        # Try generic code blocks that contain modules
        pattern = r'```\s*\n(.*?)\n```'
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            code = match.strip()
            if self._is_valid_rtl_structure(code):
                return code

        return None

    def _is_valid_rtl_structure(self, code: str) -> bool:
        """
        Comprehensive validation to check if extracted code has valid RTL structure.
        Works with single modules, multi-module hierarchies, and various RTL constructs.

        Args:
            code: Code to validate

        Returns:
            True if code appears to be valid RTL structure
        """
        if not code or not code.strip():
            return False

        code_clean = code.strip()

        # Remove comments and empty lines for analysis
        lines_clean = []
        for line in code_clean.split('\n'):
            line_no_comment = re.sub(r'//.*', '', line).strip()
            if line_no_comment:
                lines_clean.append(line_no_comment)

        if not lines_clean:
            return False

        code_no_comments = '\n'.join(lines_clean)

        # Check 1: Must contain at least one module declaration
        module_declarations = re.findall(r'\bmodule\s+(\w+)', code_no_comments, re.IGNORECASE)
        if not module_declarations:
            return False

        # Check 2: Each module must have corresponding endmodule
        endmodule_count = len(re.findall(r'\bendmodule\b', code_no_comments, re.IGNORECASE))
        if len(module_declarations) != endmodule_count:
            return False

        # Check 3: Must have reasonable content (not just module/endmodule)
        if len(code_no_comments) < 50:  # Minimum reasonable module size
            return False

        # Check 4: Should contain typical RTL constructs
        rtl_constructs = [
            r'\b(input|output|inout)\b',  # Port declarations
            r'\b(wire|reg|logic)\b',      # Signal declarations
            r'\b(always|assign)\b',       # RTL logic
            r'\b(initial|task|function)\b', # Testbench/verification logic
            r'\b(parameter|localparam)\b' # Parameters
        ]

        construct_found = any(re.search(pattern, code_no_comments, re.IGNORECASE)
                            for pattern in rtl_constructs)

        if not construct_found:
            return False

        # Check 5: Balanced parentheses and brackets (basic syntax check)
        if not self._check_balanced_delimiters(code_no_comments):
            return False

        return True

    def _check_balanced_delimiters(self, code: str) -> bool:
        """Check if parentheses, brackets, and braces are balanced."""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}

        # Simple state machine to skip string literals
        in_string = False
        escape_next = False

        for char in code:
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not in_string:
                in_string = True
                continue
            elif char == '"' and in_string:
                in_string = False
                continue

            if in_string:
                continue

            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                last_open = stack.pop()
                if pairs[last_open] != char:
                    return False

        return len(stack) == 0

    # Keep backward compatibility
    def _is_valid_rtl(self, code: str) -> bool:
        """Backward compatibility wrapper."""
        return self._is_valid_rtl_structure(code)
