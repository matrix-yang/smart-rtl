# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Deep Error Analyzer - Replaces ComprehensiveErrorAnalyzer with deeper analysis.

Uses SignalTracer and TestPatternAnalyzer to provide actionable insights.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from .primary_failure_extractor import PrimaryFailureExtractor, PrimaryFailure
from .signal_tracer import SignalTracer
from .test_pattern_analyzer import TestPatternAnalyzer
from .rtl_rule_advisor import RTLRuleAdvisor


@dataclass
class TestCaseResult:
    """Result of a specific test case."""
    test_name: str
    passed: bool
    assertion_details: Optional[str] = None
    timing_info: Optional[str] = None


@dataclass
class DeepErrorAnalysis:
    """Complete deep analysis of test results."""
    primary_failure: Optional[PrimaryFailure]
    passing_tests: List[TestCaseResult]
    failing_tests: List[TestCaseResult]

    # Deep analysis components
    core_reasoning: str  # Actionable root cause analysis
    signal_trace: str  # Signal path analysis
    test_patterns: Dict[str, Any]  # Pattern analysis results
    potential_issues: List[str]  # Specific RTL issues identified

    # Context checking flags
    should_check_testbench: bool
    should_check_specification: bool
    reasoning_for_context_check: str

    # RTL advice
    rtl_rule_advice: Optional[str] = None


class DeepErrorAnalyzer:
    """
    Deep error analyzer that provides actionable insights.

    Combines:
    1. Signal tracing (where is the issue in RTL)
    2. Test pattern analysis (what inputs/conditions trigger failure)
    3. RTL issue detection (specific bugs identified)
    4. Root cause hypotheses (why it's happening)
    """

    def __init__(self, logger, model_name: str = "nvidia/nemotron-3-ultra-550b-a55b", llm_context_limit: int = 300000):
        self.logger = logger
        self.model_name = model_name
        self.primary_failure_extractor = PrimaryFailureExtractor(logger, model_name, llm_context_limit)
        self.signal_tracer = SignalTracer(logger)
        self.test_pattern_analyzer = TestPatternAnalyzer(logger)
        self.rtl_rule_advisor = RTLRuleAdvisor(model_name=model_name)
        self.llm_context_limit = llm_context_limit
        self.cached_specification: Optional[str] = None

    def analyze_error_deeply(self, error_report: str, rtl_code: str, iteration: int,
                            previous_analyses: List[Dict] = None,
                            specification: str = None,
                            is_restart: bool = False) -> DeepErrorAnalysis:
        """
        Perform deep error analysis with actionable insights.

        Args:
            error_report: Full error report from test
            rtl_code: Current RTL implementation
            iteration: Current iteration number
            previous_analyses: Previous iteration analyses
            specification: Design specification
            is_restart: Whether this is a restart iteration

        Returns:
            Deep analysis with signal traces, patterns, and specific issues
        """
        self.logger.info(f"Starting deep error analysis for iteration {iteration}")

        try:
            # Step 1: Extract primary failure
            primary_failure = self.primary_failure_extractor.extract_primary_failure(error_report)
            self.logger.info(f"Primary failure extracted: {primary_failure is not None}")

            # Step 2: Categorize test results
            passing_tests, failing_tests = self._categorize_test_results(error_report)
            self.logger.info(f"Test categorization: {len(passing_tests)} pass, {len(failing_tests)} fail")

            # Step 3: Analyze RTL structure
            self.logger.info("Analyzing RTL structure with SignalTracer")
            rtl_analysis = self.signal_tracer.analyze_rtl(rtl_code)

            # Step 4: Trace failing signal paths
            signal_trace = ""
            if primary_failure:
                failing_signal = self._identify_failing_signal(primary_failure, rtl_analysis)
                if failing_signal:
                    self.logger.info(f"Tracing signal path for: {failing_signal}")
                    signal_trace = self.signal_tracer.trace_signal_path(failing_signal, rtl_code)
                else:
                    self.logger.info("Could not identify specific failing signal")
                    signal_trace = "Could not identify specific failing signal from error report."
            else:
                signal_trace = "No primary failure identified - cannot trace signal path."

            # Step 5: Analyze test patterns
            self.logger.info("Analyzing test patterns")
            test_patterns = self.test_pattern_analyzer.analyze_patterns(
                passing_tests, failing_tests, error_report
            )

            # Step 6: Identify potential RTL issues
            self.logger.info("Identifying potential RTL issues")
            potential_issues = self.signal_tracer.identify_potential_issues(
                rtl_code,
                failing_signal=self._identify_failing_signal(primary_failure, rtl_analysis)
            )

            # Step 7: Generate core reasoning with all insights
            self.logger.info("Generating core reasoning")
            core_reasoning = self._generate_deep_reasoning(
                primary_failure, error_report, iteration,
                signal_trace, test_patterns, potential_issues,
                rtl_analysis, previous_analyses
            )

            # Step 8: Decide context checking
            should_check_tb, should_check_spec, context_reasoning = self._decide_context_checking(
                primary_failure, error_report, iteration, previous_analyses
            )

            # Step 9: Get RTL rule advice if we have specification
            rtl_rule_advice = None
            if specification:
                self.cached_specification = specification
                try:
                    # Retrieve rules from specification if not already done
                    if not self.rtl_rule_advisor.specification_processed:
                        self.rtl_rule_advisor.retrieve_rules_from_specification(specification)

                    # Get and format the retrieved rules
                    if self.rtl_rule_advisor.has_relevant_rules():
                        rules = self.rtl_rule_advisor.get_retrieved_rules()
                        rtl_rule_advice = self.rtl_rule_advisor.format_advice_for_agent(rules, include_detailed=False)
                except Exception as e:
                    self.logger.warning(f"RTL rule advisor failed: {e}")

            return DeepErrorAnalysis(
                primary_failure=primary_failure,
                passing_tests=passing_tests,
                failing_tests=failing_tests,
                core_reasoning=core_reasoning,
                signal_trace=signal_trace,
                test_patterns=test_patterns,
                potential_issues=potential_issues,
                should_check_testbench=should_check_tb,
                should_check_specification=should_check_spec,
                reasoning_for_context_check=context_reasoning,
                rtl_rule_advice=rtl_rule_advice
            )

        except Exception as e:
            self.logger.error(f"Deep error analysis failed: {e}")
            import traceback
            traceback.print_exc()

            # Return minimal analysis
            return DeepErrorAnalysis(
                primary_failure=None,
                passing_tests=[],
                failing_tests=[],
                core_reasoning=f"Error analysis failed: {str(e)}",
                signal_trace="",
                test_patterns={},
                potential_issues=[],
                should_check_testbench=False,
                should_check_specification=False,
                reasoning_for_context_check="",
                rtl_rule_advice=None
            )

    def _categorize_test_results(self, error_report: str) -> Tuple[List[TestCaseResult], List[TestCaseResult]]:
        """Categorize tests into passing and failing."""
        passing_tests = []
        failing_tests = []

        # Parse cocotb test results
        # Pattern: "test_name passed" or "test_name failed"
        test_pattern = r'([\w\.]+)\s+(passed|failed)'
        matches = re.finditer(test_pattern, error_report, re.IGNORECASE)

        seen_tests = set()
        for match in matches:
            test_name = match.group(1)
            status = match.group(2).lower()

            # Avoid duplicates
            if test_name in seen_tests:
                continue
            seen_tests.add(test_name)

            # Extract assertion details for this test
            assertion_details = self._extract_assertion_for_test(test_name, error_report)

            test_case = TestCaseResult(
                test_name=test_name,
                passed=(status == 'passed'),
                assertion_details=assertion_details
            )

            if status == 'passed':
                passing_tests.append(test_case)
            else:
                failing_tests.append(test_case)

        # If no tests found via pattern, parse assertion errors
        if not passing_tests and not failing_tests:
            assertion_errors = re.findall(r'AssertionError: ([^\n]+)', error_report)
            for i, assertion in enumerate(assertion_errors):
                failing_tests.append(TestCaseResult(
                    test_name=f"test_case_{i+1}",
                    passed=False,
                    assertion_details=f"Error: AssertionError: {assertion}"
                ))

        return passing_tests, failing_tests

    def _extract_assertion_for_test(self, test_name: str, error_report: str) -> Optional[str]:
        """Extract assertion details for a specific test."""
        # Find the section for this test
        lines = error_report.split('\n')

        for i, line in enumerate(lines):
            if test_name in line:
                # Look ahead for assertion errors
                for j in range(i, min(i + 10, len(lines))):
                    if 'AssertionError:' in lines[j]:
                        return lines[j].strip()
                    elif 'Error:' in lines[j]:
                        return lines[j].strip()

        return None

    def _identify_failing_signal(self, primary_failure: Optional[PrimaryFailure],
                                 rtl_analysis: Dict) -> Optional[str]:
        """
        Identify which signal is failing based on the error message.

        General approach: extract signal names from assertion and match to RTL signals.
        """
        if not primary_failure:
            return None

        # Extract potential signal names from assertion
        assertion_text = primary_failure.raw_assertion + " " + (primary_failure.context_lines or "")

        # Common patterns: dut.SIGNAL.value, output signal names
        signal_patterns = [
            r'dut\.(\w+)\.value',  # cocotb pattern
            r'output\s+(\w+)',
            r'signal\s+(\w+)',
            r'(\w+)\s*=\s*\d+'  # Any signal = value pattern
        ]

        potential_signals = set()
        for pattern in signal_patterns:
            matches = re.findall(pattern, assertion_text, re.IGNORECASE)
            potential_signals.update(matches)

        # Match against RTL signals (prefer outputs)
        rtl_signals = rtl_analysis.get('signals', {})

        # First try: exact match with outputs
        for sig_name in potential_signals:
            if sig_name in rtl_signals and rtl_signals[sig_name].signal_type == 'output':
                return sig_name

        # Second try: exact match with any signal
        for sig_name in potential_signals:
            if sig_name in rtl_signals:
                return sig_name

        # Third try: case-insensitive match
        for sig_name in potential_signals:
            for rtl_sig in rtl_signals:
                if sig_name.lower() == rtl_sig.lower():
                    return rtl_sig

        # Last resort: return first output signal
        for sig_name, sig_info in rtl_signals.items():
            if sig_info.signal_type == 'output':
                return sig_name

        return None

    def _generate_deep_reasoning(self, primary_failure: Optional[PrimaryFailure],
                                 error_report: str, iteration: int,
                                 signal_trace: str, test_patterns: Dict,
                                 potential_issues: List[str], rtl_analysis: Dict,
                                 previous_analyses: List[Dict] = None) -> str:
        """
        Generate deep, actionable reasoning about the failure.

        This combines all analysis components into a coherent explanation.
        """
        reasoning_parts = []

        reasoning_parts.append("## ROOT CAUSE ANALYSIS")
        reasoning_parts.append("")

        # Part 1: What's failing (from primary failure)
        if primary_failure:
            reasoning_parts.append("### Primary Failure:")
            reasoning_parts.append(f"- **Expected**: {primary_failure.expected_value}")
            reasoning_parts.append(f"- **Actual**: {primary_failure.actual_value}")
            reasoning_parts.append(f"- **Timing**: {primary_failure.timing_context}")
            reasoning_parts.append("")

        # Part 2: Test patterns (what conditions trigger failure)
        if test_patterns and test_patterns.get('patterns_found'):
            reasoning_parts.append("### Test Pattern Analysis:")
            patterns_list = test_patterns['patterns_found']
            # Ensure it's a list (defensive programming)
            if not isinstance(patterns_list, list):
                patterns_list = list(patterns_list) if patterns_list else []
            for pattern in patterns_list[:4]:  # Top 4 patterns
                reasoning_parts.append(f"- {pattern}")
            reasoning_parts.append("")

        # Part 3: Signal path analysis (where in RTL)
        if signal_trace and "not found" not in signal_trace.lower():
            reasoning_parts.append("### Signal Path Trace:")
            # Add abbreviated trace (first few lines)
            trace_lines = signal_trace.split('\n')[:10]
            reasoning_parts.extend(trace_lines)
            if len(signal_trace.split('\n')) > 10:
                reasoning_parts.append("... (trace abbreviated)")
            reasoning_parts.append("")

        # Part 4: Specific RTL issues found (concrete problems)
        if potential_issues:
            reasoning_parts.append("### RTL Issues Identified:")
            for issue in potential_issues[:5]:  # Top 5 issues
                reasoning_parts.append(f"- {issue}")
            reasoning_parts.append("")

        # Part 5: Hypotheses (why it's happening)
        if test_patterns and test_patterns.get('hypotheses'):
            reasoning_parts.append("### Root Cause Hypotheses:")
            hypotheses_list = test_patterns['hypotheses']
            # Ensure it's a list (defensive programming)
            if not isinstance(hypotheses_list, list):
                hypotheses_list = list(hypotheses_list) if hypotheses_list else []
            for hypothesis in hypotheses_list[:3]:  # Top 3
                reasoning_parts.append(f"- {hypothesis}")
            reasoning_parts.append("")

        # Part 6: RTL structure context
        if rtl_analysis:
            reasoning_parts.append("### RTL Structure Context:")
            reasoning_parts.append(f"- **Signals**: {len(rtl_analysis.get('signals', {}))} total")
            reasoning_parts.append(f"- **Always blocks**: {len(rtl_analysis.get('always_blocks', []))}")
            if rtl_analysis.get('state_machine_detected'):
                reasoning_parts.append(f"- **State machine detected**: Yes (state signal: {rtl_analysis.get('state_signal', 'unknown')})")
                states = rtl_analysis.get('states', [])
                if states:
                    # Ensure it's a list (defensive programming)
                    if not isinstance(states, list):
                        states = list(states) if states else []
                    reasoning_parts.append(f"- **States found**: {', '.join(states[:5])}")
            reasoning_parts.append("")

        # Part 7: Historical context (repeated issues)
        if previous_analyses and len(previous_analyses) >= 2:
            reasoning_parts.append("### Historical Context:")

            # Check for repeated failures
            if primary_failure:
                repeated_count = 0
                for prev_analysis in previous_analyses:
                    prev_pf = prev_analysis.get('primary_failure', {})
                    if isinstance(prev_pf, dict):
                        if (prev_pf.get('expected_value') == primary_failure.expected_value and
                            prev_pf.get('actual_value') == primary_failure.actual_value):
                            repeated_count += 1

                if repeated_count >= 2:
                    reasoning_parts.append(f"- **REPEATED FAILURE**: This exact issue occurred in {repeated_count} previous iterations")
                    reasoning_parts.append(f"- Previous fixes did not address the root cause - need different approach")

            reasoning_parts.append(f"- This is iteration {iteration}, {len(previous_analyses)} previous attempts")
            reasoning_parts.append("")

        # Part 8: Actionable recommendations
        reasoning_parts.append("### Recommended Actions:")

        # Based on patterns and issues
        if any("stuck at value 0" in str(p) for p in potential_issues):
            reasoning_parts.append("1. Check that output signal is assigned in all code paths")
            reasoning_parts.append("2. Verify always block sensitivity list includes all necessary signals")
            reasoning_parts.append("3. Confirm reset logic initializes signal properly")
        elif any("never assigned" in str(p) for p in potential_issues):
            reasoning_parts.append("1. Add assignment logic for undriven signals")
            reasoning_parts.append("2. Ensure combinational or sequential logic drives all outputs")
        elif test_patterns and any("bit width" in str(h).lower() for h in test_patterns.get('hypotheses', [])):
            reasoning_parts.append("1. Review bit widths of all signals in computation path")
            reasoning_parts.append("2. Check for overflow/underflow conditions")
            reasoning_parts.append("3. Ensure intermediate results fit in declared bit widths")
        elif rtl_analysis.get('state_machine_detected'):
            reasoning_parts.append("1. Verify state machine transitions are complete (all states reachable)")
            reasoning_parts.append("2. Check state transition conditions match specification")
            reasoning_parts.append("3. Ensure output logic considers all states")
        else:
            reasoning_parts.append("1. Review the failing signal's assignment logic")
            reasoning_parts.append("2. Check control flow (if/case statements) covers all cases")
            reasoning_parts.append("3. Verify computation matches specification requirements")

        return "\n".join(reasoning_parts)

    def _decide_context_checking(self, primary_failure: Optional[PrimaryFailure],
                                error_report: str, iteration: int,
                                previous_analyses: List[Dict] = None) -> Tuple[bool, bool, str]:
        """
        Decide when to check testbench/specification.

        More conservative than before - only when really necessary.
        """
        should_check_testbench = False
        should_check_spec = False
        reasoning_parts = []

        # Check testbench if: Very unclear what the error means
        if not primary_failure or "parse error" in error_report.lower():
            should_check_testbench = True
            reasoning_parts.append("Testbench check needed: Error report is unclear or unparseable")

        # Check spec if: Repeated failures after many iterations
        if iteration >= 5 and previous_analyses:
            if len(previous_analyses) >= 4:
                should_check_spec = True
                reasoning_parts.append(f"Specification check needed: {iteration} iterations without success - may have misunderstood requirements")

        # Check spec if: All tests fail
        if "0 passed" in error_report or "all tests failed" in error_report.lower():
            if iteration >= 3:
                should_check_spec = True
                reasoning_parts.append("Specification check needed: All tests failing - fundamental misunderstanding likely")

        return should_check_testbench, should_check_spec, "; ".join(reasoning_parts) if reasoning_parts else "No additional context needed"
