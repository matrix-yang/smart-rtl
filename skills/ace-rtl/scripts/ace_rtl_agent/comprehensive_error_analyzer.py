# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive Error Analyzer - Deep error analysis following user guidelines
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from .primary_failure_extractor import PrimaryFailureExtractor, PrimaryFailure
from .utils.llm_client import LLMClient
from .rtl_rule_advisor import RTLRuleAdvisor


@dataclass
class TestCaseResult:
    """Result of a specific test case."""
    test_name: str
    passed: bool
    assertion_details: Optional[str] = None
    timing_info: Optional[str] = None


@dataclass
class ComprehensiveErrorAnalysis:
    """Complete analysis of test results."""
    primary_failure: Optional[PrimaryFailure]
    passing_tests: List[TestCaseResult]
    failing_tests: List[TestCaseResult]
    core_reasoning: str
    should_check_testbench: bool
    should_check_specification: bool
    reasoning_for_context_check: str
    rtl_rule_advice: Optional[str] = None


class ComprehensiveErrorAnalyzer:
    """
    Comprehensive error analysis that follows user guidelines:
    1. Deep reasoning of error reports
    2. Track passing/failing test cases
    3. Focus on primary error only
    4. Decide when to check testbench/specification
    """

    def __init__(self, logger, model_name: str = "nvidia/nemotron-3-ultra-550b-a55b", llm_context_limit: int = 300000):
        self.logger = logger
        self.model_name = model_name
        self.primary_failure_extractor = PrimaryFailureExtractor(logger, model_name, llm_context_limit)
        self.llm_context_limit = llm_context_limit
        self.rtl_rule_advisor = RTLRuleAdvisor()
        # Cache full specification so retrieval can run even if spec isn't passed later
        self.cached_specification: Optional[str] = None

    def analyze_error_comprehensively(self, error_report: str, iteration: int,
                                    previous_analyses: List[Dict] = None,
                                    rtl_code: str = None,
                                    specification: str = None,
                                    is_restart: bool = False) -> ComprehensiveErrorAnalysis:
        """
        Perform comprehensive error analysis following guidelines.

        Args:
            error_report: Full error report from test
            iteration: Current iteration number
            previous_analyses: Previous iteration analyses for learning
            rtl_code: Current RTL implementation (for rule advice)
            specification: Design specification (for better rule advice)

        Returns:
            Comprehensive analysis with all required information
        """
        print(f"DEBUG: === Starting comprehensive error analysis for iteration {iteration} ===")
        try:
            self.logger.info(f"Performing comprehensive error analysis for iteration {iteration}")

            # Step 1: Extract primary failure (focus on ONE issue)
            primary_failure = self.primary_failure_extractor.extract_primary_failure(error_report)
            # Removed duplicate logging - primary_failure_extractor already logs the result
            print(f"DEBUG: ComprehensiveErrorAnalyzer iteration {iteration} - primary_failure result: {primary_failure}")
            if primary_failure:
                print(f"DEBUG: primary_failure type: {type(primary_failure)}, expected: {primary_failure.expected_value}, actual: {primary_failure.actual_value}")
            else:
                print(f"DEBUG: primary_failure is None/False")

            # Step 2: Identify passing and failing test cases
            print(f"DEBUG: Step 2 - Categorizing test results")
            passing_tests, failing_tests = self._categorize_test_results(error_report)
            print(f"DEBUG: Found {len(passing_tests)} passing, {len(failing_tests)} failing tests")

            # Step 3: Generate core reasoning about the specific failure
            print(f"DEBUG: Step 3 - Generating core reasoning")
            try:
                core_reasoning = self._generate_core_reasoning(
                    primary_failure, error_report, iteration, previous_analyses
                )
                print(f"DEBUG: Core reasoning generated successfully")
            except Exception as e:
                print(f"ERROR: Core reasoning generation failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to basic reasoning
                core_reasoning = f"Error during reasoning generation: {e}. Using fallback analysis."

            # Step 4: Intelligent decision about context checking
            print(f"DEBUG: Step 4 - Context checking decisions")
            try:
                should_check_testbench, should_check_spec, context_reasoning = self._decide_context_checking(
                    primary_failure, error_report, iteration, previous_analyses
                )
                print(f"DEBUG: Context checking completed")
            except Exception as e:
                print(f"ERROR: Context checking decisions failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to basic context checking
                should_check_testbench, should_check_spec, context_reasoning = False, False, f"Context checking error: {e}"

            # Update cached specification when provided
            if specification:
                self.cached_specification = specification

            # Step 5: Retrieve architectural guidance at first iteration and on explicit restarts
            rtl_rule_advice = None

            # Retrieve rules at first iteration OR when explicit restart signal is provided
            should_retrieve_rules = (iteration == 1 or is_restart)
            print(f"DEBUG: Rule retrieval check - iteration={iteration}, is_restart={is_restart}, should_retrieve={should_retrieve_rules}")
            if should_retrieve_rules:
                spec_to_use = specification if specification else self.cached_specification
                if spec_to_use:
                    print(f"DEBUG: Step 5 - {'First iteration' if iteration == 1 else ('Explicit restart' if is_restart else 'Fresh start')}, retrieving architectural guidance for RTL generation (using {'provided' if specification else 'cached'} spec)")
                    self.rtl_rule_advisor.retrieve_rules_from_specification(spec_to_use)

            # Always include architectural guidance once retrieved (persist across iterations until reset)
            if self.rtl_rule_advisor.has_relevant_rules():
                advice_list = self.rtl_rule_advisor.get_retrieved_rules()
                rtl_rule_advice = self.rtl_rule_advisor.format_advice_for_agent(advice_list)
                if should_retrieve_rules:
                    guidance_context = "initial" if iteration == 1 else ("explicit restart" if is_restart else "fresh start")
                else:
                    guidance_context = "persistent"
                print(f"DEBUG: Providing {len(advice_list)} architectural guidance rules for {guidance_context} RTL generation")
            elif should_retrieve_rules:
                guidance_context = "initial" if iteration == 1 else ("explicit restart" if is_restart else "fresh start")
                print(f"DEBUG: No architectural guidance retrieved for this {guidance_context} design")

            result = ComprehensiveErrorAnalysis(
                primary_failure=primary_failure,
                passing_tests=passing_tests,
                failing_tests=failing_tests,
                core_reasoning=core_reasoning,
                should_check_testbench=should_check_testbench,
                should_check_specification=should_check_spec,
                reasoning_for_context_check=context_reasoning,
                rtl_rule_advice=rtl_rule_advice
            )

            print(f"DEBUG: === ComprehensiveErrorAnalysis completed successfully for iteration {iteration} ===")
            return result

        except Exception as e:
            print(f"DEBUG: === EXCEPTION in comprehensive error analysis for iteration {iteration} ===")
            print(f"DEBUG: Exception type: {type(e)}")
            print(f"DEBUG: Exception message: {e}")
            import traceback
            traceback.print_exc()
            print(f"DEBUG: === End exception details ===")
            raise

    def _categorize_test_results(self, error_report: str) -> Tuple[List[TestCaseResult], List[TestCaseResult]]:
        """Extract passing and failing test cases from error report - handles multiple formats."""
        passing_tests = []
        failing_tests = []

        # Pattern 1: Cocotb individual test results table
        # Example: ** tb.test_empty                  PASS         664.50           0.01      63082.59  **
        cocotb_individual_results = re.findall(
            r'\*\*\s+([\w\.]+)\s+(PASS|FAIL)\s+[\d\.]+\s+',
            error_report, re.IGNORECASE
        )

        if cocotb_individual_results:
            self.logger.info(f"Found cocotb individual test results with {len(cocotb_individual_results)} tests")
            for test_name, status in cocotb_individual_results:
                if status.upper() == 'PASS':
                    # Extract context for passing tests to understand what worked
                    pass_context = self._extract_test_context(error_report, test_name)
                    passing_tests.append(TestCaseResult(test_name, True, pass_context))
                elif status.upper() == 'FAIL':
                    # Extract detailed failure context
                    assertion_details = self._extract_assertion_details(error_report, test_name)
                    failing_tests.append(TestCaseResult(test_name, False, assertion_details))

        # Pattern 2: Cocotb summary results (as fallback)
        # Example: ** TESTS=4 PASS=3 FAIL=1 SKIP=0 **
        if not cocotb_individual_results:
            summary_match = re.search(r'\*\*\s*TESTS=(\d+)\s+PASS=(\d+)\s+FAIL=(\d+)', error_report)
            if summary_match:
                total_tests, pass_count, fail_count = map(int, summary_match.groups())
                self.logger.info(f"Found cocotb summary: {pass_count} pass, {fail_count} fail out of {total_tests}")

                # Extract test names from running logs
                test_runs = re.findall(r'running\s+([\w\.]+)\s+\(\d+/\d+\)', error_report)
                test_results = re.findall(r'([\w\.]+)\s+(passed|failed)', error_report)

                # Match test names with results
                for test_name, result in test_results:
                    if result == 'passed':
                        pass_context = self._extract_test_context(error_report, test_name)
                        passing_tests.append(TestCaseResult(test_name, True, pass_context))
                    elif result == 'failed':
                        assertion_details = self._extract_assertion_details(error_report, test_name)
                        failing_tests.append(TestCaseResult(test_name, False, assertion_details))

                # If we don't have enough specific results, create generic ones
                if len(passing_tests) + len(failing_tests) < total_tests:
                    remaining_pass = pass_count - len(passing_tests)
                    remaining_fail = fail_count - len(failing_tests)

                    for i in range(remaining_pass):
                        passing_tests.append(TestCaseResult(f"test_{i+1+len(passing_tests)}", True, "Test functionality validated"))
                    for i in range(remaining_fail):
                        failing_tests.append(TestCaseResult(f"test_{i+1+len(failing_tests)}", False, "Test execution failed"))

        # Pattern 3: Simple pytest format
        # Example: "test_runner PASSED [100%]" or "5 passed, 2 warnings in 11.19s"
        if not passing_tests and not failing_tests:
            pytest_summary = re.search(r'(\d+)\s+(passed|failed)', error_report)
            if pytest_summary:
                count, result = pytest_summary.groups()
                count = int(count)
                self.logger.info(f"Found pytest summary: {count} {result}")

                if result == 'passed':
                    for i in range(count):
                        passing_tests.append(TestCaseResult(f"test_{i+1}", True, "Test functionality validated"))
                elif result == 'failed':
                    for i in range(count):
                        assertion_details = self._extract_generic_assertion_details(error_report)
                        failing_tests.append(TestCaseResult(f"test_{i+1}", False, assertion_details))

            # Single test pass/fail
            elif "PASSED" in error_report:
                passing_tests.append(TestCaseResult("test_case", True, "Test functionality validated"))
            elif "FAILED" in error_report:
                assertion_details = self._extract_generic_assertion_details(error_report)
                failing_tests.append(TestCaseResult("test_case", False, assertion_details))

        # Pattern 4: Multiple parametrized test failures
        # Look for repeated test patterns with the same assertion failure
        if not passing_tests and not failing_tests:
            parametrized_failures = re.findall(r'FAILED.*?test_\w+\[(\d+)\].*?SystemExit: ERROR', error_report, re.DOTALL)
            if parametrized_failures:
                self.logger.info(f"Found {len(parametrized_failures)} parametrized test failures")
                assertion_details = self._extract_generic_assertion_details(error_report)
                for param in parametrized_failures:
                    failing_tests.append(TestCaseResult(f"parametrized_test[{param}]", False, assertion_details))

        # Pattern 5: Compilation/Environment failures (before generic fallback)
        if not passing_tests and not failing_tests:
            compilation_failure = self._detect_compilation_failure(error_report)
            if compilation_failure:
                failing_tests.append(TestCaseResult("compilation_error", False, compilation_failure))
            else:
                # Pattern 6: Generic assertion failures (final fallback)
                self.logger.info("Using generic assertion parsing as fallback")
                assertion_failures = re.findall(r'(AssertionError.*?)(?=\n|$)', error_report)
                if assertion_failures:
                    for i, assertion in enumerate(assertion_failures):
                        failing_tests.append(TestCaseResult(f"assertion_failure_{i+1}", False, assertion))
                else:
                    # Pattern 7: LLM-based intelligent analysis (final fallback)
                    self.logger.info("Using LLM-based intelligent analysis as final fallback")
                    llm_analysis = self._llm_analyze_error_report(error_report)
                    if llm_analysis:
                        passing_tests.extend(llm_analysis.get('passing_tests', []))
                        failing_tests.extend(llm_analysis.get('failing_tests', []))

        # Print clean statistics
        self._print_test_statistics(passing_tests, failing_tests, error_report)

        self.logger.info(f"Categorized tests: {len(passing_tests)} passing, {len(failing_tests)} failing")
        return passing_tests, failing_tests

    def _extract_assertion_details(self, error_report: str, test_name: str) -> Optional[str]:
        """Extract detailed assertion context for a specific failing test."""
        # Look for assertion details related to this specific test
        test_section_pattern = rf'running {re.escape(test_name)}.*?(?=running \w+\.|=+|$)'
        test_section_match = re.search(test_section_pattern, error_report, re.DOTALL)

        context_parts = []

        if test_section_match:
            test_section = test_section_match.group(0)

            # Get test context first
            test_context = self._extract_test_context(error_report, test_name)
            if test_context and test_context != "Test functionality validated":
                context_parts.append(f"Context: {test_context}")

            # Extract the assertion error
            assertion_match = re.search(r'(AssertionError.*?)(?=\n|\+)', test_section, re.DOTALL)
            if assertion_match:
                assertion_details = assertion_match.group(1).strip()
                context_parts.append(f"Error: {assertion_details}")

            # Look for expected vs actual in the test section
            expected_actual = re.search(r'Expected:?\s*([^,\n]+).*(?:got|actual|received):?\s*([^,\n]+)',
                                       test_section, re.IGNORECASE)
            if expected_actual and not any("Expected" in part for part in context_parts):
                expected, actual = expected_actual.groups()
                context_parts.append(f"Expected {expected.strip()}, got {actual.strip()}")

        if context_parts:
            return " | ".join(context_parts)

        # Fallback: look for any assertion error near the test name
        lines = error_report.split('\n')
        for i, line in enumerate(lines):
            if test_name in line and 'failed' in line:
                # Look for AssertionError in nearby lines
                for j in range(max(0, i-5), min(len(lines), i+10)):
                    if 'AssertionError' in lines[j]:
                        return f"Error: {lines[j].strip()}"

        return "Test execution failed"

    def _extract_generic_assertion_details(self, error_report: str) -> str:
        """Extract the first/main assertion error from the report."""
        # Look for AssertionError with details
        assertion_patterns = [
            r'AssertionError:\s*(.*?)(?=\n|\+|$)',  # AssertionError: message
            r'assert\s+([^,]+),\s*f?"([^"]*)"',      # assert condition, "message"
            r'ERROR:\s*(.*?)(?=\n|$)',               # ERROR: message
            r'FAIL:\s*(.*?)(?=\n|$)',                # FAIL: message
        ]

        for pattern in assertion_patterns:
            match = re.search(pattern, error_report, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # Look for expected vs actual patterns
        expected_actual = re.search(r'Expected:?\s*([^,\n]+).*(?:got|actual):?\s*([^,\n]+)',
                                   error_report, re.IGNORECASE)
        if expected_actual:
            expected, actual = expected_actual.groups()
            return f"Expected {expected.strip()}, got {actual.strip()}"

        return "Test execution failed"

    def _extract_test_context(self, error_report: str, test_name: str) -> str:
        """Extract rich context for what a test validated (for both pass and fail)."""
        # Look for the test execution section
        test_section_pattern = rf'running {re.escape(test_name)}.*?(?=running \w+\.|INFO.*regression.*\w+\s+(passed|failed)|=+|$)'
        test_section_match = re.search(test_section_pattern, error_report, re.DOTALL)

        context_parts = []

        if test_section_match:
            test_section = test_section_match.group(0)

            # Extract meaningful operation logs
            operation_patterns = [
                r'INFO.*?([Tt]est \w+[^.]*(?:Started|began|initiated).*?)(?=\n)',  # Test descriptions
                r'INFO.*?((?:Write|Read|Wrote|Successfully|Finished|Completed|Testing|Validating).*?)(?=\n)', # Operations
                r'INFO.*?(executing \d+.*?)(?=\n)',  # Test procedures
                r'INFO.*?(concurrent|parallel|simultaneous.*?)(?=\n)',  # Test types
                r'INFO.*?(checking|verifying|testing|validating.*?)(?=\n)',  # Validation activities
            ]

            for pattern in operation_patterns:
                matches = re.findall(pattern, test_section, re.IGNORECASE)
                for match in matches[:3]:  # Limit to most relevant operations
                    clean_match = match.strip().rstrip('.')
                    if len(clean_match) > 10 and clean_match not in context_parts:  # Avoid duplicates and noise
                        context_parts.append(clean_match)

        # Fallback: extract from test name and common patterns
        if not context_parts:
            name_context = self._infer_context_from_test_name(test_name)
            if name_context:
                context_parts.append(name_context)

        # Look for expected behavior patterns in the broader section
        if test_section_match:
            expected_patterns = [
                r'Expected.*?(?=\n)',  # Expected results
                r'should.*?(?=\n)',    # Should behaviors
                r'assert.*?(?=,)',     # Assertion conditions
            ]

            for pattern in expected_patterns:
                matches = re.findall(pattern, test_section, re.IGNORECASE)
                for match in matches[:2]:
                    clean_match = match.strip().rstrip(',.')
                    if len(clean_match) > 5 and clean_match not in context_parts:
                        context_parts.append(clean_match)

        if context_parts:
            return " | ".join(context_parts[:4])  # Limit to most important context

        return "Test functionality validated"

    def _infer_context_from_test_name(self, test_name: str) -> str:
        """Infer test purpose from naming patterns."""
        name_lower = test_name.lower()

        # Common test name patterns
        context_mappings = {
            'empty': 'FIFO empty condition handling',
            'basic': 'Basic read/write operations',
            'full': 'FIFO full condition handling',
            'random': 'Random/concurrent operations',
            'stress': 'Stress testing under load',
            'edge': 'Edge case validation',
            'overflow': 'Overflow condition handling',
            'underflow': 'Underflow condition handling',
            'reset': 'Reset functionality',
            'clock': 'Clock domain behavior',
            'async': 'Asynchronous operation',
            'sync': 'Synchronous operation',
            'write': 'Write operation validation',
            'read': 'Read operation validation',
            'register': 'Register access validation',
            'memory': 'Memory functionality',
            'fifo': 'FIFO behavior validation',
            'axi': 'AXI protocol compliance',
            'apb': 'APB protocol compliance'
        }

        for keyword, context in context_mappings.items():
            if keyword in name_lower:
                return context

        return None

    def _detect_compilation_failure(self, error_report: str) -> Optional[str]:
        """Detect compilation failures, Docker errors, and build issues."""

        # Pattern 1: RTL Compilation/Syntax Errors
        compilation_patterns = [
            # Verilog/SystemVerilog syntax errors
            r'Error.*syntax error.*line (\d+)',
            r'Error.*Parse error.*line (\d+)',
            r'vlog.*Error.*\((\d+)\)',
            r'ERROR:\s*(.*(?:syntax|parse|compilation).*?)(?=\n|$)',
            # Missing modules/files
            r'Error.*Cannot find module.*?[\'"]([^\'"]+)[\'"]',
            r'Error.*file not found.*?[\'"]([^\'"]+)[\'"]',
            r'Fatal.*Cannot open.*?[\'"]([^\'"]+)[\'"]',
        ]

        for pattern in compilation_patterns:
            match = re.search(pattern, error_report, re.IGNORECASE)
            if match:
                if 'line' in pattern:
                    line_num = match.group(1) if match.groups() else 'unknown'
                    return f"RTL Compilation Error: Syntax error at line {line_num}"
                elif 'module' in pattern or 'file' in pattern:
                    file_name = match.group(1) if match.groups() else 'unknown'
                    return f"RTL Compilation Error: Cannot find module/file '{file_name}'"
                else:
                    return f"RTL Compilation Error: {match.group(1) if match.groups() else match.group(0)}"

        # Pattern 2: Docker Environment Errors
        docker_patterns = [
            r'docker.*error.*?(?:failed|cannot|unable).*?(?=\n|$)',
            r'Container.*exited.*code (\d+)',
            r'docker-compose.*error.*?(?=\n|$)',
            r'ERROR.*docker.*?(?=\n|$)',
            r'Cannot connect to.*Docker.*daemon',
            r'docker.*permission denied',
            r'Error response from daemon.*?(?=\n|$)',
        ]

        for pattern in docker_patterns:
            match = re.search(pattern, error_report, re.IGNORECASE)
            if match:
                if 'exited' in match.group(0):
                    exit_code = match.group(1) if match.groups() else 'unknown'
                    return f"Docker Environment Error: Container exited with code {exit_code}"
                else:
                    return f"Docker Environment Error: {match.group(0).strip()}"

        # Pattern 3: Build System Failures
        build_patterns = [
            r'make.*error.*?(?=\n|$)',
            r'Error.*Makefile.*?(?=\n|$)',
            r'fatal error.*?(?=\n|$)',
            r'ERROR.*build.*?(?=\n|$)',
            r'No rule to make target.*?(?=\n|$)',
            r'Permission denied.*?(?=\n|$)',
            r'command not found.*?(?=\n|$)',
        ]

        for pattern in build_patterns:
            match = re.search(pattern, error_report, re.IGNORECASE)
            if match:
                return f"Build System Error: {match.group(0).strip()}"

        # Pattern 4: Tool Chain Issues
        toolchain_patterns = [
            r'License.*(?:error|expired|not found).*?(?=\n|$)',
            r'ERROR.*(?:iverilog|modelsim|questasim|vcs).*?(?=\n|$)',
            r'command.*(?:iverilog|vlog|vsim).*not found',
            r'FATAL.*tool.*?(?=\n|$)',
            r'Error.*simulator.*?(?=\n|$)',
        ]

        for pattern in toolchain_patterns:
            match = re.search(pattern, error_report, re.IGNORECASE)
            if match:
                return f"Tool Chain Error: {match.group(0).strip()}"

        # Pattern 5: Python/Cocotb Environment Issues
        python_patterns = [
            r'ModuleNotFoundError.*?(?=\n|$)',
            r'ImportError.*?(?=\n|$)',
            r'Python.*ERROR.*?(?=\n|$)',
            r'cocotb.*ERROR.*?(?=\n|$)',
            r'No module named.*?(?=\n|$)',
        ]

        for pattern in python_patterns:
            match = re.search(pattern, error_report, re.IGNORECASE)
            if match:
                return f"Python Environment Error: {match.group(0).strip()}"

        # Pattern 6: Early termination without test execution
        if self._is_early_termination(error_report):
            return "Build/Environment Error: Process terminated before test execution"

        return None

    def _is_early_termination(self, error_report: str) -> bool:
        """Check if process terminated early without reaching test execution."""
        # Look for signs that suggest early termination
        early_termination_indicators = [
            # No test collection/execution signs
            not re.search(r'collecting.*collected.*items?', error_report, re.IGNORECASE),
            not re.search(r'running.*test|test.*session.*starts', error_report, re.IGNORECASE),
            # But has error indicators
            bool(re.search(r'ERROR|FATAL|failed.*build|compilation.*failed', error_report, re.IGNORECASE)),
            # Short report (likely terminated early)
            len(error_report.split('\n')) < 20 and 'ERROR' in error_report.upper()
        ]

        # If most indicators suggest early termination
        return sum(early_termination_indicators) >= 2

    def _print_test_statistics(self, passing_tests: List[TestCaseResult],
                             failing_tests: List[TestCaseResult], error_report: str):
        """Print clean and concise test statistics."""
        total_tests = len(passing_tests) + len(failing_tests)

        if total_tests == 0:
            # Check if this is a non-test error
            if any(keyword in error_report.upper() for keyword in
                  ['COMPILATION', 'SYNTAX ERROR', 'DOCKER', 'BUILD', 'FATAL']):
                self.logger.info("Error Type: Build/Environment issue (no tests executed)")
            else:
                self.logger.info("Error Type: Unknown issue (no tests detected)")
        else:
            pass_rate = (len(passing_tests) / total_tests * 100) if total_tests > 0 else 0
            self.logger.info(f"Test Statistics: {total_tests} total | "
                           f"{len(passing_tests)} pass | {len(failing_tests)} fail | "
                           f"{pass_rate:.1f}% pass rate")

            # Log sample of failing tests if any
            if failing_tests:
                sample_failures = failing_tests[:2]  # Show first 2 failures
                for test in sample_failures:
                    short_details = (test.assertion_details[:60] + "...") if len(test.assertion_details) > 60 else test.assertion_details
                    self.logger.info(f"  {test.test_name}: {short_details}")
                if len(failing_tests) > 2:
                    self.logger.info(f"  ... and {len(failing_tests)-2} more failures")

    def _llm_analyze_error_report(self, error_report: str) -> Optional[Dict[str, Any]]:
        """Use LLM to intelligently analyze error reports that regex patterns couldn't parse."""
        try:
            # For long reports, take top and bottom sections to capture key info
            if len(error_report) > self.llm_context_limit:
                half_limit = self.llm_context_limit // 2
                top_section = error_report[:half_limit]
                bottom_section = error_report[-half_limit:]
                truncated_report = f"{top_section}\n\n[... middle section truncated ...]\n\n{bottom_section}"
                self.logger.info(f"Long error report ({len(error_report)} chars) - analyzing top {half_limit} and bottom {half_limit} chars")
            else:
                truncated_report = error_report

            prompt = f"""You are an expert at analyzing RTL testbench error reports. Analyze this error report and determine:

1. Is this a normal testbench error (test failures) or a different type of error?
2. If testbench error: Extract passing and failing test information
3. If other error: Identify the specific error type

NOTE: For long reports, you may see "[... middle section truncated ...]" - focus on the top (setup/compilation) and bottom (test results/summary) sections which contain the most critical information.

Error Report:
```
{truncated_report}
```

Respond in this JSON format:
{{
  "error_type": "testbench" | "compilation" | "docker" | "environment" | "other",
  "error_message": "specific error description (if not testbench)",
  "passing_tests": [
    {{"test_name": "test_name", "context": "what it validated"}}
  ],
  "failing_tests": [
    {{"test_name": "test_name", "error_details": "failure context"}}
  ],
  "analysis": "brief explanation of findings"
}}

Only include passing_tests and failing_tests if error_type is "testbench"."""

            client = LLMClient(model=self.model_name, logger=self.logger)
            response = client.call_llm(prompt, max_tokens=1000)

            if response and response.strip():
                try:
                    analysis = json.loads(response.strip())

                    # Convert to TestCaseResult objects
                    result = {
                        'passing_tests': [],
                        'failing_tests': [],
                        'error_type': analysis.get('error_type', 'other'),
                        'error_message': analysis.get('error_message', ''),
                        'analysis': analysis.get('analysis', '')
                    }

                    if analysis.get('error_type') == 'testbench':
                        # Parse normal testbench results
                        for test_data in analysis.get('passing_tests', []):
                            result['passing_tests'].append(
                                TestCaseResult(test_data['test_name'], True, test_data.get('context', 'Test passed'))
                            )

                        for test_data in analysis.get('failing_tests', []):
                            result['failing_tests'].append(
                                TestCaseResult(test_data['test_name'], False, test_data.get('error_details', 'Test failed'))
                            )

                        self.logger.info(f"LLM Analysis: Found testbench with {len(result['passing_tests'])} pass, {len(result['failing_tests'])} fail")
                    else:
                        # Non-testbench error - create error entry
                        error_type = analysis.get('error_type', 'unknown')
                        error_msg = analysis.get('error_message', 'Unknown error')
                        result['failing_tests'].append(
                            TestCaseResult(f"{error_type}_error", False, error_msg)
                        )
                        self.logger.info(f"LLM Analysis: Detected {error_type} error - {error_msg[:100]}")

                    return result

                except json.JSONDecodeError as e:
                    self.logger.warning(f"LLM returned invalid JSON: {e}")
                    # Fallback: treat as generic error
                    return {
                        'passing_tests': [],
                        'failing_tests': [TestCaseResult("llm_analysis_error", False, "LLM analysis failed - invalid response format")],
                        'error_type': 'other'
                    }

            return None

        except Exception as e:
            self.logger.warning(f"LLM analysis failed: {e}")
            # Create a generic error entry as absolute fallback
            return {
                'passing_tests': [],
                'failing_tests': [TestCaseResult("analysis_error", False, f"Error analysis failed: {str(e)}")],
                'error_type': 'other'
            }

    def _generate_core_reasoning(self, primary_failure: Optional[PrimaryFailure],
                               error_report: str, iteration: int,
                               previous_analyses: List[Dict] = None) -> str:
        """Generate core reasoning about the specific failure case."""

        if not primary_failure:
            return "No clear primary failure identified. Need to analyze error report structure."

        # Build reasoning about this specific failure
        reasoning_parts = []

        # Basic failure analysis
        reasoning_parts.append(f"PRIMARY FAILURE ANALYSIS:")
        reasoning_parts.append(f"- Expected: {primary_failure.expected_value}")
        reasoning_parts.append(f"- Actual: {primary_failure.actual_value}")
        reasoning_parts.append(f"- Context: {primary_failure.timing_context}")

        # Analyze what this failure means
        if primary_failure.expected_value != primary_failure.actual_value:
            if primary_failure.actual_value == "0" and primary_failure.expected_value != "0":
                reasoning_parts.append("- Issue Type: Signal not being driven correctly (reads as 0)")
                reasoning_parts.append("- Likely Causes: Logic not implemented, clock/reset issues, wrong signal assignment")
            elif "mismatch" in primary_failure.raw_assertion.lower():
                reasoning_parts.append("- Issue Type: Value mismatch in register/signal")
                reasoning_parts.append("- Likely Causes: Write logic not working, read logic incorrect, timing issue")
            else:
                reasoning_parts.append(f"- Issue Type: Unexpected value change")
                reasoning_parts.append(f"- Need to trace signal path from source to destination")

        # Check for patterns from previous iterations
        if previous_analyses:
            print(f"DEBUG: Checking {len(previous_analyses)} previous analyses for repeated failures")
            for i, a in enumerate(previous_analyses):
                print(f"DEBUG: Previous analysis {i}: primary_failure = {a.get('primary_failure')}")
                print(f"DEBUG: Previous analysis {i}: type = {type(a.get('primary_failure'))}")

            print(f"DEBUG: About to compare with current primary_failure.expected_value = {primary_failure.expected_value}")
            repeated_failures = []
            for i, a in enumerate(previous_analyses):
                try:
                    print(f"DEBUG: Processing previous analysis {i}")
                    pf_dict = a.get('primary_failure', {})
                    print(f"DEBUG: pf_dict = {pf_dict}, type = {type(pf_dict)}")
                    if pf_dict is None:
                        print(f"DEBUG: pf_dict is None, skipping")
                        continue
                    expected_val = pf_dict.get('expected_value')
                    print(f"DEBUG: expected_val = {expected_val}")
                    if expected_val == primary_failure.expected_value:
                        repeated_failures.append(a)
                        print(f"DEBUG: Found matching failure: {expected_val}")
                except Exception as e:
                    print(f"DEBUG: Exception in repeated failure check for analysis {i}: {e}")
                    print(f"DEBUG: Exception type: {type(e)}")
                    import traceback
                    traceback.print_exc()

            if len(repeated_failures) >= 2:
                reasoning_parts.append(f"- REPEATED FAILURE: This exact issue occurred {len(repeated_failures)} times before")
                reasoning_parts.append("- Previous fixes didn't address root cause")

        return "\n".join(reasoning_parts)

    def _decide_context_checking(self, primary_failure: Optional[PrimaryFailure],
                               error_report: str, iteration: int,
                               previous_analyses: List[Dict] = None) -> Tuple[bool, bool, str]:
        """
        Intelligently decide when to check testbench/specification.

        Returns:
            (should_check_testbench, should_check_specification, reasoning)
        """
        should_check_testbench = False
        should_check_spec = False
        reasoning_parts = []

        # Check testbench when:
        # 1. Unclear what test is trying to do
        # 2. Multiple assertion failures
        # 3. Timing-related issues
        if not primary_failure:
            should_check_testbench = True
            reasoning_parts.append("No clear primary failure - need to understand test structure")

        elif "timing" in error_report.lower() or "clock" in error_report.lower():
            should_check_testbench = True
            reasoning_parts.append("Timing-related failure - need to understand test sequence")

        elif iteration >= 5 and previous_analyses and primary_failure:
            # Check if we're stuck on the same issue
            print(f"DEBUG: _decide_context_checking - checking recent failures for iteration {iteration}")
            recent_failures = []
            for i, a in enumerate(previous_analyses[-3:]):
                try:
                    print(f"DEBUG: Recent analysis {i}: primary_failure = {a.get('primary_failure')}")
                    pf_dict = a.get('primary_failure')
                    if pf_dict and pf_dict.get('expected_value') == primary_failure.expected_value:
                        recent_failures.append(a)
                        print(f"DEBUG: Found recent matching failure")
                except Exception as e:
                    print(f"DEBUG: Exception in recent failure check: {e}")

            if len(recent_failures) >= 2:
                should_check_testbench = True
                reasoning_parts.append("Repeated failure - need deeper test understanding")

        # Check specification when:
        # 1. Implementation seems fundamentally wrong
        # 2. Multiple different failures across iterations
        # 3. Late iterations with no progress
        if iteration >= 8:
            should_check_spec = True
            reasoning_parts.append("Late iteration - verify specification understanding")

        elif previous_analyses and len(previous_analyses) >= 3:
            # Check if we're seeing different failure types (suggests spec misunderstanding)
            print(f"DEBUG: _decide_context_checking - checking failure types for spec understanding")
            failure_types = []
            for i, a in enumerate(previous_analyses[-3:]):
                try:
                    print(f"DEBUG: Failure type analysis {i}: primary_failure = {a.get('primary_failure')}")
                    pf_dict = a.get('primary_failure')
                    if pf_dict:
                        raw_assertion = pf_dict.get('raw_assertion', '')
                        failure_types.append(raw_assertion)
                        print(f"DEBUG: Added failure type: {raw_assertion[:50]}...")
                    else:
                        failure_types.append('')
                        print(f"DEBUG: Added empty failure type (pf_dict is None)")
                except Exception as e:
                    print(f"DEBUG: Exception in failure type check: {e}")
                    failure_types.append('')

            unique_failures = len(set(failure_types))
            print(f"DEBUG: unique_failures count: {unique_failures}")
            if unique_failures >= 2:
                should_check_spec = True
                reasoning_parts.append("Multiple different failures - may have spec misunderstanding")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Standard debugging approach sufficient"

        return should_check_testbench, should_check_spec, reasoning
