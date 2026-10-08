# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Testbench Analyzer - reads and understands testbench files.
"""

import re
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

@dataclass
class TestCase:
    """Information about a test case."""
    name: str
    description: str
    test_type: str  # "reset", "overflow", "functional", etc.
    expected_behavior: str
    key_assertions: List[str]

@dataclass
class TestbenchInsight:
    """Complete understanding of testbench behavior."""
    test_cases: List[TestCase]
    test_sequence: str
    reset_behavior: str
    key_signals_tested: List[str]
    timing_requirements: Dict[str, Any]
    failure_conditions: List[str]

class TestbenchAnalyzer:
    """
    Analyzes testbench files to understand test intent.

    This provides context about what the test is trying to do,
    not just parsing error reports after it fails.
    """

    def __init__(self, logger):
        self.logger = logger

    def analyze_testbench_files(self, harness_dir: Path) -> Optional[TestbenchInsight]:
        """
        Read and analyze testbench files to understand test intent.

        Args:
            harness_dir: Path to harness directory (e.g., harness/1/)

        Returns:
            TestbenchInsight with complete test understanding
        """
        self.logger.info("Analyzing testbench files for test intent understanding")

        # Find testbench source files
        src_dir = harness_dir / "src"
        if not src_dir.exists():
            self.logger.warning(f"No src directory found in {harness_dir}")
            return None

        # Look for Python testbench files (cocotb)
        test_files = list(src_dir.glob("test_*.py"))
        if not test_files:
            self.logger.warning(f"No test_*.py files found in {src_dir}")
            return None

        # Analyze the main test file
        main_test_file = test_files[0]  # Usually test_<module_name>.py

        try:
            with open(main_test_file, 'r') as f:
                test_content = f.read()

            insight = self._parse_testbench_content(test_content, main_test_file.name)
            self.logger.info(f"Successfully analyzed testbench: {len(insight.test_cases)} test cases found")
            return insight

        except Exception as e:
            self.logger.error(f"Failed to analyze testbench {main_test_file}: {e}")
            return None

    def _parse_testbench_content(self, content: str, filename: str) -> TestbenchInsight:
        """Parse testbench content to extract test understanding."""

        # Extract test cases
        test_cases = self._extract_test_cases(content)

        # Analyze test sequence and behavior
        test_sequence = self._analyze_test_sequence(content)
        reset_behavior = self._analyze_reset_behavior(content)
        key_signals = self._extract_key_signals(content)
        timing_requirements = self._analyze_timing(content)
        failure_conditions = self._extract_failure_conditions(content)

        return TestbenchInsight(
            test_cases=test_cases,
            test_sequence=test_sequence,
            reset_behavior=reset_behavior,
            key_signals_tested=key_signals,
            timing_requirements=timing_requirements,
            failure_conditions=failure_conditions
        )

    def _extract_test_cases(self, content: str) -> List[TestCase]:
        """Extract individual test cases from testbench."""
        test_cases = []

        # Find test functions (cocotb.test() decorators)
        test_pattern = r'@cocotb\.test\(\).*?async def (test_\w+)\(.*?\):\s*"""(.*?)"""'
        matches = re.findall(test_pattern, content, re.DOTALL)

        for test_name, description in matches:
            test_case = TestCase(
                name=test_name,
                description=description.strip(),
                test_type=self._classify_test_type(test_name, description),
                expected_behavior=self._extract_expected_behavior(test_name, content),
                key_assertions=self._extract_assertions_for_test(test_name, content)
            )
            test_cases.append(test_case)

        return test_cases

    def _classify_test_type(self, test_name: str, description: str) -> str:
        """Classify what type of test this is."""
        name_lower = test_name.lower()
        desc_lower = description.lower()

        if "reset" in name_lower or "reset" in desc_lower:
            return "reset_test"
        elif "overflow" in name_lower or "overflow" in desc_lower:
            return "overflow_test"
        elif "functional" in name_lower or "sequence" in desc_lower:
            return "functional_test"
        else:
            return "generic_test"

    def _extract_expected_behavior(self, test_name: str, content: str) -> str:
        """Extract what behavior the test expects."""

        # Find the test function
        test_pattern = rf'async def {test_name}\(.*?\):(.*?)(?=async def|\Z)'
        match = re.search(test_pattern, content, re.DOTALL)

        if match:
            test_body = match.group(1)

            # Look for comments describing expected behavior
            comments = re.findall(r'#\s*(.*)', test_body)
            if comments:
                return '; '.join(comments)

            # Look for assert statements to infer expected behavior
            assertions = re.findall(r'assert.*?, f"(.*?)"', test_body)
            if assertions:
                return f"Expects: {', '.join(assertions)}"

        return "Expected behavior unclear from test code"

    def _extract_assertions_for_test(self, test_name: str, content: str) -> List[str]:
        """Extract key assertions from a specific test."""
        assertions = []

        # Find the test function
        test_pattern = rf'async def {test_name}\(.*?\):(.*?)(?=async def|\Z)'
        match = re.search(test_pattern, content, re.DOTALL)

        if match:
            test_body = match.group(1)

            # Extract assert statements
            assert_patterns = [
                r'assert\s+(.*?),\s*f"(.*?)"',  # assert condition, f"message"
                r'assert\s+(.*?),\s*"(.*?)"',   # assert condition, "message"
                r'assert\s+(.*?)$'              # assert condition
            ]

            for pattern in assert_patterns:
                matches = re.findall(pattern, test_body, re.MULTILINE)
                for match_tuple in matches:
                    if isinstance(match_tuple, tuple):
                        if len(match_tuple) == 2:
                            condition, message = match_tuple
                            assertions.append(f"{condition.strip()} ({message})")
                        else:
                            assertions.append(match_tuple[0].strip())
                    else:
                        assertions.append(match_tuple.strip())

        return assertions

    def _analyze_test_sequence(self, content: str) -> str:
        """Analyze the overall test sequence."""

        # Look for common patterns
        if "apply_reset" in content:
            sequence_parts = ["Reset application"]
        else:
            sequence_parts = []

        if any(seq_keyword in content.lower() for seq_keyword in ["calculate", "sequence", "series", "generate"]):
            sequence_parts.append("Sequence verification")

        if "overflow" in content.lower():
            sequence_parts.append("Overflow detection")

        return " -> ".join(sequence_parts) if sequence_parts else "Sequence unclear"

    def _analyze_reset_behavior(self, content: str) -> str:
        """Analyze how reset is supposed to work."""

        # Look for reset function
        reset_match = re.search(r'async def apply_reset.*?\):(.*?)(?=async def|\Z)', content, re.DOTALL)
        if reset_match:
            reset_body = reset_match.group(1)

            # Extract reset behavior - check for general reset patterns
            if re.search(r"rst\.value\s*=\s*1", reset_body) or re.search(r"reset.*=.*1", reset_body):
                behavior = "Active-high reset"
            elif re.search(r"rst\.value\s*=\s*0", reset_body) or re.search(r"reset.*=.*0", reset_body):
                behavior = "Active-low reset"
            else:
                behavior = "Reset polarity unclear"

            # Look for reset timing
            timer_match = re.search(r'Timer\((\d+)', reset_body)
            if timer_match:
                duration = timer_match.group(1)
                behavior += f", held for {duration}ns"

            return behavior

        return "Reset behavior not clearly defined in testbench"

    def _extract_key_signals(self, content: str) -> List[str]:
        """Extract key signals being tested."""
        signals = set()

        # Look for dut.signal patterns
        signal_patterns = re.findall(r'dut\.(\w+)\.value', content)
        signals.update(signal_patterns)

        # Look for dut.signal in assertions
        assert_signals = re.findall(r'dut\.(\w+)\s*==', content)
        signals.update(assert_signals)

        return list(signals)

    def _analyze_timing(self, content: str) -> Dict[str, Any]:
        """Analyze timing requirements from testbench."""
        timing = {}

        # Look for clock period
        clock_match = re.search(r'Clock\(.*?,\s*(\d+)', content)
        if clock_match:
            timing["clock_period_ns"] = int(clock_match.group(1))

        # Look for timing delays
        timer_matches = re.findall(r'Timer\((\d+)', content)
        if timer_matches:
            timing["delays_used"] = [int(t) for t in timer_matches]

        return timing

    def _extract_failure_conditions(self, content: str) -> List[str]:
        """Extract conditions that would cause test failure."""
        conditions = []

        # Extract assert conditions (these are failure points)
        assertions = re.findall(r'assert\s+(.*?)(?:,|\n)', content)
        for assertion in assertions:
            # Convert assertion to failure condition
            if "==" in assertion:
                # dut_output == a becomes "dut_output != expected"
                parts = assertion.split("==")
                if len(parts) == 2:
                    conditions.append(f"{parts[0].strip()} != {parts[1].strip()}")
            else:
                conditions.append(f"NOT ({assertion.strip()})")

        return conditions

    def get_test_context_for_error(self, insight: TestbenchInsight, error_report: str) -> str:
        """Get relevant test context for a specific error."""

        context_parts = ["## TESTBENCH CONTEXT"]

        # Identify which test failed
        failed_test = None
        for test_case in insight.test_cases:
            if test_case.name in error_report:
                failed_test = test_case
                break

        if failed_test:
            context_parts.append(f"**Failed Test**: {failed_test.name}")
            context_parts.append(f"**Test Purpose**: {failed_test.description}")
            context_parts.append(f"**Test Type**: {failed_test.test_type}")
            context_parts.append(f"**Expected Behavior**: {failed_test.expected_behavior}")

            if failed_test.key_assertions:
                context_parts.append("**Key Assertions**:")
                for assertion in failed_test.key_assertions:
                    context_parts.append(f"  - {assertion}")

        # Add general test info
        context_parts.append(f"**Test Sequence**: {insight.test_sequence}")
        context_parts.append(f"**Reset Behavior**: {insight.reset_behavior}")
        context_parts.append(f"**Signals Tested**: {', '.join(insight.key_signals_tested)}")

        if insight.timing_requirements:
            context_parts.append(f"**Timing**: {insight.timing_requirements}")

        return "\n".join(context_parts)
