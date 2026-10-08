# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regression Prevention System - Prevents fixes from being lost in subsequent iterations
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class TestPhase(Enum):
    """Different phases of testing that RTL must pass."""
    COMPILATION = "compilation"
    RESET_BEHAVIOR = "reset_behavior"
    BASIC_FUNCTIONALITY = "basic_functionality"
    EDGE_CASES = "edge_cases"
    TIMING_REQUIREMENTS = "timing_requirements"


@dataclass
class PassingAssertion:
    """Represents a test assertion that was passing."""
    assertion_text: str
    test_phase: TestPhase
    iteration_first_passed: int
    expected_value: str
    signal_name: str
    timing_context: str = ""


@dataclass
class RegressionReport:
    """Report of regressions detected."""
    newly_failing_assertions: List[PassingAssertion]
    maintained_passing_assertions: List[PassingAssertion]
    regression_severity: str  # HIGH, MEDIUM, LOW
    recommendation: str


class RegressionPreventionSystem:
    """
    Tracks which test assertions have passed in previous iterations and
    prevents regressions by ensuring fixes are cumulative, not replacements.

    This addresses the core issue where agents fix one problem but break
    previously working functionality.
    """

    def __init__(self, logger):
        self.logger = logger
        self.passing_assertions: Dict[str, PassingAssertion] = {}
        self.test_phase_progress: Dict[TestPhase, bool] = {
            phase: False for phase in TestPhase
        }

    def analyze_test_results(self, error_report: str, iteration: int) -> Tuple[List[str], List[str]]:
        """
        Analyze test results to identify passing and failing assertions.

        Returns:
            Tuple of (newly_passing_assertions, still_failing_assertions)
        """
        newly_passing = []
        still_failing = []

        # Extract all assertions from the error report
        assertions = self._extract_all_assertions(error_report)

        for assertion in assertions:
            assertion_key = self._create_assertion_key(assertion)

            if assertion['status'] == 'PASS':
                if assertion_key not in self.passing_assertions:
                    # New assertion is now passing
                    passing_assertion = PassingAssertion(
                        assertion_text=assertion['text'],
                        test_phase=self._classify_test_phase(assertion['text']),
                        iteration_first_passed=iteration,
                        expected_value=assertion.get('expected', ''),
                        signal_name=assertion.get('signal', ''),
                        timing_context=assertion.get('timing', '')
                    )
                    self.passing_assertions[assertion_key] = passing_assertion
                    newly_passing.append(assertion['text'])
                    self.logger.info(f"New passing assertion: {assertion['text']}")
            else:
                still_failing.append(assertion['text'])

        return newly_passing, still_failing

    def check_for_regressions(self, error_report: str, iteration: int) -> RegressionReport:
        """
        Check if any previously passing assertions are now failing.

        This is the core regression detection logic.
        """
        current_failing = self._extract_failing_assertions(error_report)

        newly_failing = []
        maintained_passing = []

        for key, passing_assertion in self.passing_assertions.items():
            assertion_is_failing = any(
                self._assertion_matches(passing_assertion.assertion_text, failing)
                for failing in current_failing
            )

            if assertion_is_failing:
                newly_failing.append(passing_assertion)
                self.logger.warning(f"REGRESSION DETECTED: {passing_assertion.assertion_text} "
                                  f"was passing since iteration {passing_assertion.iteration_first_passed}")
            else:
                maintained_passing.append(passing_assertion)

        severity = self._assess_regression_severity(newly_failing)
        recommendation = self._generate_regression_recommendation(newly_failing, maintained_passing)

        return RegressionReport(
            newly_failing_assertions=newly_failing,
            maintained_passing_assertions=maintained_passing,
            regression_severity=severity,
            recommendation=recommendation
        )

    def generate_cumulative_requirements(self) -> str:
        """
        Generate a requirements checklist based on all passing assertions.

        This helps the LLM maintain all previously working functionality.
        """
        if not self.passing_assertions:
            return "No passing assertions to maintain yet."

        requirements = ["## CUMULATIVE REQUIREMENTS CHECKLIST", ""]
        requirements.append("The following assertions MUST continue to pass (regression prevention):")
        requirements.append("")

        # Group by test phase for better organization
        by_phase = {}
        for assertion in self.passing_assertions.values():
            phase = assertion.test_phase
            if phase not in by_phase:
                by_phase[phase] = []
            by_phase[phase].append(assertion)

        for phase, assertions in by_phase.items():
            requirements.append(f"### {phase.value.title().replace('_', ' ')} Requirements:")
            for assertion in assertions:
                requirements.append(f"- {assertion.assertion_text}")
                if assertion.expected_value:
                    requirements.append(f"  - Expected: {assertion.expected_value}")
                if assertion.timing_context:
                    requirements.append(f"  - Timing: {assertion.timing_context}")
            requirements.append("")

        requirements.append("**CRITICAL**: Any changes must maintain ALL the above requirements.")
        requirements.append("If a fix would break any existing requirement, find an alternative approach.")

        return "\n".join(requirements)

    def _extract_all_assertions(self, error_report: str) -> List[Dict[str, Any]]:
        """Extract all assertions from error report, both passing and failing."""
        assertions = []

        # Look for successful test messages
        success_patterns = [
            r"reset succesfull",  # Common success message
            r"Testing completed succesfully",
            r"PASS.*",
            r"\u2705.*",
        ]

        for pattern in success_patterns:
            matches = re.finditer(pattern, error_report, re.IGNORECASE)
            for match in matches:
                assertions.append({
                    'text': match.group(0).strip(),
                    'status': 'PASS',
                    'raw_match': match
                })

        # Look for assertion failures
        failure_patterns = [
            r"AssertionError.*",
            r"assert.*failed",
            r"FAIL.*",
            r"\u274c.*",
        ]

        for pattern in failure_patterns:
            matches = re.finditer(pattern, error_report, re.IGNORECASE)
            for match in matches:
                # Extract expected/actual values if available
                expected = self._extract_expected_value(match.group(0))
                actual = self._extract_actual_value(match.group(0))
                signal = self._extract_signal_name(match.group(0))

                assertions.append({
                    'text': match.group(0).strip(),
                    'status': 'FAIL',
                    'expected': expected,
                    'actual': actual,
                    'signal': signal,
                    'raw_match': match
                })

        return assertions

    def _extract_failing_assertions(self, error_report: str) -> List[str]:
        """Extract just the failing assertion texts."""
        assertions = self._extract_all_assertions(error_report)
        return [a['text'] for a in assertions if a['status'] == 'FAIL']

    def _create_assertion_key(self, assertion: Dict[str, Any]) -> str:
        """Create a unique key for an assertion."""
        # Use the core assertion text, normalized
        text = assertion['text']
        # Remove timing-specific details that might change
        text = re.sub(r'\d+\.?\d*ns', 'XXXns', text)
        text = re.sub(r'iteration \d+', 'iteration X', text)
        return text.lower().strip()

    def _classify_test_phase(self, assertion_text: str) -> TestPhase:
        """Classify which test phase an assertion belongs to."""
        text_lower = assertion_text.lower()

        if any(keyword in text_lower for keyword in ['reset', 'initialize']):
            return TestPhase.RESET_BEHAVIOR
        elif any(keyword in text_lower for keyword in ['compile', 'syntax', 'parse']):
            return TestPhase.COMPILATION
        elif any(keyword in text_lower for keyword in ['timing', 'clock', 'edge']):
            return TestPhase.TIMING_REQUIREMENTS
        elif any(keyword in text_lower for keyword in ['edge', 'boundary', 'overflow']):
            return TestPhase.EDGE_CASES
        else:
            return TestPhase.BASIC_FUNCTIONALITY

    def _assertion_matches(self, passing_text: str, failing_text: str) -> bool:
        """Check if a failing assertion matches a previously passing one."""
        # Normalize both texts for comparison
        passing_normalized = re.sub(r'\d+\.?\d*ns', 'XXXns', passing_text.lower())
        failing_normalized = re.sub(r'\d+\.?\d*ns', 'XXXns', failing_text.lower())

        # Look for key signal names or assertion patterns
        passing_signals = self._extract_signal_names(passing_text)
        failing_signals = self._extract_signal_names(failing_text)

        # Match if they share signal names and similar assertion patterns
        return (bool(passing_signals & failing_signals) and
                len(passing_normalized) > 10 and  # Avoid matching trivial assertions
                (passing_normalized in failing_normalized or
                 failing_normalized in passing_normalized))

    def _extract_signal_names(self, text: str) -> Set[str]:
        """Extract signal names from assertion text."""
        # Look for common RTL signal patterns
        signal_patterns = [
            r'\b([a-z_]+_(?:valid|ready|data|enable|reset))\b',
            r'\b([a-z_]+(?:_o|_i))\b',  # Common RTL naming
            r'\bdut\.([a-z_]+)\b',      # Testbench signal references
        ]

        signals = set()
        for pattern in signal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            signals.update(matches)

        return signals

    def _extract_expected_value(self, assertion_text: str) -> str:
        """Extract expected value from assertion."""
        patterns = [
            r"expected:?\s*([0-9a-fA-FxX]+)",
            r"Expected\s+([0-9a-fA-FxX]+)",
            r"should be\s+([0-9a-fA-FxX]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, assertion_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _extract_actual_value(self, assertion_text: str) -> str:
        """Extract actual value from assertion."""
        patterns = [
            r"actual:?\s*([0-9a-fA-FxX]+)",
            r"got\s+([0-9a-fA-FxX]+)",
            r"but was\s+([0-9a-fA-FxX]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, assertion_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _extract_signal_name(self, assertion_text: str) -> str:
        """Extract the primary signal name from assertion."""
        patterns = [
            r"dut\.([a-z_]+)\.value",
            r"([a-z_]+)\s*==",
            r"([a-z_]+)\s*!=",
        ]

        for pattern in patterns:
            match = re.search(pattern, assertion_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _assess_regression_severity(self, newly_failing: List[PassingAssertion]) -> str:
        """Assess the severity of detected regressions."""
        if not newly_failing:
            return "NONE"

        # High severity if reset or compilation regressions
        high_severity_phases = {TestPhase.COMPILATION, TestPhase.RESET_BEHAVIOR}
        if any(assertion.test_phase in high_severity_phases for assertion in newly_failing):
            return "HIGH"

        # Medium severity if multiple basic functionality regressions
        if len(newly_failing) >= 3:
            return "MEDIUM"

        return "LOW"

    def _generate_regression_recommendation(self, newly_failing: List[PassingAssertion],
                                          maintained_passing: List[PassingAssertion]) -> str:
        """Generate recommendation for handling regressions."""
        if not newly_failing:
            return "No regressions detected. Continue with current approach."

        recommendations = []

        if len(newly_failing) == 1:
            assertion = newly_failing[0]
            recommendations.append(f"SINGLE REGRESSION: The fix broke '{assertion.assertion_text}' "
                                 f"which was working since iteration {assertion.iteration_first_passed}.")
            recommendations.append("RECOMMENDATION: Make a more targeted fix that maintains the previous working behavior.")
        else:
            recommendations.append(f"MULTIPLE REGRESSIONS: {len(newly_failing)} previously passing assertions now fail.")
            recommendations.append("RECOMMENDATION: The current approach is too disruptive. "
                                 "Consider reverting to the last known good state and making smaller changes.")

        if maintained_passing:
            recommendations.append(f"POSITIVE: {len(maintained_passing)} assertions are still passing.")

        return " ".join(recommendations)
