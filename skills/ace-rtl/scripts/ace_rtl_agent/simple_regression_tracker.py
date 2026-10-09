# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple Regression Tracker - Minimal rule-based tracking, maximum LLM reasoning
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class IterationResult:
    """Simple tracking of iteration results."""
    iteration: int
    test_passed: bool
    error_report: str
    code: str
    improvements_noted: List[str] = None


class SimpleRegressionTracker:
    """
    Minimal regression tracking that relies on LLM reasoning rather than complex rules.

    Philosophy: Let the LLM do the pattern recognition and reasoning.
    Our job is just to provide context and detect obvious regressions.
    """

    def __init__(self, logger):
        self.logger = logger
        self.iteration_history: List[IterationResult] = []
        self.last_successful_patterns: List[str] = []

    def record_iteration(self, iteration: int, test_passed: bool,
                        error_report: str, code: str):
        """Record an iteration result."""
        result = IterationResult(
            iteration=iteration,
            test_passed=test_passed,
            error_report=error_report,
            code=code
        )
        self.iteration_history.append(result)

        # Simple pattern: if test passed, note what was working
        if test_passed:
            self.last_successful_patterns = self._extract_simple_success_patterns(error_report)

    def generate_context_for_llm(self, current_iteration: int) -> str:
        """
        Generate simple context for LLM reasoning.
        No complex rules - just provide the facts.
        """
        if not self.iteration_history:
            return "No previous iterations to reference."

        context = ["## ITERATION HISTORY CONTEXT", ""]

        # Show recent history (last 3 iterations)
        recent_history = self.iteration_history[-3:]

        for result in recent_history:
            status = "PASSED" if result.test_passed else "FAILED"
            context.append(f"**Iteration {result.iteration}**: {status}")

            # Show key parts of error report (let LLM interpret)
            if result.error_report:
                # Just extract the key failure line, not complex parsing
                key_line = self._extract_key_failure_line(result.error_report)
                if key_line:
                    context.append(f"  - Key issue: {key_line}")
            context.append("")

        # Show what was last working (if anything)
        if self.last_successful_patterns:
            context.append("### Last Known Working Patterns:")
            for pattern in self.last_successful_patterns:
                context.append(f"- {pattern}")
            context.append("")

        # Simple regression detection
        regression_warning = self._detect_simple_regression()
        if regression_warning:
            context.append("### POSSIBLE REGRESSION DETECTED:")
            context.append(regression_warning)
            context.append("")

        context.append("**LLM Task**: Analyze the above history and ensure new fixes don't break previously working functionality.")

        return "\n".join(context)

    def _extract_simple_success_patterns(self, error_report: str) -> List[str]:
        """Extract simple success patterns - no complex rules."""
        patterns = []

        # Look for obvious success indicators
        success_indicators = [
            "completed successfully", "passed", "success",
            "\u2705", "PASS", "OK", "succeeded"
        ]

        lines = error_report.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in success_indicators):
                patterns.append(line.strip())

        return patterns[:3]  # Limit to avoid noise

    def _extract_key_failure_line(self, error_report: str) -> Optional[str]:
        """Extract the key failure line - simple pattern matching."""
        lines = error_report.split('\n')

        # Look for assertion errors or key failure patterns
        failure_patterns = [
            "AssertionError", "FAILED", "Error:", "failed",
            "expected", "assert", "\u274c"
        ]

        for line in lines:
            if any(pattern in line for pattern in failure_patterns):
                # Return first significant failure line
                if len(line.strip()) > 20:  # Avoid trivial lines
                    return line.strip()

        return None

    def _detect_simple_regression(self) -> Optional[str]:
        """Simple regression detection - no complex rules."""
        if len(self.iteration_history) < 2:
            return None

        current = self.iteration_history[-1]
        previous = self.iteration_history[-2]

        # Simple check: if we went from passing to failing
        if previous.test_passed and not current.test_passed:
            return f"Iteration {previous.iteration} was passing, but iteration {current.iteration} failed. Possible regression."

        # Check if we're seeing repeated similar failures
        if len(self.iteration_history) >= 3:
            last_three = self.iteration_history[-3:]
            if all(not result.test_passed for result in last_three):
                return "Multiple consecutive failures detected. Consider if fixes are conflicting."

        return None
