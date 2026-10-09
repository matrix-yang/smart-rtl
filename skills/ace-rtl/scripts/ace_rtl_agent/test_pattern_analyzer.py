# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Pattern Analyzer - Identifies patterns in test failures vs passes.

This analyzer is design-agnostic and works by analyzing test inputs/outputs
to find patterns that distinguish failing from passing tests.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TestCase:
    """Represents a single test case with inputs/outputs."""
    name: str
    passed: bool
    inputs: Dict[str, Any] = None  # Input signal values
    expected_outputs: Dict[str, Any] = None
    actual_outputs: Dict[str, Any] = None
    timing_info: Optional[str] = None
    assertion_message: Optional[str] = None

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = {}
        if self.expected_outputs is None:
            self.expected_outputs = {}
        if self.actual_outputs is None:
            self.actual_outputs = {}


class TestPatternAnalyzer:
    """
    Analyzes test results to find patterns in failures.

    Design-agnostic approach:
    1. Extract test inputs/outputs from error messages
    2. Compare passing vs failing test characteristics
    3. Identify numerical/categorical patterns
    4. Suggest likely root causes based on patterns
    """

    def __init__(self, logger):
        self.logger = logger

    def analyze_patterns(self, passing_tests: List, failing_tests: List,
                        error_report: str) -> Dict[str, Any]:
        """
        Analyze patterns in test results.

        Returns:
            Dictionary with pattern analysis including:
            - Common characteristics of failing tests
            - Differences from passing tests
            - Hypotheses about root cause
        """
        # Parse test cases from error report
        parsed_passing = self._parse_test_cases(passing_tests, error_report, passed=True)
        parsed_failing = self._parse_test_cases(failing_tests, error_report, passed=False)

        analysis = {
            'total_passing': len(parsed_passing),
            'total_failing': len(parsed_failing),
            'pass_rate': len(parsed_passing) / (len(parsed_passing) + len(parsed_failing)) * 100 if (len(parsed_passing) + len(parsed_failing)) > 0 else 0,
            'patterns_found': [],
            'hypotheses': [],
            'summary': ""
        }

        # Find patterns if we have enough data
        if parsed_failing:
            # Pattern 1: Input value patterns
            input_patterns = self._analyze_input_patterns(parsed_passing, parsed_failing)
            if input_patterns:
                analysis['patterns_found'].extend(input_patterns)

            # Pattern 2: Output discrepancies
            output_patterns = self._analyze_output_patterns(parsed_failing)
            if output_patterns:
                analysis['patterns_found'].extend(output_patterns)

            # Pattern 3: Timing patterns
            timing_patterns = self._analyze_timing_patterns(parsed_passing, parsed_failing)
            if timing_patterns:
                analysis['patterns_found'].extend(timing_patterns)

            # Pattern 4: Sequential test patterns
            sequence_patterns = self._analyze_sequence_patterns(parsed_passing, parsed_failing)
            if sequence_patterns:
                analysis['patterns_found'].extend(sequence_patterns)

            # Generate hypotheses based on patterns
            analysis['hypotheses'] = self._generate_hypotheses(analysis['patterns_found'],
                                                               parsed_passing, parsed_failing)

            # Create summary
            analysis['summary'] = self._create_summary(analysis)

        return analysis

    def _parse_test_cases(self, test_results: List, error_report: str, passed: bool) -> List[TestCase]:
        """
        Parse test cases from error report to extract inputs/outputs.

        Uses general patterns that work across different test formats.
        """
        parsed_tests = []

        for test in test_results:
            test_name = test.test_name if hasattr(test, 'test_name') else str(test)
            assertion_msg = test.assertion_details if hasattr(test, 'assertion_details') else ""

            test_case = TestCase(
                name=test_name,
                passed=passed,
                assertion_message=assertion_msg
            )

            # Extract inputs from assertion message
            # Common patterns: "A=5, B=3", "input=10", "data[0]=255"
            if assertion_msg:
                input_matches = re.findall(r'(\w+)\s*[=:]\s*(\d+)', assertion_msg)
                for sig_name, value in input_matches:
                    try:
                        test_case.inputs[sig_name] = int(value)
                    except ValueError:
                        test_case.inputs[sig_name] = value

                # Extract expected vs actual
                # Patterns: "Expected X, got Y", "expected: X actual: Y"
                expected_match = re.search(r'[Ee]xpected[:\s]+(\d+)', assertion_msg)
                actual_match = re.search(r'[Gg]ot[:\s]+(\d+)|[Aa]ctual[:\s]+(\d+)', assertion_msg)

                if expected_match:
                    try:
                        test_case.expected_outputs['result'] = int(expected_match.group(1))
                    except ValueError:
                        pass

                if actual_match:
                    try:
                        actual_val = actual_match.group(1) or actual_match.group(2)
                        test_case.actual_outputs['result'] = int(actual_val)
                    except (ValueError, TypeError):
                        pass

            # Extract timing info
            timing_match = re.search(r'(\d+\.?\d*)\s*ns|cycle\s*(\d+)', assertion_msg or "")
            if timing_match:
                test_case.timing_info = timing_match.group(0)

            parsed_tests.append(test_case)

        return parsed_tests

    def _analyze_input_patterns(self, passing: List[TestCase], failing: List[TestCase]) -> List[str]:
        """
        Find patterns in input values that distinguish passing from failing tests.
        """
        patterns = []

        if not passing or not failing:
            return patterns

        # Collect all input signal names
        all_input_names = set()
        for test in passing + failing:
            all_input_names.update(test.inputs.keys())

        # Analyze each input signal
        for input_name in all_input_names:
            # Get values for passing and failing tests
            passing_values = [t.inputs.get(input_name) for t in passing if input_name in t.inputs]
            failing_values = [t.inputs.get(input_name) for t in failing if input_name in t.inputs]

            # Filter out None values
            passing_values = [v for v in passing_values if v is not None]
            failing_values = [v for v in failing_values if v is not None]

            if not passing_values or not failing_values:
                continue

            # Check for numerical patterns
            if all(isinstance(v, (int, float)) for v in passing_values + failing_values):
                pass_avg = sum(passing_values) / len(passing_values) if passing_values else 0
                fail_avg = sum(failing_values) / len(failing_values) if failing_values else 0

                pass_max = max(passing_values) if passing_values else 0
                fail_min = min(failing_values) if failing_values else 0
                fail_max = max(failing_values) if failing_values else 0

                # Pattern: Failing tests have larger/smaller values
                if fail_min > pass_max:
                    patterns.append(f"INPUT PATTERN: Signal '{input_name}' - all failing tests have values > {pass_max} (pass max: {pass_max}, fail min: {fail_min})")

                # Pattern: Significant difference in average
                if abs(fail_avg - pass_avg) > (pass_avg * 0.3 if pass_avg else 1):
                    patterns.append(f"INPUT PATTERN: Signal '{input_name}' - failing tests avg={fail_avg:.1f} vs passing avg={pass_avg:.1f}")

                # Pattern: Check for specific problematic values
                fail_value_counts = defaultdict(int)
                for v in failing_values:
                    fail_value_counts[v] += 1

                for value, count in fail_value_counts.items():
                    if count >= len(failing_values) * 0.5:  # More than 50% of failures
                        if value not in passing_values:
                            patterns.append(f"INPUT PATTERN: Signal '{input_name}' = {value} appears in {count}/{len(failing_values)} failures but 0/{len(passing_values)} passes")

        # Check for multi-signal patterns (relationships between inputs)
        if len(all_input_names) >= 2:
            relationship_pattern = self._analyze_input_relationships(passing, failing, list(all_input_names))
            if relationship_pattern:
                patterns.append(relationship_pattern)

        return patterns

    def _analyze_input_relationships(self, passing: List[TestCase], failing: List[TestCase],
                                    input_names: List[str]) -> Optional[str]:
        """
        Analyze relationships between multiple inputs.

        Example: Tests fail when A > B but pass when A <= B
        """
        if len(input_names) < 2:
            return None

        # Try common pairs
        for i in range(len(input_names)):
            for j in range(i + 1, len(input_names)):
                sig1, sig2 = input_names[i], input_names[j]

                # Check if both signals are numeric
                pass_pairs = [(t.inputs.get(sig1), t.inputs.get(sig2)) for t in passing
                             if sig1 in t.inputs and sig2 in t.inputs]
                fail_pairs = [(t.inputs.get(sig1), t.inputs.get(sig2)) for t in failing
                             if sig1 in t.inputs and sig2 in t.inputs]

                pass_pairs = [(a, b) for a, b in pass_pairs if isinstance(a, (int, float)) and isinstance(b, (int, float))]
                fail_pairs = [(a, b) for a, b in fail_pairs if isinstance(a, (int, float)) and isinstance(b, (int, float))]

                if not pass_pairs or not fail_pairs:
                    continue

                # Check: Do failures occur when sig1 > sig2?
                fail_when_greater = sum(1 for a, b in fail_pairs if a > b)
                pass_when_greater = sum(1 for a, b in pass_pairs if a > b)
                fail_when_less_eq = sum(1 for a, b in fail_pairs if a <= b)
                pass_when_less_eq = sum(1 for a, b in pass_pairs if a <= b)

                if fail_when_greater >= len(fail_pairs) * 0.7 and pass_when_less_eq >= len(pass_pairs) * 0.7:
                    return f"RELATIONSHIP PATTERN: Tests tend to fail when {sig1} > {sig2}, pass when {sig1} <= {sig2}"

                if fail_when_less_eq >= len(fail_pairs) * 0.7 and pass_when_greater >= len(pass_pairs) * 0.7:
                    return f"RELATIONSHIP PATTERN: Tests tend to fail when {sig1} <= {sig2}, pass when {sig1} > {sig2}"

                # Check: Do failures occur when values are equal?
                fail_when_equal = sum(1 for a, b in fail_pairs if a == b)
                pass_when_equal = sum(1 for a, b in pass_pairs if a == b)

                if fail_when_equal >= len(fail_pairs) * 0.5 and pass_when_equal == 0:
                    return f"RELATIONSHIP PATTERN: Tests fail when {sig1} == {sig2} (special case handling issue?)"

        return None

    def _analyze_output_patterns(self, failing: List[TestCase]) -> List[str]:
        """
        Analyze patterns in output errors.
        """
        patterns = []

        if not failing:
            return patterns

        # Collect expected vs actual
        errors = []
        for test in failing:
            if 'result' in test.expected_outputs and 'result' in test.actual_outputs:
                expected = test.expected_outputs['result']
                actual = test.actual_outputs['result']
                errors.append((expected, actual))

        if not errors:
            return patterns

        # Pattern: Always outputs same value (stuck-at fault)
        actual_values = [actual for _, actual in errors]
        if len(set(actual_values)) == 1:
            stuck_value = actual_values[0]
            patterns.append(f"OUTPUT PATTERN: Output is stuck at value {stuck_value} for all failing tests (signal not driven correctly?)")

        # Pattern: Off by constant amount
        if all(isinstance(e, (int, float)) and isinstance(a, (int, float)) for e, a in errors):
            differences = [expected - actual for expected, actual in errors]
            if len(set(differences)) == 1:
                diff = differences[0]
                patterns.append(f"OUTPUT PATTERN: Output is consistently off by {diff} (expected - actual = {diff} for all tests)")

            # Pattern: Off by factor
            if all(e != 0 for e, a in errors):
                ratios = [expected / actual if actual != 0 else float('inf') for expected, actual in errors]
                if len(set(ratios)) == 1 and ratios[0] != float('inf'):
                    ratio = ratios[0]
                    patterns.append(f"OUTPUT PATTERN: Output is consistently {ratio:.2f}x smaller than expected")

        # Pattern: Output is always 0 or always max value
        if all(actual == 0 for _, actual in errors):
            patterns.append(f"OUTPUT PATTERN: Output is always 0 (never driven/initialized?)")

        return patterns

    def _analyze_timing_patterns(self, passing: List[TestCase], failing: List[TestCase]) -> List[str]:
        """
        Analyze timing-related patterns.
        """
        patterns = []

        # Extract timing info
        pass_timings = [t.timing_info for t in passing if t.timing_info]
        fail_timings = [t.timing_info for t in failing if t.timing_info]

        if not fail_timings:
            return patterns

        # Parse timing values
        def extract_time_ns(timing_str):
            match = re.search(r'(\d+\.?\d*)', timing_str)
            return float(match.group(1)) if match else None

        pass_times = [extract_time_ns(t) for t in pass_timings]
        fail_times = [extract_time_ns(t) for t in fail_timings]

        pass_times = [t for t in pass_times if t is not None]
        fail_times = [t for t in fail_times if t is not None]

        if pass_times and fail_times:
            avg_pass_time = sum(pass_times) / len(pass_times)
            avg_fail_time = sum(fail_times) / len(fail_times)

            if avg_fail_time < avg_pass_time * 0.5:
                patterns.append(f"TIMING PATTERN: Failures occur very early (avg {avg_fail_time:.1f}ns) vs passes (avg {avg_pass_time:.1f}ns) - initialization issue?")
            elif avg_fail_time > avg_pass_time * 2:
                patterns.append(f"TIMING PATTERN: Failures occur much later (avg {avg_fail_time:.1f}ns) vs passes (avg {avg_pass_time:.1f}ns) - timeout/convergence issue?")

        # Check if all failures at same time
        if fail_times and len(set(fail_times)) == 1:
            fail_time = fail_times[0]
            patterns.append(f"TIMING PATTERN: All failures occur at exactly {fail_time}ns - specific clock cycle issue")

        return patterns

    def _analyze_sequence_patterns(self, passing: List[TestCase], failing: List[TestCase]) -> List[str]:
        """
        Analyze if test execution order matters (state not resetting, etc.)
        """
        patterns = []

        # Check if first test passes but later ones fail
        all_tests = passing + failing
        if len(all_tests) >= 3:
            # Check if there's a pattern in test order
            first_tests_pass = sum(1 for t in all_tests[:len(all_tests)//3] if t.passed)
            last_tests_pass = sum(1 for t in all_tests[-len(all_tests)//3:] if t.passed)

            if first_tests_pass > last_tests_pass * 2:
                patterns.append(f"SEQUENCE PATTERN: Earlier tests more likely to pass ({first_tests_pass}/{len(all_tests)//3}) than later tests ({last_tests_pass}/{len(all_tests)//3}) - state not resetting between tests?")

        return patterns

    def _generate_hypotheses(self, patterns: List[str], passing: List[TestCase],
                           failing: List[TestCase]) -> List[str]:
        """
        Generate specific hypotheses about root causes based on patterns.
        """
        hypotheses = []

        # Based on output patterns
        if any("stuck at value 0" in p for p in patterns):
            hypotheses.append("HYPOTHESIS: Output signal not assigned or not driven by logic (check always blocks, assignments)")

        if any("stuck at value" in p and "0" not in p for p in patterns):
            hypotheses.append("HYPOTHESIS: Output holding initial/reset value (check if computation logic executes)")

        if any("off by" in p for p in patterns):
            hypotheses.append("HYPOTHESIS: Arithmetic operation incorrect (check operators: +, -, *, /, %, bit operations)")

        # Based on input patterns
        if any("all failing tests have values >" in p for p in patterns):
            hypotheses.append("HYPOTHESIS: Bit width issue - larger values overflow or aren't handled correctly")

        if any("when" in p and ">" in p for p in patterns):
            hypotheses.append("HYPOTHESIS: Comparison logic reversed or conditional branch taken incorrectly")

        if any("when" in p and "==" in p for p in patterns):
            hypotheses.append("HYPOTHESIS: Edge case not handled (equality condition needs special logic)")

        # Based on timing patterns
        if any("very early" in p for p in patterns):
            hypotheses.append("HYPOTHESIS: Reset or initialization issue (outputs checked before computation completes)")

        if any("specific clock cycle" in p for p in patterns):
            hypotheses.append("HYPOTHESIS: FSM stuck in wrong state or transition condition not met at critical clock cycle")

        # Based on sequence patterns
        if any("state not resetting" in p for p in patterns):
            hypotheses.append("HYPOTHESIS: Module state persists between tests - missing reset logic or reset not asserted properly")

        # General hypothesis if no specific patterns found
        if not hypotheses and failing:
            if len(passing) == 0:
                hypotheses.append("HYPOTHESIS: Fundamental logic error - all tests fail (check specification understanding, module interface)")
            else:
                hypotheses.append("HYPOTHESIS: Partial implementation - some cases work but edge cases or specific conditions not handled")

        return hypotheses

    def _create_summary(self, analysis: Dict) -> str:
        """Create a human-readable summary of the pattern analysis."""
        summary_parts = []

        summary_parts.append(f"**Test Results**: {analysis['total_passing']} passing, {analysis['total_failing']} failing ({analysis['pass_rate']:.1f}% pass rate)")

        if analysis['patterns_found']:
            summary_parts.append(f"\n**Patterns Identified**: {len(analysis['patterns_found'])} patterns found:")
            for pattern in analysis['patterns_found'][:5]:  # Show top 5
                summary_parts.append(f"  - {pattern}")
        else:
            summary_parts.append("\n**Patterns Identified**: No clear patterns found in test data")

        if analysis['hypotheses']:
            summary_parts.append(f"\n**Root Cause Hypotheses** ({len(analysis['hypotheses'])} likely causes):")
            for hypothesis in analysis['hypotheses'][:3]:  # Show top 3
                summary_parts.append(f"  - {hypothesis}")

        return "\n".join(summary_parts)
