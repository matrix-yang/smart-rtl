# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Sequence Analyzer - Analyzes test sequences to understand timing and behavioral requirements
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class SequencePhase(Enum):
    """Different phases in a typical RTL test sequence."""
    INITIALIZATION = "initialization"
    RESET_SEQUENCE = "reset_sequence"
    INPUT_APPLICATION = "input_application"
    PROCESSING_WAIT = "processing_wait"
    OUTPUT_VERIFICATION = "output_verification"
    CLEANUP = "cleanup"


@dataclass
class TestStep:
    """Represents a single step in the test sequence."""
    phase: SequencePhase
    timing: str  # e.g., "250.00ns", "Clock 1"
    action: str  # What happens in this step
    expected_behavior: str  # What should happen
    signals_involved: List[str]  # Which signals are relevant


@dataclass
class TimingRequirement:
    """Represents a timing requirement extracted from test analysis."""
    signal_name: str
    requirement_type: str  # e.g., "setup_time", "hold_time", "response_delay"
    timing_constraint: str  # e.g., "must be 0 for 2 cycles after reset"
    context: str  # Additional context


@dataclass
class TestSequenceAnalysis:
    """Complete analysis of a test sequence."""
    test_steps: List[TestStep]
    timing_requirements: List[TimingRequirement]
    critical_transitions: List[str]
    reset_behavior_requirements: List[str]
    handshaking_patterns: List[str]
    common_pitfalls: List[str]


class TestSequenceAnalyzer:
    """
    Analyzes test sequences to understand the expected timing and behavioral patterns.

    This addresses the issue where agents don't understand the test's timing expectations
    and make fixes that work functionally but fail timing requirements.
    """

    def __init__(self, logger):
        self.logger = logger

    def analyze_test_sequence(self, error_report: str, specification: str = "") -> TestSequenceAnalysis:
        """
        Analyze the test sequence to understand timing and behavioral requirements.

        This is the main analysis function that extracts timing patterns and requirements.
        """
        self.logger.info("Analyzing test sequence for timing and behavioral patterns")

        # Extract test steps from the error report
        test_steps = self._extract_test_steps(error_report)

        # Analyze timing requirements
        timing_requirements = self._analyze_timing_requirements(error_report, test_steps)

        # Identify critical transitions
        critical_transitions = self._identify_critical_transitions(error_report, test_steps)

        # Extract reset behavior requirements
        reset_requirements = self._extract_reset_requirements(error_report)

        # Identify handshaking patterns
        handshaking_patterns = self._identify_handshaking_patterns(error_report, specification)

        # Identify common pitfalls based on error patterns
        common_pitfalls = self._identify_common_pitfalls(error_report)

        return TestSequenceAnalysis(
            test_steps=test_steps,
            timing_requirements=timing_requirements,
            critical_transitions=critical_transitions,
            reset_behavior_requirements=reset_requirements,
            handshaking_patterns=handshaking_patterns,
            common_pitfalls=common_pitfalls
        )

    def generate_timing_guidance(self, analysis: TestSequenceAnalysis) -> str:
        """
        Generate specific timing guidance for the RTL implementation.

        This helps the LLM understand what timing behavior is expected.
        """
        guidance = ["## TEST SEQUENCE TIMING ANALYSIS", ""]

        if analysis.test_steps:
            guidance.append("### Test Sequence Breakdown:")
            for i, step in enumerate(analysis.test_steps):
                guidance.append(f"{i+1}. **{step.phase.value.title()}** ({step.timing})")
                guidance.append(f"   - Action: {step.action}")
                guidance.append(f"   - Expected: {step.expected_behavior}")
                if step.signals_involved:
                    guidance.append(f"   - Signals: {', '.join(step.signals_involved)}")
                guidance.append("")

        if analysis.timing_requirements:
            guidance.append("### Critical Timing Requirements:")
            for req in analysis.timing_requirements:
                guidance.append(f"- **{req.signal_name}**: {req.timing_constraint}")
                if req.context:
                    guidance.append(f"  - Context: {req.context}")
            guidance.append("")

        if analysis.reset_behavior_requirements:
            guidance.append("### Reset Behavior Requirements:")
            for req in analysis.reset_behavior_requirements:
                guidance.append(f"- {req}")
            guidance.append("")

        if analysis.handshaking_patterns:
            guidance.append("### Handshaking Patterns:")
            for pattern in analysis.handshaking_patterns:
                guidance.append(f"- {pattern}")
            guidance.append("")

        if analysis.common_pitfalls:
            guidance.append("### Common Implementation Pitfalls to Avoid:")
            for pitfall in analysis.common_pitfalls:
                guidance.append(f"- {pitfall}")
            guidance.append("")

        guidance.append("**IMPLEMENTATION GUIDANCE:**")
        guidance.append("Your RTL implementation must satisfy ALL the above timing requirements.")
        guidance.append("Pay special attention to the exact timing of signal transitions.")

        return "\n".join(guidance)

    def _extract_test_steps(self, error_report: str) -> List[TestStep]:
        """Extract the sequence of test steps from the error report."""
        steps = []

        # Look for common test sequence patterns
        lines = error_report.split('\n')

        for i, line in enumerate(lines):
            # Look for timing markers
            timing_match = re.search(r'(\d+\.?\d*)\s*ns', line)
            if timing_match:
                timing = timing_match.group(0)

                # Classify the phase based on content
                phase = self._classify_test_phase(line, i, lines)

                # Extract action and expected behavior
                action = self._extract_action(line, i, lines)
                expected = self._extract_expected_behavior(line, i, lines)
                signals = self._extract_signals_from_line(line)

                if action or expected:  # Only add meaningful steps
                    steps.append(TestStep(
                        phase=phase,
                        timing=timing,
                        action=action,
                        expected_behavior=expected,
                        signals_involved=signals
                    ))

        return steps

    def _analyze_timing_requirements(self, error_report: str, test_steps: List[TestStep]) -> List[TimingRequirement]:
        """Analyze timing requirements from the test sequence."""
        requirements = []

        # Look for specific timing patterns in error messages
        timing_patterns = [
            # Pattern: signal should be X after reset
            (r'(\w+)\s+should be\s+(\d+)\s+after reset', 'reset_response'),
            # Pattern: signal must be X when Y is Z
            (r'(\w+)\s+must be\s+(\d+)\s+when\s+(\w+)\s+is\s+(\d+)', 'conditional_timing'),
            # Pattern: expected X at time Y
            (r'expected\s+(\w+)\s+at\s+(\d+\.?\d*ns)', 'absolute_timing'),
        ]

        for pattern, req_type in timing_patterns:
            matches = re.finditer(pattern, error_report, re.IGNORECASE)
            for match in matches:
                signal = match.group(1) if len(match.groups()) >= 1 else "unknown"
                constraint = match.group(0)
                context = self._extract_context_around_match(error_report, match)

                requirements.append(TimingRequirement(
                    signal_name=signal,
                    requirement_type=req_type,
                    timing_constraint=constraint,
                    context=context
                ))

        # Analyze reset timing requirements specifically
        reset_requirements = self._analyze_reset_timing(error_report)
        requirements.extend(reset_requirements)

        return requirements

    def _analyze_reset_timing(self, error_report: str) -> List[TimingRequirement]:
        """Analyze reset-specific timing requirements."""
        requirements = []

        # Look for reset success/failure patterns
        if "reset succesfull" in error_report.lower():
            # Reset passed, so understand what made it pass
            requirements.append(TimingRequirement(
                signal_name="reset_sequence",
                requirement_type="reset_success",
                timing_constraint="Reset sequence completed successfully",
                context="The reset behavior is working correctly"
            ))

        # Look for reset-related assertion failures
        reset_failure_patterns = [
            r'(\w+)\s+.*should be\s+(\d+).*after.*reset',
            r'reset.*(\w+).*expected\s+(\d+).*got\s+(\d+)',
        ]

        for pattern in reset_failure_patterns:
            matches = re.finditer(pattern, error_report, re.IGNORECASE)
            for match in matches:
                signal = match.group(1) if len(match.groups()) >= 1 else "unknown"

                requirements.append(TimingRequirement(
                    signal_name=signal,
                    requirement_type="reset_timing",
                    timing_constraint=f"Must be correct immediately after reset: {match.group(0)}",
                    context="Reset timing requirement"
                ))

        return requirements

    def _identify_critical_transitions(self, error_report: str, test_steps: List[TestStep]) -> List[str]:
        """Identify critical state transitions in the test."""
        transitions = []

        # Look for state changes in the test steps
        for i in range(len(test_steps) - 1):
            current_step = test_steps[i]
            next_step = test_steps[i + 1]

            # Identify phase transitions
            if current_step.phase != next_step.phase:
                transition = f"{current_step.phase.value} -> {next_step.phase.value}"
                transitions.append(f"Transition at {next_step.timing}: {transition}")

        # Look for signal transitions mentioned in errors
        transition_patterns = [
            r'(\w+)\s+(?:changed|transitioned|went)\s+from\s+(\w+)\s+to\s+(\w+)',
            r'(\w+)\s+should\s+(?:become|change to)\s+(\w+)',
        ]

        for pattern in transition_patterns:
            matches = re.finditer(pattern, error_report, re.IGNORECASE)
            for match in matches:
                transitions.append(f"Signal transition: {match.group(0)}")

        return transitions

    def _extract_reset_requirements(self, error_report: str) -> List[str]:
        """Extract specific reset behavior requirements."""
        requirements = []

        # Common reset requirement patterns
        reset_patterns = [
            r'(\w+)\s+should be\s+(\d+)\s+(?:after|during)\s+reset',
            r'reset.*(\w+).*must.*(\d+)',
            r'on reset.*(\w+).*(?:should|must).*(\w+)',
        ]

        for pattern in reset_patterns:
            matches = re.finditer(pattern, error_report, re.IGNORECASE)
            for match in matches:
                requirements.append(f"Reset requirement: {match.group(0)}")

        # Look for reset success indicators
        if "reset succesfull" in error_report.lower():
            requirements.append("Reset sequence must complete successfully")

        return requirements

    def _identify_handshaking_patterns(self, error_report: str, specification: str) -> List[str]:
        """Identify handshaking patterns from AXI, Avalon, or other protocols."""
        patterns = []

        # AXI Stream handshaking patterns
        axi_signals = ['valid', 'ready', 'data']
        if any(signal in error_report.lower() for signal in axi_signals):
            patterns.append("AXI Stream handshaking: valid/ready protocol required")
            patterns.append("Data transfer occurs when both valid=1 and ready=1")

        # Look for handshaking-related errors
        handshake_patterns = [
            r'(\w+)\s+and\s+(\w+)\s+must be\s+(?:high|1)\s+for\s+transfer',
            r'handshake.*(\w+).*(\w+)',
            r'transfer.*when.*(\w+).*(\w+)',
        ]

        for pattern in handshake_patterns:
            matches = re.finditer(pattern, error_report, re.IGNORECASE)
            for match in matches:
                patterns.append(f"Handshaking requirement: {match.group(0)}")

        return patterns

    def _identify_common_pitfalls(self, error_report: str) -> List[str]:
        """Identify common RTL implementation pitfalls based on error patterns."""
        pitfalls = []

        # Timing-related pitfalls
        if re.search(r'expected.*0.*got.*1', error_report, re.IGNORECASE):
            pitfalls.append("Signal may be asserted too early - check initialization timing")

        if re.search(r'expected.*1.*got.*0', error_report, re.IGNORECASE):
            pitfalls.append("Signal may not be asserted when required - check enable conditions")

        # Reset-related pitfalls
        if "reset" in error_report.lower() and "expected" in error_report.lower():
            pitfalls.append("Reset behavior may need delayed activation - signals should not be active immediately after reset")

        # Data formatting pitfalls
        if re.search(r'data.*not matching', error_report, re.IGNORECASE):
            pitfalls.append("Data formatting logic may be incorrect - check bit field assignments")

        # Handshaking pitfalls
        if any(signal in error_report.lower() for signal in ['ready', 'valid']):
            pitfalls.append("Handshaking timing may be incorrect - ensure proper ready/valid relationship")

        return pitfalls

    def _classify_test_phase(self, line: str, line_index: int, all_lines: List[str]) -> SequencePhase:
        """Classify which phase of testing a line represents."""
        line_lower = line.lower()

        if any(keyword in line_lower for keyword in ['initialization', 'init', 'before']):
            return SequencePhase.INITIALIZATION
        elif any(keyword in line_lower for keyword in ['reset', 'rst']):
            return SequencePhase.RESET_SEQUENCE
        elif any(keyword in line_lower for keyword in ['input', 'apply', 'set']):
            return SequencePhase.INPUT_APPLICATION
        elif any(keyword in line_lower for keyword in ['wait', 'delay', 'cycle']):
            return SequencePhase.PROCESSING_WAIT
        elif any(keyword in line_lower for keyword in ['assert', 'check', 'verify', 'expected']):
            return SequencePhase.OUTPUT_VERIFICATION
        else:
            return SequencePhase.PROCESSING_WAIT  # Default

    def _extract_action(self, line: str, line_index: int, all_lines: List[str]) -> str:
        """Extract the action being performed in this test step."""
        # Look for action keywords
        action_patterns = [
            r'(?:set|apply|drive)\s+(\w+)\s+(?:to|=)\s+(\w+)',
            r'(\w+)\s*=\s*(\w+)',
            r'wait\s+(\d+)\s+(?:cycle|clock)',
        ]

        for pattern in action_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(0)

        return ""

    def _extract_expected_behavior(self, line: str, line_index: int, all_lines: List[str]) -> str:
        """Extract the expected behavior for this test step."""
        # Look for expectation keywords
        expectation_patterns = [
            r'expected?\s+(\w+)\s+(?:to be|=)\s+(\w+)',
            r'should\s+be\s+(\w+)',
            r'must\s+(?:be|equal)\s+(\w+)',
        ]

        for pattern in expectation_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(0)

        return ""

    def _extract_signals_from_line(self, line: str) -> List[str]:
        """Extract signal names mentioned in a line."""
        # Common RTL signal patterns
        signal_patterns = [
            r'\b([a-z_]+_(?:valid|ready|data|enable|reset|clk))\b',
            r'\bdut\.([a-z_]+)\b',
            r'\b([a-z_]+_[io])\b',
        ]

        signals = []
        for pattern in signal_patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            signals.extend(matches)

        return list(set(signals))  # Remove duplicates

    def _extract_context_around_match(self, text: str, match) -> str:
        """Extract context around a regex match."""
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        return text[start:end].strip()
