# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Context Builder - Generates smart additional context for LLM.

Provides design-specific hints, testbench interpretation, and spec clarifications.
"""

import re
from typing import Dict, List, Optional


class ContextBuilder:
    """
    Builds smart additional context based on the specific situation.

    Design-agnostic approach:
    1. Identify design patterns in specification
    2. Extract common pitfalls for that design type
    3. Interpret what tests are checking
    4. Clarify ambiguous specification points
    """

    def __init__(self, logger):
        self.logger = logger

        # General RTL debugging tips (applicable to any design)
        self.general_tips = [
            "**Reset Logic**: All sequential logic should have proper reset conditions",
            "**Sensitivity Lists**: Combinational always blocks should use @(*) or @(posedge clk) for sequential",
            "**Assignment Types**: Use <= for sequential logic, = for combinational logic",
            "**Default Cases**: Always include 'default' in case statements to avoid latch inference",
            "**Bit Widths**: Ensure intermediate results don't overflow signal bit widths",
            "**State Machines**: All states should be reachable and have defined transitions"
        ]

    def build_context(self, specification: str, error_report: str,
                     rtl_code: str, iteration: int,
                     testbench_code: Optional[str] = None) -> str:
        """
        Build smart additional context for the current debugging situation.

        Returns context string with:
        - Design-specific hints
        - Test interpretation
        - Specification clarifications
        - Common pitfalls
        """
        context_parts = []

        # Part 1: Identify design type and provide specific hints
        design_hints = self._generate_design_hints(specification, error_report)
        if design_hints:
            context_parts.append("## DESIGN-SPECIFIC GUIDANCE")
            context_parts.append(design_hints)
            context_parts.append("")

        # Part 2: Interpret what tests are checking
        if testbench_code or error_report:
            test_interpretation = self._interpret_test_expectations(
                testbench_code, error_report, specification
            )
            if test_interpretation:
                context_parts.append("## TEST EXPECTATIONS")
                context_parts.append(test_interpretation)
                context_parts.append("")

        # Part 3: Clarify specification ambiguities
        spec_clarifications = self._clarify_specification(specification, error_report, rtl_code)
        if spec_clarifications:
            context_parts.append("## SPECIFICATION CLARIFICATIONS")
            context_parts.append(spec_clarifications)
            context_parts.append("")

        # Part 4: Provide relevant general tips based on the error
        relevant_tips = self._select_relevant_tips(error_report, rtl_code)
        if relevant_tips:
            context_parts.append("## RELEVANT RTL BEST PRACTICES")
            for tip in relevant_tips:
                context_parts.append(f"- {tip}")
            context_parts.append("")

        return "\n".join(context_parts) if context_parts else ""

    def _generate_design_hints(self, specification: str, error_report: str) -> str:
        """
        Generate hints specific to the type of design being implemented.

        Uses pattern matching on specification to identify design type.
        """
        spec_lower = specification.lower()
        hints = []

        # Pattern: Arithmetic/computation modules
        if any(word in spec_lower for word in ['compute', 'calculate', 'arithmetic', 'gcd', 'multiply', 'divide']):
            hints.append("**Arithmetic Design Tips**:")
            hints.append("- Ensure computation logic is in a loop or state machine for iterative algorithms")
            hints.append("- Output should only update when computation is complete")
            hints.append("- Consider using a 'done' or 'valid' signal to indicate result ready")
            hints.append("- Handle edge cases: zero inputs, equal inputs, maximum values")

        # Pattern: State machines
        if any(word in spec_lower for word in ['state', 'fsm', 'sequence', 'protocol']):
            hints.append("**State Machine Tips**:")
            hints.append("- Use separate always blocks for state transitions and output logic")
            hints.append("- Ensure all states have transitions defined (no dead states)")
            hints.append("- Reset should set state machine to initial state")
            hints.append("- Check that transition conditions are mutually exclusive to avoid ambiguity")

        # Pattern: Memory/storage modules
        if any(word in spec_lower for word in ['memory', 'fifo', 'buffer', 'queue', 'ram', 'register file']):
            hints.append("**Memory/Storage Design Tips**:")
            hints.append("- Read and write operations should be properly synchronized")
            hints.append("- Handle full/empty conditions with appropriate flags")
            hints.append("- Consider read-write conflicts (same address accessed simultaneously)")
            hints.append("- Pointer wrapping should be handled correctly")

        # Pattern: Counters/timers
        if any(word in spec_lower for word in ['count', 'timer', 'increment', 'decrement']):
            hints.append("**Counter/Timer Tips**:")
            hints.append("- Check for overflow/underflow conditions")
            hints.append("- Ensure enable/load signals work correctly")
            hints.append("- Reset should set counter to initial value")
            hints.append("- Terminal count conditions should be properly detected")

        # Pattern: Data path modules
        if any(word in spec_lower for word in ['alu', 'datapath', 'operation', 'opcode']):
            hints.append("**Datapath Design Tips**:")
            hints.append("- All operations should be covered in case statement")
            hints.append("- Flags (zero, carry, overflow) should be updated correctly")
            hints.append("- Ensure bit widths are sufficient for all operations")
            hints.append("- Consider signed vs unsigned operations")

        # Pattern: Control signal modules
        if any(word in spec_lower for word in ['control', 'handshake', 'valid', 'ready', 'ack']):
            hints.append("**Control Signal Tips**:")
            hints.append("- Handshake protocols must follow specification exactly")
            hints.append("- Valid/ready signals should have correct timing relationships")
            hints.append("- Avoid combinational loops between control signals")
            hints.append("- Document which signals are active high vs active low")

        # If no specific type identified, provide general guidance
        if not hints:
            hints.append("**General Implementation Tips**:")
            hints.append("- Break down specification into clear inputs, outputs, and behavior")
            hints.append("- Identify if design needs sequential (FSM) or combinational logic")
            hints.append("- List all possible states/modes the design can be in")
            hints.append("- Handle all edge cases mentioned in specification")

        return "\n".join(hints)

    def _interpret_test_expectations(self, testbench_code: Optional[str],
                                    error_report: str, specification: str) -> str:
        """
        Interpret what the tests are actually checking.

        Helps LLM understand test intent, not just assertion messages.
        """
        interpretations = []

        # Parse assertion messages from error report
        assertions = re.findall(r'AssertionError: ([^\n]+)', error_report)

        if assertions:
            interpretations.append("**What Tests Are Checking**:")

            for assertion in assertions[:5]:  # Show first 5
                # Extract the key requirement
                if 'expected' in assertion.lower() and 'got' in assertion.lower():
                    # Extract values
                    expected_match = re.search(r'[Ee]xpected[:\s]+(\w+)', assertion)
                    got_match = re.search(r'[Gg]ot[:\s]+(\w+)', assertion)

                    if expected_match and got_match:
                        expected = expected_match.group(1)
                        got = got_match.group(1)
                        interpretations.append(f"- Test expects output = {expected}, but got {got}")

                        # Interpret what this means
                        if got == '0':
                            interpretations.append(f"  -> Output not driven or stuck at 0")
                        elif expected.isdigit() and got.isdigit():
                            exp_val = int(expected)
                            got_val = int(got)
                            if got_val < exp_val:
                                interpretations.append(f"  -> Output is too small (computation incomplete?)")
                            else:
                                interpretations.append(f"  -> Output is too large (wrong formula?)")

                # Check for timing requirements
                if 'cycle' in assertion.lower() or 'time' in assertion.lower():
                    interpretations.append(f"- Test has timing requirements: {assertion[:80]}")
                    interpretations.append(f"  -> Design must produce result within specified time/cycles")

                # Check for signal state requirements
                if 'high' in assertion.lower() or 'low' in assertion.lower():
                    interpretations.append(f"- Test checks signal state: {assertion[:80]}")
                    interpretations.append(f"  -> Control signals (done, valid, etc.) must have correct values")

        # Look for test patterns in testbench code
        if testbench_code:
            # Check for wait statements
            wait_patterns = re.findall(r'wait.*?(\d+)\s*(ns|cycles?)', testbench_code, re.IGNORECASE)
            if wait_patterns:
                interpretations.append(f"\n**Timing Constraints**:")
                interpretations.append(f"- Tests wait for results - ensure computation completes within deadline")

        # Infer from specification
        if 'latency' in specification.lower():
            interpretations.append(f"\n**Latency Requirements**:")
            latency_match = re.search(r'latency.*?(\d+)\s*(cycle|clock)', specification, re.IGNORECASE)
            if latency_match:
                cycles = latency_match.group(1)
                interpretations.append(f"- Design must complete in {cycles} clock cycles (from specification)")

        return "\n".join(interpretations) if interpretations else ""

    def _clarify_specification(self, specification: str, error_report: str, rtl_code: str) -> str:
        """
        Identify and clarify ambiguous parts of the specification.
        """
        clarifications = []

        spec_lower = specification.lower()

        # Common ambiguity: Output behavior before ready
        if 'output' in spec_lower and 'done' not in spec_lower:
            clarifications.append("**Output Initialization**: Specification doesn't explicitly state output value before computation completes")
            clarifications.append("  -> Best practice: Set output to 0 initially, update when result ready")

        # Common ambiguity: Signal timing
        if 'signal' in error_report.lower() and ('high' in error_report.lower() or 'pulse' in error_report.lower()):
            clarifications.append("**Signal Duration**: Tests may expect control signals to pulse (high for 1 cycle) vs hold")
            clarifications.append("  -> Check test assertions to see if signal should pulse or remain high")

        # Common ambiguity: Reset behavior
        if 'reset' in spec_lower or 'rst' in rtl_code.lower():
            if 'active' not in spec_lower:
                clarifications.append("**Reset Polarity**: Specification doesn't state if reset is active high or low")
                clarifications.append("  -> Check testbench to determine reset polarity")

        # Common ambiguity: Input timing
        if 'input' in spec_lower and 'valid' not in spec_lower:
            clarifications.append("**Input Timing**: Specification may not state when inputs are valid")
            clarifications.append("  -> Assume inputs are valid on same clock edge as computation starts")

        # Check for undefined edge cases
        if any(word in spec_lower for word in ['compute', 'calculate', 'arithmetic']):
            if 'zero' not in spec_lower:
                clarifications.append("**Edge Case**: Specification may not define behavior for zero inputs")
                clarifications.append("  -> Handle gracefully (typically: result = 0 or result = other input)")

        return "\n".join(clarifications) if clarifications else ""

    def _select_relevant_tips(self, error_report: str, rtl_code: str) -> List[str]:
        """
        Select relevant tips from general best practices based on current situation.
        """
        relevant = []

        error_lower = error_report.lower()
        code_lower = rtl_code.lower()

        # If reset mentioned in error
        if 'reset' in error_lower or 'initialization' in error_lower:
            relevant.append(self.general_tips[0])  # Reset logic tip

        # If timing issues
        if 'time' in error_lower or 'cycle' in error_lower:
            relevant.append(self.general_tips[1])  # Sensitivity list tip

        # If output is 0 or wrong value
        if 'got 0' in error_lower or 'stuck' in error_lower:
            relevant.append(self.general_tips[2])  # Assignment type tip

        # If code has case statements
        if 'case' in code_lower and 'default' not in code_lower:
            relevant.append(self.general_tips[3])  # Default case tip

        # If values seem off
        if 'expected' in error_lower and 'got' in error_lower:
            # Check if values are very different
            expected_match = re.search(r'expected[:\s]+(\d+)', error_lower)
            got_match = re.search(r'got[:\s]+(\d+)', error_lower)
            if expected_match and got_match:
                try:
                    expected = int(expected_match.group(1))
                    got = int(got_match.group(1))
                    if abs(expected - got) > max(expected, got) * 0.5:  # Off by >50%
                        relevant.append(self.general_tips[4])  # Bit width tip
                except ValueError:
                    pass

        # If state machine detected
        if 'state' in code_lower:
            relevant.append(self.general_tips[5])  # State machine tip

        # If not many relevant tips, add a few general ones
        if len(relevant) < 2:
            relevant.extend(self.general_tips[:3])

        return relevant[:4]  # Return top 4 most relevant
