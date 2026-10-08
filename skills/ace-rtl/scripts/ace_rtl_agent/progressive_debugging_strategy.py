# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Progressive Debugging Strategy - Escalates complexity based on iteration success, not error type
"""

from typing import Dict, Any, Tuple, Optional, List
from enum import Enum
import re

class DebuggingApproach(Enum):
    """Different debugging approaches in order of complexity."""
    FOCUSED_ASSERTION = "focused_assertion"      # Iterations 1-3: Focus on ONE assertion
    CONTEXTUAL_ANALYSIS = "contextual_analysis"  # Iterations 4-7: Add full context
    SYSTEMATIC_DEBUGGING = "systematic_debugging" # Iterations 8+: Complex analysis

class ReasoningFramework:
    """Structured reasoning steps that force systematic analysis."""

    STEP_BY_STEP_ANALYSIS = [
        "STEP 1: UNDERSTAND THE EXACT FAILURE",
        "STEP 2: TRACE THROUGH YOUR LOGIC",
        "STEP 3: IDENTIFY THE ROOT CAUSE",
        "STEP 4: DESIGN THE APPROPRIATE FIX",
        "STEP 5: VERIFY THE FIX LOGIC"
    ]

    VERIFICATION_QUESTIONS = [
        "What exact condition causes this specific failure?",
        "How does your current logic handle this condition?",
        "What should happen instead according to specification?",
        "What level of change is needed: surface fix vs architectural change?",
        "Will this change break any other scenarios?"
    ]

class ProgressiveDebuggingStrategy:
    """
    Implements progressive debugging complexity.

    Key insight: start simple and only add complexity when needed,
    rather than using the same complex approach for every iteration.
    """

    def __init__(self, logger):
        self.logger = logger

    def select_debugging_approach(self, iteration: int, error_history: list = None) -> DebuggingApproach:
        """
        Select debugging approach based on iteration and history.

        This is the general principle: start simple, escalate complexity progressively.
        """

        # Early iterations: Focus on single assertion failures
        if iteration <= 3:
            self.logger.info(f"Iteration {iteration}: Using focused assertion debugging")
            return DebuggingApproach.FOCUSED_ASSERTION

        # Middle iterations: Add full context if simple approach failed
        elif iteration <= 7:
            self.logger.info(f"Iteration {iteration}: Using contextual analysis")
            return DebuggingApproach.CONTEXTUAL_ANALYSIS

        # Late iterations: Use complex systematic debugging
        else:
            self.logger.info(f"Iteration {iteration}: Using systematic debugging")
            return DebuggingApproach.SYSTEMATIC_DEBUGGING

    def create_approach_prompt(self, approach: DebuggingApproach,
                              specification: str, current_code: str, error_report: str,
                              primary_failure: Any = None, spec_examples: list = None,
                              context_analysis: Dict = None, enhanced_context: str = "") -> str:
        """Create debugging prompt based on selected approach with enhanced analysis."""

        # Extract detailed test case information for all approaches
        test_details = self.extract_detailed_test_case(error_report)
        test_details = self.validate_extracted_information(test_details)

        # Extract architectural constraints to prevent inconsistent design decisions
        # Try to include testbench code for better constraint analysis
        testbench_code = self._attempt_testbench_extraction(error_report)
        arch_constraints = self.extract_architectural_constraints(specification, error_report, testbench_code)

        if approach == DebuggingApproach.FOCUSED_ASSERTION:
            return self._create_enhanced_focused_prompt(
                specification, current_code, error_report, primary_failure,
                spec_examples, enhanced_context, test_details, arch_constraints
            )
        elif approach == DebuggingApproach.CONTEXTUAL_ANALYSIS:
            return self._create_enhanced_contextual_prompt(
                specification, current_code, error_report, context_analysis,
                enhanced_context, test_details, arch_constraints
            )
        else:  # SYSTEMATIC_DEBUGGING
            return self._create_enhanced_systematic_prompt(
                specification, current_code, error_report, enhanced_context, test_details, arch_constraints
            )

    def _create_focused_assertion_prompt(self, specification: str, current_code: str,
                                       error_report: str, primary_failure: Any,
                                       spec_examples: list, enhanced_context: str = "") -> str:
        """Create simple, focused prompt for ONE assertion failure."""

        if not primary_failure:
            return self._fallback_prompt(specification, current_code, error_report)

        # Find relevant spec example based on failure context (general approach)
        relevant_example = ""
        if spec_examples:
            # Match based on failure context and test scenario
            failure_context = f"{primary_failure.context_lines} {primary_failure.timing_context}"
            for example in spec_examples:
                # Look for context keywords that indicate scenario relevance
                example_context = example.context.lower()
                if any(keyword in example_context for keyword in ['reset', 'init', 'start', 'error', 'boundary']):
                    if any(keyword in failure_context.lower() for keyword in ['reset', 'init', 'start', 'error', 'boundary']):
                        relevant_example = f"\n**Spec Example**: {example.title} - {', '.join(f'{k}={v}' for k, v in example.signal_values.items())}"
                        break



        prompt = f"""Fix this ONE specific assertion failure:

**The Problem**: Expected {primary_failure.expected_value}, got {primary_failure.actual_value}
**When**: {primary_failure.timing_context}
**Context**: {primary_failure.context_lines}



{relevant_example}

{enhanced_context}

**SPECIFICATION REQUIREMENTS**:
Follow the exact interface and behavior specified in the requirements.
Generate ONLY the target RTL module - no testbench code.

**Current Code**:
```systemverilog
{current_code}
```

**Analysis**: Why would the DUT output {primary_failure.actual_value} when {primary_failure.expected_value} was expected?

**Fix**: Make the appropriate-level change to address this root cause.
- If it's a surface issue: make a targeted fix
- If it's an architectural issue: make bold structural changes
- Focus on effectiveness over minimalism

```systemverilog
[Your corrected code - appropriate scope for the identified problem]
```
"""
        return prompt

    def _create_enhanced_focused_prompt(self, specification: str, current_code: str,
                                      error_report: str, primary_failure: Any,
                                      spec_examples: list, enhanced_context: str = "",
                                      test_details: Dict[str, Any] = None,
                                      arch_constraints: Dict[str, Any] = None) -> str:
        """Create enhanced focused prompt with deep error analysis."""

        if not primary_failure:
            return self._fallback_prompt(specification, current_code, error_report)

        # Use enhanced test details if available
        if not test_details:
            test_details = self.extract_detailed_test_case(error_report)
            test_details = self.validate_extracted_information(test_details)

        # Find relevant spec example based on test context, not specific expected values
        relevant_example = ""
        if spec_examples:
            # Match example context to test scenario (general approach)
            test_context = test_details.get('sequence_step', '').lower()
            test_name = test_details.get('test_name', '').lower()

            # Find example that matches the current test context
            for example in spec_examples:
                example_context = example.context.lower()
                # Look for context overlap between example and current test scenario
                if any(keyword in example_context for keyword in ['reset', 'init', 'start', 'error', 'boundary']) and \
                   any(keyword in test_context or keyword in test_name for keyword in ['reset', 'init', 'start', 'error', 'boundary']):
                    relevant_example = f"\n**Spec Example**: {example.title} - {', '.join(f'{k}={v}' for k, v in example.signal_values.items())}"
                    break

        # Create specification alignment analysis
        spec_analysis = self.analyze_specification_alignment(specification, test_details)

        # Create test-driven analysis
        test_driven_analysis = self.create_test_driven_analysis(test_details, specification)

        # Create architectural constraint guidance
        arch_guidance = self._create_architectural_guidance(arch_constraints) if arch_constraints else ""

        prompt = f"""Fix this ONE specific assertion failure with deep analysis:

**DETAILED FAILURE ANALYSIS**:
- Test Case: {test_details['test_name']}
- Timing: {test_details['cycle_number']} ({primary_failure.timing_context})
- Expected: {primary_failure.expected_value}, Got: {primary_failure.actual_value}
- Context: {primary_failure.context_lines}
- Input State: {test_details['input_values']}

{relevant_example}

{spec_analysis}

{test_driven_analysis}

{arch_guidance}

{enhanced_context}

**SPECIFICATION REQUIREMENTS**:
Follow the exact interface and behavior specified in the requirements.
Generate ONLY the target RTL module - no testbench code.

**Current Code**:
```systemverilog
{current_code}
```

**MANDATORY STRUCTURED REASONING**:
You MUST complete ALL steps before writing any code:

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[0]}**
- What exactly failed? Expected: {primary_failure.expected_value}, Actual: {primary_failure.actual_value}
- At what timing/condition? {primary_failure.timing_context}
- What was the test doing when this happened?

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[1]}**
- Trace through your RTL logic with the exact failing inputs
- Step through each assignment, condition, and state change
- Identify where your logic produces {primary_failure.actual_value}

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[2]}**
- Why does your logic produce the wrong result?
- Is this a timing issue, logic issue, or specification misunderstanding?
- What fundamental assumption is incorrect?

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[3]}**
- What is the smallest change that fixes this specific failure?
- Focus on the root cause, not symptoms

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[4]}**
- Will your fix handle the failing test case correctly?
- Will your fix break any other scenarios?
- Does your fix align with specification requirements?

**ONLY AFTER COMPLETING ALL STEPS**: Write the corrected code.
**Fix Scope**: Use the appropriate level of change based on your root cause analysis:
- Surface logic issues -> Targeted fixes
- Architectural/timing issues -> Structural changes
- Don't be constrained by "minimal" if bold changes are needed

```systemverilog
[Your corrected code - appropriate scope for the root cause]
```
"""
        return prompt

    def _create_contextual_analysis_prompt(self, specification: str, current_code: str,
                                         error_report: str, context_analysis: Dict,
                                         enhanced_context: str = "") -> str:
        """Create prompt with full context analysis."""

        enhanced_section = f"\n{enhanced_context}\n" if enhanced_context else ""

        prompt = f"""Debug this RTL issue using comprehensive context:

**Complete Error Report**:
```
{error_report}
```

**Specification**:
```
{specification}
```
{enhanced_section}
**Current Code**:
```systemverilog
{current_code}
```

**Analysis**: Read the complete error report and understand all aspects of the failure.

**Solution**: Address the root cause with targeted changes.

```systemverilog
[Your corrected code]
```
"""
        return prompt

    def _create_systematic_debugging_prompt(self, specification: str, current_code: str,
                                          error_report: str, enhanced_context: str = "") -> str:
        """Create comprehensive systematic debugging prompt."""

        enhanced_section = f"\n{enhanced_context}\n" if enhanced_context else ""

        prompt = f"""Use systematic debugging for this complex RTL issue:

**Complete Context**:
- Specification: {specification}
- Error Report: {error_report}
- Current Code: {len(current_code)} lines
{enhanced_section}
**Systematic Analysis**:
1. What is the fundamental issue?
2. What systematic changes are needed?
3. How should the design be restructured?

```systemverilog
[Your corrected or redesigned code]
```
"""
        return prompt

    def _fallback_prompt(self, specification: str, current_code: str, error_report: str) -> str:
        """Fallback prompt when primary failure extraction fails."""

        return f"""Debug this RTL issue:

**Error**: {error_report}
**Specification**: {specification}
**Code**:
```systemverilog
{current_code}
```

Analyze and fix:
```systemverilog
[Your corrected code]
```
"""

    # === ENHANCED DEBUGGING METHODS ===

    def extract_detailed_test_case(self, error_report: str) -> Dict[str, Any]:
        """
        Extract detailed information about the failing test case.

        Implements Deep-Dive Error Analysis - understand exactly which input
        causes failure and the complete test scenario.
        """
        test_details = {
            'test_name': 'unknown',
            'input_values': {},
            'expected_output': 'unknown',
            'actual_output': 'unknown',
            'cycle_number': 'unknown',
            'sequence_step': 'unknown',
            'full_scenario': ''
        }

        try:
            # Extract test name/case - look for more specific patterns first
            test_name_patterns = [
                r"running\s+([\w\.]+\.test_[\w_]+)",    # "running test_module.test_name"
                r"(test_[\w_]+)\s+failed",              # "test_name failed"
                r"(test_[\w_]+)\s*\(",                  # "test_name(" in function calls
                r"@cocotb\.test\(\).*?async def\s+(test_\w+)", # cocotb decorator pattern
                r"test[_\s]+(\w+)",                     # generic "test name"
                r"Test[:\s]+(\w+)",                     # "Test: name"
                r"Running\s+(\w+)",                     # "Running name"
            ]
            for pattern in test_name_patterns:
                match = re.search(pattern, error_report, re.IGNORECASE)
                if match:
                    test_details['test_name'] = match.group(1)
                    break

            # Extract input values that led to failure - focus on actual DUT signals
            input_patterns = [
                r"dut\.(\w+)\.value\s*[=:]\s*(\w+)",    # cocotb: "dut.signal.value = X"
                r"(\w+)\s*<=\s*(\w+)",                  # SystemVerilog: "signal <= value"
                r"(\w+)\s*=\s*(\w+)",                   # General: "signal = value"
                r"input\s+(\w+)[:\s]*(\w+)",            # "input signal: value"
                r"signal\s+(\w+)[:\s]*(\w+)"            # "signal name: value"
            ]

            # Only extract signal names that look like actual DUT ports
            valid_signal_patterns = [r'^(clk|rst|reset|enable|valid|ready|data|addr|sel).*', r'^[a-z][a-z0-9_]*$']

            for pattern in input_patterns:
                matches = re.finditer(pattern, error_report, re.IGNORECASE)
                for match in matches:
                    signal_name = match.group(1)
                    signal_value = match.group(2)

                    # Filter out non-signal names (like test infrastructure)
                    if any(re.match(valid_pattern, signal_name, re.IGNORECASE) for valid_pattern in valid_signal_patterns):
                        test_details['input_values'][signal_name] = signal_value
                    elif signal_name.lower() in ['clk', 'clock', 'rst', 'reset', 'enable', 'valid', 'ready']:
                        test_details['input_values'][signal_name] = signal_value

            # Extract cycle/timing information - prioritize context around assertion failures
            timing_context = ""
            if "AssertionError" in error_report:
                # Look for timing info near the assertion
                assertion_pos = error_report.find("AssertionError")
                timing_context = error_report[max(0, assertion_pos-500):assertion_pos+100]
            else:
                timing_context = error_report

            cycle_patterns = [
                r"(\d+\.?\d*)\s*ns\s+INFO.*AssertionError",  # Timing right before assertion
                r"(\d+\.?\d*)\s*ns.*(?:Expected|Assert)",     # Timing near expectation
                r"cycle[:\s]*(\d+)",                          # "cycle: N"
                r"(\d+\.?\d*)\s*ns",                          # Any timing in ns
                r"time[:\s]*(\d+)",                           # "time: N"
                r"step[:\s]*(\d+)"                            # "step: N"
            ]
            for pattern in cycle_patterns:
                match = re.search(pattern, timing_context, re.IGNORECASE)
                if match:
                    test_details['cycle_number'] = match.group(1)
                    break

            # Extract expected vs actual values with more context
            expectation_patterns = [
                r"[Ee]xpected[:\s]*(\w+).*(?:got|actual|received)[:\s]*(\w+)",
                r"should\s+be[:\s]*(\w+).*but\s+was[:\s]*(\w+)",
                r"expected[:\s]*(\w+).*found[:\s]*(\w+)"
            ]
            for pattern in expectation_patterns:
                match = re.search(pattern, error_report, re.IGNORECASE)
                if match:
                    test_details['expected_output'] = match.group(1)
                    test_details['actual_output'] = match.group(2)
                    break

            # Extract sequence step context - focus on test operation context
            sequence_patterns = [
                r"after\s+(reset|initialization|error|completion)",  # Key RTL operations (general)
                r"during\s+(reset|operation|test|execution)",        # Test phases (general)
                r"when\s+(reset|enable|trigger|signal)",             # Trigger conditions (general)
                r"Test case to cover\s+(.*?)(?:\.|$)",               # Test description
                r"after\s+(\w+\s+\w+)",                              # General "after X Y"
                r"during\s+(\w+\s+\w+)",                             # General "during X Y"
                r"when\s+(\w+.*?)(?:\.|,|$)",                        # General "when X..."
                r"step[:\s]*(\d+)[:\s]*(\w+.*)"                      # "step: N operation"
            ]
            for pattern in sequence_patterns:
                match = re.search(pattern, error_report, re.IGNORECASE)
                if match:
                    test_details['sequence_step'] = match.group(1)
                    break

            # Extract full test scenario description
            if "AssertionError" in error_report:
                # Extract more comprehensive context around the failure
                scenario_start = max(0, error_report.find("AssertionError") - 1000)
                scenario_end = min(len(error_report), error_report.find("AssertionError") + 1000)
                test_details['full_scenario'] = error_report[scenario_start:scenario_end]
            else:
                # If no AssertionError, use the full error report
                test_details['full_scenario'] = error_report

        except Exception as e:
            self.logger.warning(f"Failed to extract detailed test case: {e}")

        return test_details

    def analyze_specification_alignment(self, specification: str, test_details: Dict[str, Any]) -> str:
        """
        Analyze specification for potential misunderstandings.

        Implements Specification Review - systematic check of reset behavior,
        timing, interface requirements vs current understanding.
        """
        analysis_points = []

        # Analyze reset behavior
        if "reset" in specification.lower():
            analysis_points.append("**RESET BEHAVIOR CHECK**:")
            analysis_points.append("- When reset is asserted, what should be the initial output?")
            analysis_points.append("- Is reset synchronous or asynchronous?")
            analysis_points.append("- What happens on the first clock after reset deassertion?")
            analysis_points.append("")

        # Analyze initial conditions for any potential initial value issues
        if (test_details.get('test_name', '').lower().find('reset') != -1 or
            test_details.get('test_name', '').lower().find('init') != -1 or
            test_details.get('sequence_step', '').lower().find('reset') != -1):
            analysis_points.append("**INITIAL CONDITIONS CHECK**:")
            analysis_points.append("- What should be the initial output value after reset/initialization?")
            analysis_points.append("- Is there an initialization cycle before normal operation?")
            analysis_points.append("- What are the defined starting values according to specification?")
            analysis_points.append("")

        # Analyze timing requirements
        analysis_points.append("**TIMING REQUIREMENTS CHECK**:")
        analysis_points.append("- Is output supposed to be combinational or registered?")
        analysis_points.append("- Should output change on rising or falling edge?")
        analysis_points.append("- Are there setup/hold requirements for inputs?")
        analysis_points.append("")

        # Analyze interface protocol
        analysis_points.append("**INTERFACE PROTOCOL CHECK**:")
        analysis_points.append("- What exactly triggers the next output value?")
        analysis_points.append("- Is there a valid/ready handshake protocol?")
        analysis_points.append("- Should outputs be held stable or can they glitch?")
        analysis_points.append("")

        return "\n".join(analysis_points)

    def create_test_driven_analysis(self, test_details: Dict[str, Any],
                                  specification: str) -> str:
        """
        Create test-driven debugging analysis.

        Implements Test-Driven Debugging - manual step-through verification
        and minimal reproduction of the failure case.
        """
        analysis = [
            "**TEST-DRIVEN DEBUG APPROACH**:",
            "",
            f"**Failing Test Case**: {test_details['test_name']}",
            f"**Cycle/Time**: {test_details['cycle_number']}",
            f"**Expected**: {test_details['expected_output']}",
            f"**Actual**: {test_details['actual_output']}",
            "",
            "**MANUAL STEP-THROUGH REQUIRED**:",
            "1. **Input Conditions**: What are the exact input values when failure occurs?",
        ]

        # Add input details if available
        if test_details['input_values']:
            analysis.append("   Current inputs:")
            for signal, value in test_details['input_values'].items():
                analysis.append(f"   - {signal} = {value}")
        else:
            analysis.append("   - Identify all input signal values at failure point")

        analysis.extend([
            "",
            "2. **Logic Trace**: Manually trace through your RTL logic:",
            "   - What should happen with these exact inputs?",
            "   - Step through each assignment and conditional",
            "   - Verify combinational vs sequential behavior",
            "",
            "3. **Root Cause Identification**:",
            "   - Is this a timing issue (wrong clock edge)?",
            "   - Is this a logic issue (wrong condition)?",
            "   - Is this an initialization issue (wrong starting state)?",
            "",
            "4. **Minimal Fix Verification**:",
            "   - What scope of change is needed: targeted fix or architectural change?",
            "   - Will this change break other test cases?",
            "   - Does the fix align with specification requirements?",
            "",
            f"**Test Scenario Context**: {test_details['sequence_step']}",
            "",
            "**CRITICAL**: Don't just modify code - first understand WHY the current logic produces the wrong result."
        ])

        return "\n".join(analysis)

    def extract_llm_reasoning(self, response: str) -> Dict[str, Any]:
        """
        Extract and validate LLM reasoning from response to structured prompts.

        This captures whether the LLM actually followed our structured reasoning
        framework and extracts the insights for feedback and learning.
        """
        reasoning_analysis = {
            'followed_framework': False,
            'completed_steps': [],
            'answered_questions': [],
            'identified_root_cause': '',
            'proposed_fix_strategy': '',
            'reasoning_quality': 'unknown'
        }

        try:
            # Check if LLM followed the structured reasoning framework
            step_patterns = [
                r"(?:STEP 1|understand.*failure).*?:(.*?)(?=STEP 2|\*\*|$)",
                r"(?:STEP 2|trace.*logic).*?:(.*?)(?=STEP 3|\*\*|$)",
                r"(?:STEP 3|root.*cause).*?:(.*?)(?=STEP 4|\*\*|$)",
                r"(?:STEP 4|minimal.*fix).*?:(.*?)(?=STEP 5|\*\*|$)",
                r"(?:STEP 5|verify.*fix).*?:(.*?)(?=\*\*|```|$)"
            ]

            for i, pattern in enumerate(step_patterns, 1):
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    step_content = match.group(1).strip()
                    if len(step_content) > 20:  # Meaningful content, not just placeholder
                        reasoning_analysis['completed_steps'].append(f"Step {i}: {step_content}")

            # Check if verification questions were answered
            question_patterns = [
                r"1\.\s*(.*?)(?=2\.|$)",
                r"2\.\s*(.*?)(?=3\.|$)",
                r"3\.\s*(.*?)(?=4\.|$)",
                r"4\.\s*(.*?)(?=5\.|$)",
                r"5\.\s*(.*?)(?=\*\*|```|$)"
            ]

            for i, pattern in enumerate(question_patterns, 1):
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    answer = match.group(1).strip()
                    if len(answer) > 10:  # Meaningful answer
                        reasoning_analysis['answered_questions'].append(f"Q{i}: {answer[:50]}...")

            # Extract root cause identification
            root_cause_patterns = [
                r"root.*cause.*?:(.*?)(?=\*\*|$)",
                r"fundamental.*issue.*?:(.*?)(?=\*\*|$)",
                r"why.*fails?.*?:(.*?)(?=\*\*|$)"
            ]
            for pattern in root_cause_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    reasoning_analysis['identified_root_cause'] = match.group(1).strip()
                    break

            # Extract proposed fix strategy
            fix_patterns = [
                r"fix.*strategy.*?:(.*?)(?=\*\*|```|$)",
                r"solution.*?:(.*?)(?=\*\*|```|$)",
                r"approach.*?:(.*?)(?=\*\*|```|$)"
            ]
            for pattern in fix_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    reasoning_analysis['proposed_fix_strategy'] = match.group(1).strip()
                    break

            # Assess reasoning quality
            if len(reasoning_analysis['completed_steps']) >= 3:
                reasoning_analysis['followed_framework'] = True
                if len(reasoning_analysis['answered_questions']) >= 3:
                    reasoning_analysis['reasoning_quality'] = 'comprehensive'
                else:
                    reasoning_analysis['reasoning_quality'] = 'partial'
            else:
                reasoning_analysis['reasoning_quality'] = 'insufficient'

        except Exception as e:
            self.logger.warning(f"Failed to extract LLM reasoning: {e}")

        return reasoning_analysis

    def validate_extracted_information(self, test_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean extracted test information for quality.

        This ensures extracted information is meaningful and actionable,
        not just noise from test infrastructure.
        """
        cleaned_details = test_details.copy()

        # Clean up test name - remove module prefixes, keep just the test function name
        if '.' in cleaned_details['test_name']:
            cleaned_details['test_name'] = cleaned_details['test_name'].split('.')[-1]

        # Filter input values to keep only meaningful RTL signals
        meaningful_inputs = {}
        if cleaned_details['input_values']:
            for signal, value in cleaned_details['input_values'].items():
                # Keep signal if it looks like actual DUT port or meaningful test value
                if (len(signal) <= 20 and  # Reasonable signal name length
                    not signal.lower() in ['output', 'expected', 'tests', 'pass', 'fail', 'skip', 'sources', 'module'] and
                    not signal.isupper() or signal.lower() in ['rst', 'clk']):  # Keep reset/clock even if uppercase
                    meaningful_inputs[signal] = value
        cleaned_details['input_values'] = meaningful_inputs

        # Improve sequence step description
        if cleaned_details['sequence_step'] and len(cleaned_details['sequence_step']) > 100:
            # If too long, truncate but preserve key information
            cleaned_details['sequence_step'] = cleaned_details['sequence_step']

        return cleaned_details

    def extract_architectural_constraints(self, specification: str, error_report: str,
                                        testbench_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract architectural constraints from specification and test behavior.

        This addresses the core issue where specifications are ambiguous about
        critical architectural decisions (reset style, timing, protocols).
        """
        constraints = {
            'reset_style': 'unknown',
            'reset_timing': 'unknown',
            'clocking_style': 'unknown',
            'output_timing': 'unknown',
            'interface_protocol': 'unknown',
            'constraint_evidence': []
        }

        try:
            # Analyze reset style from test behavior patterns
            if testbench_code or error_report:
                context = f"{testbench_code or ''} {error_report}"

                # Check for async reset indicators
                async_indicators = [
                    r"posedge.*rst.*posedge.*clk",     # Always block sensitivity
                    r"reset.*immediate",               # Immediate reset behavior
                    r"async.*reset"                    # Explicit async mention
                ]

                # Check for sync reset indicators
                sync_indicators = [
                    r"posedge.*clk.*if.*rst",                    # Reset inside clocked block
                    r"rst\.value\s*=\s*0.*RisingEdge",           # Test waits for clock after reset deassert
                    r"Timer.*reset.*RisingEdge",                 # Test waits for clock after reset
                    r"await.*RisingEdge.*after.*reset",          # Explicit clock sync after reset
                    r"sync.*reset",                              # Explicit sync mention
                    r"reset.*next.*cycle"                        # Reset takes effect next cycle
                ]

                for pattern in async_indicators:
                    if re.search(pattern, context, re.IGNORECASE):
                        constraints['reset_style'] = 'asynchronous'
                        constraints['constraint_evidence'].append(f"Found async reset indicator: {pattern}")
                        break

                for pattern in sync_indicators:
                    if re.search(pattern, context, re.IGNORECASE):
                        constraints['reset_style'] = 'synchronous'
                        constraints['constraint_evidence'].append(f"Found sync reset indicator: {pattern}")
                        break

            # Analyze clocking constraints
            if specification:
                if re.search(r"posedge.*clk|rising.*edge.*clock", specification, re.IGNORECASE):
                    constraints['clocking_style'] = 'posedge_clk'
                    constraints['constraint_evidence'].append("Specification mentions positive edge clocking")

                if re.search(r"updated.*each.*clock|every.*clock.*cycle", specification, re.IGNORECASE):
                    constraints['output_timing'] = 'every_cycle'
                    constraints['constraint_evidence'].append("Specification requires output updates every cycle")

            # Analyze timing from error patterns
            if error_report:
                if re.search(r"timing.*error|setup.*violation|hold.*violation", error_report, re.IGNORECASE):
                    constraints['constraint_evidence'].append("Timing violations detected in test")

        except Exception as e:
            self.logger.warning(f"Failed to extract architectural constraints: {e}")

        return constraints

    def _create_architectural_guidance(self, arch_constraints: Dict[str, Any]) -> str:
        """
        Create architectural guidance to prevent inconsistent design decisions.

        This addresses the core systemic issue where LLM makes different
        architectural choices across iterations due to specification ambiguity.
        """
        if not arch_constraints or not arch_constraints.get('constraint_evidence'):
            return ""

        guidance = [
            "**CRITICAL ARCHITECTURAL CONSTRAINTS**:",
            "To prevent inconsistent design decisions across iterations, follow these constraints:",
            ""
        ]

        # Reset style guidance
        if arch_constraints.get('reset_style') != 'unknown':
            guidance.append(f"**Reset Style**: Use {arch_constraints['reset_style']} reset")
            if arch_constraints['reset_style'] == 'synchronous':
                guidance.append("- `always @(posedge clk)` with `if (rst)` inside")
            elif arch_constraints['reset_style'] == 'asynchronous':
                guidance.append("- `always @(posedge clk or posedge rst)`")
            guidance.append("")

        # Clocking style guidance
        if arch_constraints.get('clocking_style') != 'unknown':
            guidance.append(f"**Clocking**: {arch_constraints['clocking_style']}")
            guidance.append("")

        # Output timing guidance
        if arch_constraints.get('output_timing') != 'unknown':
            guidance.append(f"**Output Timing**: {arch_constraints['output_timing']}")
            guidance.append("")

        # Add evidence
        if arch_constraints.get('constraint_evidence'):
            guidance.append("**Evidence for these constraints**:")
            for evidence in arch_constraints['constraint_evidence']:
                guidance.append(f"- {evidence}")
            guidance.append("")

        guidance.append("**CONSISTENCY REQUIREMENT**: Use the SAME architectural decisions as successful approaches")

        return "\n".join(guidance)

    def _attempt_testbench_extraction(self, error_report: str) -> Optional[str]:
        """
        Extract testbench code patterns from error reports to understand architectural constraints.

        This helps identify critical architectural decisions (reset style, timing) from
        test behavior rather than relying solely on potentially ambiguous specifications.
        """
        try:
            # Extract test function code if available in error traceback
            test_code_patterns = [
                r"File\s+\"[^\"]*test[^\"]*\.py\",.*?in\s+(\w+).*?\n(.*?)(?=File|\n\n|$)",
                r"async def\s+(test_\w+).*?:\s*(.*?)(?=async def|\n\n|$)",
                r"def\s+(test_\w+).*?:\s*(.*?)(?=def|\n\n|$)"
            ]

            extracted_code = ""
            for pattern in test_code_patterns:
                matches = re.finditer(pattern, error_report, re.DOTALL | re.MULTILINE)
                for match in matches:
                    if len(match.groups()) >= 2:
                        extracted_code += f"Function {match.group(1)}: {match.group(2)}\n"

            # Look for reset-related test patterns specifically
            reset_patterns = [
                r"(dut\.rst\.value.*?)(?=\n.*?await|\n.*?dut)",
                r"(await.*reset.*?)(?=\n.*?await|\n.*?dut)",
                r"(Timer.*reset.*?)(?=\n.*?await|\n.*?dut)",
                r"(RisingEdge.*reset.*?)(?=\n|$)"
            ]

            for pattern in reset_patterns:
                matches = re.findall(pattern, error_report, re.IGNORECASE)
                if matches:
                    extracted_code += f"Reset behavior: {', '.join(matches[:3])}\n"

            return extracted_code if extracted_code.strip() else None

        except Exception as e:
            self.logger.warning(f"Failed to extract testbench patterns: {e}")
            return None

    def _create_enhanced_contextual_prompt(self, specification: str, current_code: str,
                                         error_report: str, context_analysis: Dict,
                                         enhanced_context: str = "", test_details: Dict[str, Any] = None,
                                         arch_constraints: Dict[str, Any] = None) -> str:
        """Create enhanced contextual analysis prompt with deep debugging."""

        if not test_details:
            test_details = self.extract_detailed_test_case(error_report)
            test_details = self.validate_extracted_information(test_details)

        spec_analysis = self.analyze_specification_alignment(specification, test_details)
        test_driven_analysis = self.create_test_driven_analysis(test_details, specification)
        arch_guidance = self._create_architectural_guidance(arch_constraints) if arch_constraints else ""

        enhanced_section = f"\n{enhanced_context}\n" if enhanced_context else ""

        prompt = f"""Debug this RTL issue with comprehensive enhanced analysis:

**DETAILED TEST FAILURE ANALYSIS**:
- Test: {test_details['test_name']} at cycle {test_details['cycle_number']}
- Failed Input State: {test_details['input_values']}
- Expected: {test_details['expected_output']}, Actual: {test_details['actual_output']}
- Sequence Context: {test_details['sequence_step']}

{spec_analysis}

{test_driven_analysis}

{arch_guidance}

**Complete Error Report**:
```
{error_report}
```

**Specification**:
```
{specification}
```
{enhanced_section}
**Current Code**:
```systemverilog
{current_code}
```

**MANDATORY STRUCTURED REASONING**:
You MUST complete ALL steps systematically:

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[0]}**:
- Document the exact failure scenario with complete context
- Identify the specific test case and timing that failed

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[1]}**:
- Trace through your RTL logic step-by-step for the failing inputs
- Document each signal transition and logic decision

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[2]}**:
- Identify the fundamental root cause (timing, logic, or specification issue)
- Explain exactly WHY your current implementation fails

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[3]}**:
- Design an appropriately-scoped fix that addresses the root cause
- If root cause is architectural/timing: use bold structural changes
- If root cause is surface logic: use targeted fixes
- Focus on effectiveness, not minimalism

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[4]}**:
- Verify your fix resolves the failing test case
- Ensure your fix doesn't introduce new problems

**Answer Each Verification Question**:
1. {ReasoningFramework.VERIFICATION_QUESTIONS[0]}
2. {ReasoningFramework.VERIFICATION_QUESTIONS[1]}
3. {ReasoningFramework.VERIFICATION_QUESTIONS[2]}
4. {ReasoningFramework.VERIFICATION_QUESTIONS[3]}
5. {ReasoningFramework.VERIFICATION_QUESTIONS[4]}

**IMPLEMENT ONLY AFTER COMPLETE REASONING**:

```systemverilog
[Your corrected code]
```
"""
        return prompt

    def _create_enhanced_systematic_prompt(self, specification: str, current_code: str,
                                         error_report: str, enhanced_context: str = "",
                                         test_details: Dict[str, Any] = None,
                                         arch_constraints: Dict[str, Any] = None) -> str:
        """Create enhanced systematic debugging prompt with comprehensive analysis."""

        if not test_details:
            test_details = self.extract_detailed_test_case(error_report)
            test_details = self.validate_extracted_information(test_details)

        spec_analysis = self.analyze_specification_alignment(specification, test_details)
        test_driven_analysis = self.create_test_driven_analysis(test_details, specification)
        arch_guidance = self._create_architectural_guidance(arch_constraints) if arch_constraints else ""

        enhanced_section = f"\n{enhanced_context}\n" if enhanced_context else ""

        prompt = f"""Use systematic debugging with comprehensive analysis for this complex RTL issue:

**COMPREHENSIVE FAILURE CONTEXT**:
- Failed Test: {test_details['test_name']}
- Timing: {test_details['cycle_number']}
- Input Conditions: {test_details['input_values']}
- Expected vs Actual: {test_details['expected_output']} vs {test_details['actual_output']}
- Test Scenario: {test_details['full_scenario']}

{spec_analysis}

{test_driven_analysis}

{arch_guidance}

**Complete Context**:
- Specification: {specification}
- Error Report: {error_report}
- Current Code: {len(current_code)} lines
{enhanced_section}

**MANDATORY COMPREHENSIVE REASONING**:
You MUST systematically complete ALL analysis steps:

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[0]}**:
- Document the complete failure scenario with all context
- Identify ALL failing test cases and their relationships

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[1]}**:
- Manually trace through your RTL logic for the failing scenario
- Document each signal value at each clock cycle
- Identify exactly where the logic diverges from expectations

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[2]}**:
- Determine if this is: timing issue, logic flaw, or specification misunderstanding
- Identify the fundamental architectural or implementation problem
- Explain WHY your current approach fails

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[3]}**:
- Design a targeted solution that addresses the root cause
- Ensure the fix is minimal and doesn't introduce new issues
- Verify the fix addresses specification requirements

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[4]}**:
- Verify your fix handles the failing scenario correctly
- Check that your fix doesn't break other test scenarios
- Confirm alignment with all specification requirements

**Critical Verification** (Answer each question):
1. {ReasoningFramework.VERIFICATION_QUESTIONS[0]}
2. {ReasoningFramework.VERIFICATION_QUESTIONS[1]}
3. {ReasoningFramework.VERIFICATION_QUESTIONS[2]}
4. {ReasoningFramework.VERIFICATION_QUESTIONS[3]}
5. {ReasoningFramework.VERIFICATION_QUESTIONS[4]}

**IMPLEMENT ONLY AFTER COMPLETE ANALYSIS**:

```systemverilog
[Your corrected or redesigned code]
```
"""
        return prompt
