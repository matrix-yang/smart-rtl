# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Systematic RTL Debugger - Advanced forensic debugging with triage and systematic correction
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

from .utils.llm_client import LLMClient
from .progressive_debugging_strategy import ProgressiveDebuggingStrategy, DebuggingApproach, ReasoningFramework


class DebugStrategy(Enum):
    """Debug strategy determined by triage."""
    INCREMENTAL_FIX = "incremental_fix"
    COMPLETE_REDESIGN = "complete_redesign"
    FORENSIC_ANALYSIS_REQUIRED = "forensic_analysis_required"


@dataclass
class TriageResult:
    """Result from triage analysis."""
    strategy: DebugStrategy
    confidence: str  # HIGH, MEDIUM, LOW
    reasoning: str
    key_issues: List[str]
    next_steps: str


@dataclass
class ForensicAnalysis:
    """Result from forensic test analysis."""
    error_pattern_summary: str
    timing_analysis: str
    value_analysis: str
    sequence_analysis: str
    reset_analysis: str
    root_cause_hypothesis: str
    supporting_evidence: List[str]
    alternative_hypotheses: List[str]
    recommended_debug_strategy: str
    confidence: str


@dataclass
class DebuggingHistory:
    """Track debugging attempts and outcomes."""
    iteration: int
    strategy_used: DebugStrategy
    triage_result: Optional[TriageResult]
    forensic_analysis: Optional[ForensicAnalysis]
    rtl_code: str
    test_passed: bool
    error_report: str
    prompt_used: str
    reasoning: str


class SystematicRTLDebugger:
    """
    Advanced RTL debugger using systematic forensic debugging techniques.

    This class implements a 3-phase debugging approach:
    1. Triage: Decide whether to fix incrementally or redesign from scratch
    2. Forensic Analysis: Deep analysis of test patterns for complex failures
    3. Systematic Correction: Targeted fixes based on root cause analysis
    """

    def __init__(self, model_name: str, logger, api_key: Optional[str] = None,
                 max_retries: int = 3, initial_backoff: float = 5.0):
        """
        Initialize the Systematic RTL Debugger.

        Args:
            model_name: Name of the LLM model to use
            logger: Logger instance
            api_key: Optional LLM API key
            max_retries: Maximum number of retries for API calls
            initial_backoff: Initial backoff time in seconds
        """
        self.model_name = model_name
        self.logger = logger

        # Initialize robust LLM client
        self.client = LLMClient(
            model=model_name,
            api_key=api_key,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            logger=logger
        )

        # Initialize enhanced debugging strategy with structured reasoning
        self.debugging_strategy = ProgressiveDebuggingStrategy(logger)

        # Track debugging history across iterations
        self.debugging_history: List[DebuggingHistory] = []

        self.logger.info(f"Initialized SystematicRTLDebugger with model: {model_name}")

    def debug_rtl_systematically(self, specification: str, buggy_code: str,
                                error_report: str, iteration: int) -> Tuple[str, str, Dict[str, Any]]:
        """
        Main entry point for systematic RTL debugging.

        Args:
            specification: Original problem specification
            buggy_code: RTL code that failed testing
            error_report: Detailed error report from test failure
            iteration: Current iteration number

        Returns:
            Tuple of (corrected_rtl_code, prompt_used, debug_metadata)
        """
        self.logger.info(f"Starting systematic debugging for iteration {iteration}")

        try:
            # Step 1: Triage - decide debugging strategy
            triage_result = self._perform_triage(specification, buggy_code, error_report)
            self.logger.info(f"Triage decision: {triage_result.strategy.value} (confidence: {triage_result.confidence})")

            # Step 2: Apply appropriate debugging strategy
            if triage_result.strategy == DebugStrategy.COMPLETE_REDESIGN:
                return self._systematic_redesign(specification, triage_result, iteration)

            elif triage_result.strategy == DebugStrategy.FORENSIC_ANALYSIS_REQUIRED:
                # Perform forensic analysis first, then systematic correction
                forensic_analysis = self._perform_forensic_analysis(error_report)
                return self._systematic_correction_with_forensics(
                    specification, buggy_code, error_report, triage_result,
                    forensic_analysis, iteration
                )

            else:  # INCREMENTAL_FIX
                return self._systematic_correction(
                    specification, buggy_code, error_report, triage_result, iteration
                )

        except Exception as e:
            self.logger.error(f"Error in systematic debugging: {str(e)}")
            raise

    def _perform_triage(self, specification: str, buggy_code: str,
                       error_report: str) -> TriageResult:
        """Perform triage to determine debugging strategy."""
        self.logger.info("Performing triage analysis...")

        try:
            # Prepare previous attempts context
            previous_attempts = self._format_previous_attempts()

            # Generate enhanced triage prompt with structured reasoning
            prompt = self._create_structured_triage_prompt(
                specification, buggy_code, error_report, previous_attempts
            )

            # Call LLM for triage analysis
            response = self.client.call_llm(prompt)
            if not response:
                raise ValueError("Failed to get triage analysis from LLM")

            # Parse triage response
            triage_result = self._parse_triage_response(response)

            self.logger.info(f"Triage completed: {triage_result.strategy.value}")
            return triage_result

        except Exception as e:
            self.logger.error(f"Triage analysis failed: {str(e)}")
            # Default to incremental fix if triage fails
            return TriageResult(
                strategy=DebugStrategy.INCREMENTAL_FIX,
                confidence="LOW",
                reasoning=f"Triage analysis failed: {str(e)}, defaulting to incremental fix",
                key_issues=["Triage analysis unavailable"],
                next_steps="Proceed with incremental debugging"
            )

    def _perform_forensic_analysis(self, error_report: str) -> ForensicAnalysis:
        """Perform detailed forensic analysis of test failures."""
        self.logger.info("Performing forensic test analysis...")

        try:
            # Generate enhanced forensic analysis prompt with structured reasoning
            prompt = self._create_structured_forensic_prompt(error_report)

            # Call LLM for forensic analysis
            response = self.client.call_llm(prompt)
            if not response:
                raise ValueError("Failed to get forensic analysis from LLM")

            # Parse forensic analysis response
            forensic_result = self._parse_forensic_response(response)

            self.logger.info("Forensic analysis completed")
            return forensic_result

        except Exception as e:
            self.logger.error(f"Forensic analysis failed: {str(e)}")
            # Return basic forensic analysis if parsing fails
            return ForensicAnalysis(
                error_pattern_summary="Forensic analysis failed",
                timing_analysis="Analysis unavailable",
                value_analysis="Analysis unavailable",
                sequence_analysis="Analysis unavailable",
                reset_analysis="Analysis unavailable",
                root_cause_hypothesis=f"Analysis failed: {str(e)}",
                supporting_evidence=[],
                alternative_hypotheses=[],
                recommended_debug_strategy="Proceed with basic error correction",
                confidence="LOW"
            )

    def _systematic_correction(self, specification: str, buggy_code: str,
                             error_report: str, triage_result: TriageResult,
                             iteration: int) -> Tuple[str, str, Dict[str, Any]]:
        """Perform systematic error correction."""
        self.logger.info("Performing systematic error correction...")

        try:
            # Prepare previous attempts context
            previous_attempts = self._format_previous_attempts()

            # Generate systematic correction prompt using enhanced debugging strategy
            prompt = self.debugging_strategy.create_approach_prompt(
                DebuggingApproach.SYSTEMATIC_DEBUGGING,
                specification=specification,
                current_code=buggy_code,
                error_report=error_report,
                enhanced_context=f"**PREVIOUS ATTEMPTS**: {previous_attempts}"
            )

            # Call LLM for systematic correction
            response = self.client.call_llm(prompt)
            if not response:
                raise ValueError("Failed to get systematic correction from LLM")

            # Extract and validate LLM reasoning
            reasoning_analysis = self.debugging_strategy.extract_llm_reasoning(response)
            self.logger.info(f"Systematic debugging - LLM reasoning quality: {reasoning_analysis['reasoning_quality']}")
            if not reasoning_analysis['followed_framework']:
                self.logger.warning("LLM did not follow structured reasoning framework in systematic debugging")

            # Extract corrected RTL code
            corrected_rtl = self._extract_rtl_code(response)

            # Record debugging attempt
            debug_history = DebuggingHistory(
                iteration=iteration,
                strategy_used=DebugStrategy.INCREMENTAL_FIX,
                triage_result=triage_result,
                forensic_analysis=None,
                rtl_code=corrected_rtl,
                test_passed=False,  # Will be updated after testing
                error_report=error_report,
                prompt_used=prompt,
                reasoning="Systematic incremental correction based on triage analysis"
            )
            self.debugging_history.append(debug_history)

            debug_metadata = {
                "triage_result": asdict(triage_result),
                "forensic_analysis": None,
                "strategy_used": DebugStrategy.INCREMENTAL_FIX.value,
                "reasoning": debug_history.reasoning,
                "llm_reasoning_analysis": reasoning_analysis,
                "reasoning_compliance": {
                    "followed_framework": reasoning_analysis['followed_framework'],
                    "reasoning_quality": reasoning_analysis['reasoning_quality'],
                    "root_cause_identified": bool(reasoning_analysis['identified_root_cause']),
                    "steps_completed": len(reasoning_analysis['completed_steps'])
                }
            }

            self.logger.info(f"Systematic correction completed ({len(corrected_rtl)} chars)")
            return corrected_rtl, prompt, "", debug_metadata  # Empty response for systematic debugger

        except Exception as e:
            self.logger.error(f"Systematic correction failed: {str(e)}")
            raise

    def _systematic_correction_with_forensics(self, specification: str, buggy_code: str,
                                            error_report: str, triage_result: TriageResult,
                                            forensic_analysis: ForensicAnalysis,
                                            iteration: int) -> Tuple[str, str, Dict[str, Any]]:
        """Perform systematic correction enhanced with forensic analysis."""
        self.logger.info("Performing systematic correction with forensic analysis...")

        try:
            # Prepare previous attempts context
            previous_attempts = self._format_previous_attempts()

            # Enhance error report with forensic insights
            enhanced_error_report = f"""
{error_report}

## FORENSIC ANALYSIS INSIGHTS:
**Root Cause Hypothesis:** {forensic_analysis.root_cause_hypothesis}
**Error Pattern:** {forensic_analysis.error_pattern_summary}
**Supporting Evidence:** {', '.join(forensic_analysis.supporting_evidence)}
**Recommended Strategy:** {forensic_analysis.recommended_debug_strategy}
"""

            # Generate systematic correction prompt using enhanced debugging strategy
            prompt = self.debugging_strategy.create_approach_prompt(
                DebuggingApproach.SYSTEMATIC_DEBUGGING,
                specification=specification,
                current_code=buggy_code,
                error_report=enhanced_error_report,
                enhanced_context=f"**PREVIOUS ATTEMPTS**: {previous_attempts}"
            )

            # Call LLM for forensic-enhanced correction
            response = self.client.call_llm(prompt)
            if not response:
                raise ValueError("Failed to get forensic-enhanced correction from LLM")

            # Extract and validate LLM reasoning
            reasoning_analysis = self.debugging_strategy.extract_llm_reasoning(response)
            self.logger.info(f"Forensic debugging - LLM reasoning quality: {reasoning_analysis['reasoning_quality']}")
            if not reasoning_analysis['followed_framework']:
                self.logger.warning("LLM did not follow structured reasoning framework in forensic debugging")

            # Extract corrected RTL code
            corrected_rtl = self._extract_rtl_code(response)

            # Record debugging attempt
            debug_history = DebuggingHistory(
                iteration=iteration,
                strategy_used=DebugStrategy.FORENSIC_ANALYSIS_REQUIRED,
                triage_result=triage_result,
                forensic_analysis=forensic_analysis,
                rtl_code=corrected_rtl,
                test_passed=False,  # Will be updated after testing
                error_report=error_report,
                prompt_used=prompt,
                reasoning="Systematic correction enhanced with forensic test analysis"
            )
            self.debugging_history.append(debug_history)

            debug_metadata = {
                "triage_result": asdict(triage_result),
                "forensic_analysis": asdict(forensic_analysis),
                "strategy_used": DebugStrategy.FORENSIC_ANALYSIS_REQUIRED.value,
                "reasoning": debug_history.reasoning,
                "llm_reasoning_analysis": reasoning_analysis,
                "reasoning_compliance": {
                    "followed_framework": reasoning_analysis['followed_framework'],
                    "reasoning_quality": reasoning_analysis['reasoning_quality'],
                    "root_cause_identified": bool(reasoning_analysis['identified_root_cause']),
                    "steps_completed": len(reasoning_analysis['completed_steps'])
                }
            }

            self.logger.info(f"Forensic-enhanced correction completed ({len(corrected_rtl)} chars)")
            return corrected_rtl, prompt, "", debug_metadata  # Empty response for systematic debugger

        except Exception as e:
            self.logger.error(f"Forensic-enhanced correction failed: {str(e)}")
            raise

    def _systematic_redesign(self, specification: str, triage_result: TriageResult,
                           iteration: int) -> Tuple[str, str, Dict[str, Any]]:
        """Perform complete systematic redesign from scratch."""
        self.logger.info("Performing systematic redesign from scratch...")

        try:
            # Generate systematic redesign prompt using enhanced debugging strategy
            # For complete redesign, create a comprehensive prompt with all context
            redesign_context = f"""
**REDESIGN TRIGGER**: Complete architectural rethinking required
**PREVIOUS FAILURES**: Multiple debugging attempts have failed, suggesting fundamental design issues
**APPROACH**: Start fresh with systematic specification analysis
"""
            prompt = self.debugging_strategy.create_approach_prompt(
                DebuggingApproach.SYSTEMATIC_DEBUGGING,
                specification=specification,
                current_code="// Complete redesign - analyzing specification for fresh approach",
                error_report="Complete redesign - systematic rethinking of architecture and requirements",
                enhanced_context=redesign_context
            )

            # Call LLM for systematic redesign
            response = self.client.call_llm(prompt)
            if not response:
                raise ValueError("Failed to get systematic redesign from LLM")

            # Extract and validate LLM reasoning for redesign
            reasoning_analysis = self.debugging_strategy.extract_llm_reasoning(response)
            self.logger.info(f"Systematic redesign - LLM reasoning quality: {reasoning_analysis['reasoning_quality']}")
            if not reasoning_analysis['followed_framework']:
                self.logger.warning("LLM did not follow structured reasoning framework in systematic redesign")

            # Extract new RTL code
            new_rtl = self._extract_rtl_code(response)

            # Record debugging attempt
            debug_history = DebuggingHistory(
                iteration=iteration,
                strategy_used=DebugStrategy.COMPLETE_REDESIGN,
                triage_result=triage_result,
                forensic_analysis=None,
                rtl_code=new_rtl,
                test_passed=False,  # Will be updated after testing
                error_report="Complete redesign - no specific error report",
                prompt_used=prompt,
                reasoning="Complete systematic redesign based on triage analysis"
            )
            self.debugging_history.append(debug_history)

            debug_metadata = {
                "triage_result": asdict(triage_result),
                "forensic_analysis": None,
                "strategy_used": DebugStrategy.COMPLETE_REDESIGN.value,
                "reasoning": debug_history.reasoning,
                "llm_reasoning_analysis": reasoning_analysis,
                "reasoning_compliance": {
                    "followed_framework": reasoning_analysis['followed_framework'],
                    "reasoning_quality": reasoning_analysis['reasoning_quality'],
                    "root_cause_identified": bool(reasoning_analysis['identified_root_cause']),
                    "steps_completed": len(reasoning_analysis['completed_steps'])
                }
            }

            self.logger.info(f"Systematic redesign completed ({len(new_rtl)} chars)")
            return new_rtl, prompt, "", debug_metadata  # Empty response for systematic debugger

        except Exception as e:
            self.logger.error(f"Systematic redesign failed: {str(e)}")
            raise

    def _parse_triage_response(self, response: str) -> TriageResult:
        """Parse triage analysis response from LLM."""
        try:
            # Extract recommendation
            recommendation_match = re.search(r'\*\*RECOMMENDATION:\*\*\s*\[?(\w+)\]?', response, re.IGNORECASE)
            if recommendation_match:
                recommendation = recommendation_match.group(1).upper()
                if "REDESIGN" in recommendation:
                    strategy = DebugStrategy.COMPLETE_REDESIGN
                elif "FIX" in recommendation:
                    strategy = DebugStrategy.INCREMENTAL_FIX
                else:
                    strategy = DebugStrategy.INCREMENTAL_FIX  # Default
            else:
                strategy = DebugStrategy.INCREMENTAL_FIX

            # Extract confidence
            confidence_match = re.search(r'\*\*CONFIDENCE LEVEL:\*\*\s*\[?(\w+)\]?', response, re.IGNORECASE)
            confidence = confidence_match.group(1).upper() if confidence_match else "MEDIUM"

            # Extract reasoning
            reasoning_match = re.search(r'\*\*REASONING:\*\*(.*?)(?=\*\*|$)', response, re.DOTALL | re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "Reasoning not extracted"

            # Extract key issues
            issues_section = re.search(r'\*\*KEY ISSUES IDENTIFIED:\*\*(.*?)(?=\*\*|$)', response, re.DOTALL | re.IGNORECASE)
            key_issues = []
            if issues_section:
                # Look for numbered or bulleted lists
                issue_lines = re.findall(r'(?:\d+\.|[-*])\s*(.+)', issues_section.group(1))
                key_issues = [issue.strip() for issue in issue_lines]

            # Extract next steps
            next_steps_match = re.search(r'\*\*NEXT STEPS:\*\*(.*?)(?=\*\*|$)', response, re.DOTALL | re.IGNORECASE)
            next_steps = next_steps_match.group(1).strip() if next_steps_match else "Next steps not specified"

            # Check if forensic analysis is recommended based on keywords
            if any(keyword in response.lower() for keyword in ['forensic', 'complex', 'pattern', 'deep analysis']):
                if strategy == DebugStrategy.INCREMENTAL_FIX:
                    strategy = DebugStrategy.FORENSIC_ANALYSIS_REQUIRED

            return TriageResult(
                strategy=strategy,
                confidence=confidence,
                reasoning=reasoning,
                key_issues=key_issues,
                next_steps=next_steps
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse triage response: {str(e)}")
            # Return default triage result
            return TriageResult(
                strategy=DebugStrategy.INCREMENTAL_FIX,
                confidence="LOW",
                reasoning="Failed to parse triage response",
                key_issues=["Triage parsing failed"],
                next_steps="Proceed with incremental debugging"
            )

    def _parse_forensic_response(self, response: str) -> ForensicAnalysis:
        """Parse forensic analysis response from LLM."""
        try:
            # Helper function to extract section content
            def extract_section(pattern: str, default: str = "Not specified") -> str:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                return match.group(1).strip() if match else default

            # Extract main sections
            error_pattern_summary = extract_section(r'\*\*ERROR PATTERN SUMMARY:\*\*(.*?)(?=\*\*|$)')

            # Extract detailed analysis
            timing_analysis = extract_section(r'- Timing Analysis:\s*(.*?)(?=\n\s*-|\*\*|$)')
            value_analysis = extract_section(r'- Value Analysis:\s*(.*?)(?=\n\s*-|\*\*|$)')
            sequence_analysis = extract_section(r'- Sequence Analysis:\s*(.*?)(?=\n\s*-|\*\*|$)')
            reset_analysis = extract_section(r'- Reset Analysis:\s*(.*?)(?=\n\s*-|\*\*|$)')

            root_cause_hypothesis = extract_section(r'\*\*ROOT CAUSE HYPOTHESIS:\*\*(.*?)(?=\*\*|$)')

            # Extract supporting evidence
            evidence_section = extract_section(r'\*\*SUPPORTING EVIDENCE:\*\*(.*?)(?=\*\*|$)')
            supporting_evidence = []
            if evidence_section != "Not specified":
                evidence_lines = re.findall(r'(?:\d+\.|[-*])\s*(.+)', evidence_section)
                supporting_evidence = [evidence.strip() for evidence in evidence_lines]

            # Extract alternative hypotheses
            alt_hypotheses = extract_section(r'\*\*ALTERNATIVE HYPOTHESES:\*\*(.*?)(?=\*\*|$)')
            alternative_hypotheses = [alt_hypotheses] if alt_hypotheses != "Not specified" else []

            recommended_debug_strategy = extract_section(r'\*\*RECOMMENDED DEBUG STRATEGY:\*\*(.*?)(?=\*\*|$)')

            confidence_match = re.search(r'\*\*CONFIDENCE LEVEL:\*\*\s*\[?(\w+)\]?', response, re.IGNORECASE)
            confidence = confidence_match.group(1).upper() if confidence_match else "MEDIUM"

            return ForensicAnalysis(
                error_pattern_summary=error_pattern_summary,
                timing_analysis=timing_analysis,
                value_analysis=value_analysis,
                sequence_analysis=sequence_analysis,
                reset_analysis=reset_analysis,
                root_cause_hypothesis=root_cause_hypothesis,
                supporting_evidence=supporting_evidence,
                alternative_hypotheses=alternative_hypotheses,
                recommended_debug_strategy=recommended_debug_strategy,
                confidence=confidence
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse forensic response: {str(e)}")
            # Return basic forensic analysis
            return ForensicAnalysis(
                error_pattern_summary="Failed to parse forensic analysis",
                timing_analysis="Analysis unavailable",
                value_analysis="Analysis unavailable",
                sequence_analysis="Analysis unavailable",
                reset_analysis="Analysis unavailable",
                root_cause_hypothesis="Parsing failed - proceed with basic debugging",
                supporting_evidence=[],
                alternative_hypotheses=[],
                recommended_debug_strategy="Use basic error correction approach",
                confidence="LOW"
            )

    def _extract_rtl_code(self, response: str) -> str:
        """Extract SystemVerilog code from LLM response."""
        # Look for code blocks with systemverilog, verilog, or no language specified
        patterns = [
            r'```systemverilog\s*\n(.*?)\n```',
            r'```verilog\s*\n(.*?)\n```',
            r'```sv\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
            if matches:
                code = matches[0].strip()
                if self._is_valid_rtl(code):
                    return code

        # If no code blocks found, try to extract based on module keywords
        lines = response.split('\n')
        module_start = None
        module_end = None

        for i, line in enumerate(lines):
            if re.match(r'\s*module\s+\w+', line):
                module_start = i
            if module_start is not None and re.match(r'\s*endmodule', line):
                module_end = i
                break

        if module_start is not None and module_end is not None:
            code = '\n'.join(lines[module_start:module_end+1])
            if self._is_valid_rtl(code):
                return code

        # If still no valid code found, log warning and return response as-is
        self.logger.warning("Could not extract valid RTL code from response, returning full response")
        return response.strip()

    def _is_valid_rtl(self, code: str) -> bool:
        """Basic validation to check if extracted code looks like valid RTL."""
        # Check for module declaration
        if not re.search(r'\bmodule\s+\w+', code):
            return False

        # Check for endmodule
        if 'endmodule' not in code:
            return False

        # Check if it's not just text/comments
        if len(re.sub(r'//.*', '', code).strip()) < 20:
            return False

        return True

    def _format_previous_attempts(self) -> str:
        """Format previous debugging attempts for context."""
        if not self.debugging_history:
            return "No previous attempts"

        attempts = []
        # Use all debugging history, not just last 3 - need complete picture for analysis
        for history in self.debugging_history:
            attempt_info = f"Iteration {history.iteration}: {history.strategy_used.value} - "
            attempt_info += "PASSED" if history.test_passed else "FAILED"
            if history.triage_result:
                attempt_info += f" (Triage: {history.triage_result.reasoning})"
            attempts.append(attempt_info)

        return "\n".join(attempts)

    def update_iteration_result(self, iteration: int, test_passed: bool):
        """Update the test result for a specific iteration."""
        for history in self.debugging_history:
            if history.iteration == iteration:
                history.test_passed = test_passed
                break

    def get_debugging_summary(self) -> Dict[str, Any]:
        """Get a summary of all debugging attempts."""
        return {
            "total_iterations": len(self.debugging_history),
            "strategies_used": [h.strategy_used.value for h in self.debugging_history],
            "successful_iterations": [h.iteration for h in self.debugging_history if h.test_passed],
            "detailed_history": [
                {
                    "iteration": h.iteration,
                    "strategy": h.strategy_used.value,
                    "passed": h.test_passed,
                    "reasoning": h.reasoning,
                    "triage_result": asdict(h.triage_result) if h.triage_result else None,
                    "forensic_analysis": asdict(h.forensic_analysis) if h.forensic_analysis else None
                }
                for h in self.debugging_history
            ]
        }

    # === STRUCTURED REASONING METHODS ===

    def _create_structured_triage_prompt(self, specification: str, buggy_code: str,
                                       error_report: str, previous_attempts: str) -> str:
        """Create triage prompt using structured reasoning framework."""

        # Extract test details for better triage analysis
        test_details = self.debugging_strategy.extract_detailed_test_case(error_report)
        test_details = self.debugging_strategy.validate_extracted_information(test_details)

        prompt = f"""Perform systematic triage analysis using structured reasoning:

**MANDATORY TRIAGE ANALYSIS STEPS**:

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[0]}**: Understand the Complete Failure
- Test Case: {test_details['test_name']}
- Timing: {test_details['cycle_number']}
- Expected vs Actual: {test_details['expected_output']} vs {test_details['actual_output']}
- Context: {test_details['sequence_step']}

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[1]}**: Analyze Current Implementation
- Trace through the RTL logic that produces the wrong result
- Identify which part of the code causes the failure

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[2]}**: Determine Root Cause Category
- Is this a fundamental architectural flaw requiring COMPLETE_REDESIGN?
- Is this a complex timing/protocol issue needing FORENSIC_ANALYSIS_REQUIRED?
- Is this a targeted logic bug suitable for INCREMENTAL_FIX?

**Previous Debugging Context**:
{previous_attempts}

**Specification**:
{specification}

**Current Code** ({len(buggy_code)} chars):
```systemverilog
{buggy_code}
```

**Error Report**:
{error_report}

**SYSTEMATIC TRIAGE DECISION**:
Based on your structured analysis, select the debugging strategy:

**RECOMMENDATION**: [INCREMENTAL_FIX | COMPLETE_REDESIGN | FORENSIC_ANALYSIS_REQUIRED]
**CONFIDENCE**: [HIGH | MEDIUM | LOW]
**REASONING**: [Explain your systematic analysis and decision]
**KEY ISSUES**: [List 2-3 specific technical issues identified]
**NEXT STEPS**: [Specific actions needed for chosen strategy]
"""
        return prompt

    def _create_structured_forensic_prompt(self, error_report: str) -> str:
        """Create forensic analysis prompt using structured reasoning framework."""

        # Extract test details for forensic analysis
        test_details = self.debugging_strategy.extract_detailed_test_case(error_report)
        test_details = self.debugging_strategy.validate_extracted_information(test_details)

        prompt = f"""Perform systematic forensic analysis of test failure:

**MANDATORY FORENSIC ANALYSIS STEPS**:

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[0]}**: Document Complete Failure Scenario
- Failed Test: {test_details['test_name']}
- Failure Timing: {test_details['cycle_number']}
- Error Values: Expected {test_details['expected_output']}, Got {test_details['actual_output']}
- Test Context: {test_details['sequence_step']}

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[1]}**: Trace Test Execution
- What was the test doing when it failed?
- What sequence of operations led to this failure?
- What input conditions triggered the wrong behavior?

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[2]}**: Identify Failure Pattern
- Is this a consistent failure or intermittent?
- Does this represent a fundamental design flaw?
- What category of RTL issue does this represent?

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[3]}**: Hypothesize Root Cause
- What is the most likely underlying cause?
- What evidence supports this hypothesis?
- What debugging strategy would be most effective?

**{ReasoningFramework.STEP_BY_STEP_ANALYSIS[4]}**: Recommend Debug Strategy
- Should this use incremental fixes or complete redesign?
- What specific debugging steps are needed?
- What evidence would confirm or refute the hypothesis?

**Test Error Report**:
{error_report}

**SYSTEMATIC FORENSIC ANALYSIS**:
- **ROOT CAUSE HYPOTHESIS**: [Your systematic hypothesis based on evidence]
- **ERROR PATTERN SUMMARY**: [Concise description of the failure pattern]
- **SUPPORTING EVIDENCE**: [Key evidence that supports your hypothesis]
- **RECOMMENDED DEBUG STRATEGY**: [Specific strategy for addressing this issue]
"""
        return prompt
