# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Primary Failure Extractor - Isolates the FIRST assertion failure for focused debugging
"""

import re
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class PrimaryFailure:
    """The first/primary assertion failure in an error report."""
    expected_value: str
    actual_value: str
    timing_context: str
    raw_assertion: str
    context_lines: str

class PrimaryFailureExtractor:
    """
    Extracts the PRIMARY assertion failure from error reports.

    This implements the key debugging pattern: focus on ONE assertion failure
    at a time instead of trying to analyze multiple issues simultaneously.
    """

    def __init__(self, logger, model_name: str = "nvidia/nemotron-3-ultra-550b-a55b", llm_context_limit: int = 300000):
        self.logger = logger
        self.model_name = model_name
        self.llm_context_limit = llm_context_limit

    def extract_primary_failure(self, error_report: str) -> Optional[PrimaryFailure]:
        """
        Extract the FIRST assertion failure from error report.

        This is general for ANY testbench format that has assertion failures.
        """
        self.logger.info("Extracting primary assertion failure for focused debugging")

        # First check for compilation errors (these take priority over assertion failures)
        compilation_error_patterns = [
            # iverilog compilation errors
            r"subprocess\.CalledProcessError.*iverilog.*returned non-zero exit status (\d+)",
            # Generic compilation errors
            r"compilation.*failed|compile.*error|syntax.*error",
            # Tool-specific errors
            r"iverilog.*error|vcs.*error|modelsim.*error"
        ]

        for pattern in compilation_error_patterns:
            match = re.search(pattern, error_report, re.IGNORECASE)
            if match:
                print(f"DEBUG: Found compilation error pattern: {pattern}")
                return PrimaryFailure(
                    expected_value="Successful compilation",
                    actual_value="Compilation failed",
                    timing_context="During compilation phase",
                    raw_assertion=match.group(0),
                    context_lines="Compilation error detected - check RTL syntax and semantics"
                )

        # Generic patterns for assertion failures across different testbench formats
        assertion_patterns = [
            # CVDP format: "FAIL: ... Expected: 0x55, Got: 0xAA"
            r"FAIL.*Expected:\s*([0-9a-fA-Fx]+),\s*Got:\s*([0-9a-fA-Fx]+)",
            # Python/cocotb: "AssertionError: Expected X, got Y"
            r"AssertionError.*Expected\s+([0-9a-fA-Fx]+),\s*(?:but\s+)?got\s+([0-9a-fA-Fx]+)",
            # SystemVerilog: "Assertion failed: expected X, actual Y"
            r"[Aa]ssertion.*failed.*expected:?\s*([0-9a-fA-Fx]+).*actual:?\s*([0-9a-fA-Fx]+)",
            # Generic: "FAIL: expected X, received Y"
            r"FAIL.*expected:?\s*([0-9a-fA-Fx]+).*received:?\s*([0-9a-fA-Fx]+)",
            # More flexible patterns
            r"Expected:?\s*([0-9a-fA-Fx]+).*Got:?\s*([0-9a-fA-Fx]+)",
            r"expected:?\s*([0-9a-fA-Fx]+).*got:?\s*([0-9a-fA-Fx]+)",
        ]

        for pattern in assertion_patterns:
            match = re.search(pattern, error_report, re.IGNORECASE)
            if match:
                expected = match.group(1)
                actual = match.group(2)

                # Extract timing context if available
                timing_context = self._extract_timing_context(error_report, match.start())

                # Extract surrounding context
                context_lines = self._extract_context_lines(error_report, match.start())

                primary_failure = PrimaryFailure(
                    expected_value=expected,
                    actual_value=actual,
                    timing_context=timing_context,
                    raw_assertion=match.group(0),
                    context_lines=context_lines
                )

                self.logger.info(f"Primary failure: Expected {expected}, got {actual}")
                print(f"DEBUG: PrimaryFailureExtractor SUCCESS - Expected {expected}, got {actual}")
                return primary_failure

        # Fallback: Look for ANY AssertionError even without Expected/got pattern
        print(f"DEBUG: PrimaryFailureExtractor - No Expected/got patterns matched, trying generic AssertionError fallback")
        generic_assertion_match = re.search(r'AssertionError:\s*(.*?)(?=\n|$)', error_report, re.IGNORECASE)
        if generic_assertion_match:
            assertion_message = generic_assertion_match.group(1).strip()
            print(f"DEBUG: Found generic AssertionError: {assertion_message}")

            # Extract timing context if available
            timing_context = self._extract_timing_context(error_report, generic_assertion_match.start())

            # Extract surrounding context
            context_lines = self._extract_context_lines(error_report, generic_assertion_match.start())

            primary_failure = PrimaryFailure(
                expected_value="N/A",  # No specific expected value
                actual_value="FAILED",  # Generic failure indicator
                timing_context=timing_context,
                raw_assertion=generic_assertion_match.group(0),
                context_lines=context_lines
            )

            self.logger.info(f"Primary failure (generic): {assertion_message}")
            print(f"DEBUG: PrimaryFailureExtractor SUCCESS (generic) - {assertion_message}")
            return primary_failure

        # Final fallback: Use LLM to extract primary failure
        print(f"DEBUG: PrimaryFailureExtractor - No AssertionError found, trying LLM fallback")
        try:
            llm_failure = self._llm_extract_primary_failure(error_report)
            if llm_failure:
                print(f"DEBUG: PrimaryFailureExtractor SUCCESS (LLM) - {llm_failure.raw_assertion}")
                return llm_failure
        except Exception as e:
            print(f"ERROR: LLM primary failure extraction failed: {e}")
            self.logger.error(f"LLM primary failure extraction failed: {e}")

        print(f"DEBUG: PrimaryFailureExtractor FAILED - All extraction methods exhausted")
        print(f"DEBUG: Error report preview: {error_report[:200]}...")
        return None

    def _extract_timing_context(self, error_report: str, failure_position: int) -> str:
        """Extract timing context around the failure (general approach)."""

        # Look for timing information near the failure - use larger window for comprehensive context
        context_window = error_report[max(0, failure_position-1000):failure_position+1000]

        # Generic timing patterns
        timing_patterns = [
            r"(\d+\.?\d*)\s*ns",        # "500.00ns"
            r"cycle\s*(\d+)",           # "cycle 15"
            r"time[:\s]*(\d+)",         # "time: 100"
            r"@\s*(\d+)"                # "@ 250"
        ]

        for pattern in timing_patterns:
            match = re.search(pattern, context_window)
            if match:
                return f"at {match.group(0)}"

        return "timing unknown"

    def _extract_context_lines(self, error_report: str, failure_position: int) -> str:
        """Extract 2-3 lines of context around failure for understanding."""

        lines = error_report.split('\n')
        failure_line_num = None

        # Find which line contains the failure
        char_count = 0
        for i, line in enumerate(lines):
            char_count += len(line) + 1  # +1 for newline
            if char_count >= failure_position:
                failure_line_num = i
                break

        if failure_line_num is not None:
            # Extract 1 line before and 1 line after for context
            start = max(0, failure_line_num - 1)
            end = min(len(lines), failure_line_num + 2)
            context_lines = lines[start:end]

            # Clean up and join
            cleaned_lines = [line.strip() for line in context_lines if line.strip()]
            return " | ".join(cleaned_lines)  # Include all context lines

        return "context unavailable"

    def _llm_extract_primary_failure(self, error_report: str) -> Optional[PrimaryFailure]:
        """Use LLM as final fallback to extract primary failure from error report."""
        try:
            from .utils.llm_client import LLMClient
            import json

            # For very long reports, truncate intelligently
            if len(error_report) > self.llm_context_limit:
                # Take first part (setup) and last part (results/errors)
                half_limit = self.llm_context_limit // 2
                first_part = error_report[:half_limit]
                last_part = error_report[-half_limit:]
                truncated_report = f"{first_part}\n\n[... middle truncated ...]\n\n{last_part}"
                print(f"DEBUG: LLM fallback - truncated report from {len(error_report)} to {len(truncated_report)} chars")
            else:
                truncated_report = error_report

            prompt = f"""Analyze this RTL testbench error report and extract the PRIMARY failure information.

Error Report:
```
{truncated_report}
```

Task: Find the FIRST/PRIMARY failure and extract key information. This could be:
- Compilation errors (iverilog, vcs, etc. returning non-zero exit status)
- Runtime assertion failures (AssertionError, test failures)
- Syntax errors or RTL semantic issues
- Any other indication that the test/compilation failed

Respond with JSON in this exact format:
{{
  "found_failure": true/false,
  "expected_value": "what was expected (or 'N/A' if no specific value)",
  "actual_value": "what was received (or 'FAILED' for general failures)",
  "raw_assertion": "the exact error message or assertion",
  "timing_context": "timing information if available (or 'compilation phase' for compile errors)",
  "failure_description": "brief description of what failed"
}}

Look for ANY type of failure: compilation errors, assertion failures, syntax errors, or test failures."""

            client = LLMClient(model=self.model_name, logger=self.logger)
            response = client.call_llm(prompt, max_tokens=500)

            if not response or not response.strip():
                print(f"ERROR: LLM returned empty response for primary failure extraction")
                return None

            # Parse JSON response - robust extraction from anywhere in response
            try:
                # Look for JSON block anywhere in the response
                json_start = response.find("```json")
                if json_start != -1:
                    # Extract content between ```json and ```
                    json_content = response[json_start + 7:]  # Skip ```json
                    json_end = json_content.find("```")
                    if json_end != -1:
                        clean_response = json_content[:json_end].strip()
                    else:
                        clean_response = json_content.strip()
                else:
                    # Try to find JSON object directly (look for { and matching })
                    json_start = response.find("{")
                    if json_start != -1:
                        # Find the matching closing brace
                        brace_count = 0
                        json_end = json_start
                        for i, char in enumerate(response[json_start:], json_start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break
                        clean_response = response[json_start:json_end].strip()
                    else:
                        # Last resort - try parsing the whole response
                        clean_response = response.strip()

                result = json.loads(clean_response)
                print(f"DEBUG: Successfully parsed LLM JSON response")

                print(f"DEBUG: LLM analysis result: {result}")

                if not result.get('found_failure', False):
                    print(f"DEBUG: LLM found no failure in error report")
                    return None

                # Create PrimaryFailure from LLM analysis
                primary_failure = PrimaryFailure(
                    expected_value=result.get('expected_value', 'N/A'),
                    actual_value=result.get('actual_value', 'FAILED'),
                    timing_context=result.get('timing_context', 'timing unknown'),
                    raw_assertion=result.get('raw_assertion', result.get('failure_description', 'LLM extracted failure')),
                    context_lines=result.get('failure_description', 'LLM analysis - see raw_assertion for details')
                )

                self.logger.info(f"Primary failure (LLM): {result.get('failure_description', 'extracted via LLM')}")
                return primary_failure

            except json.JSONDecodeError as e:
                print(f"ERROR: LLM returned invalid JSON for primary failure extraction: {e}")
                print(f"ERROR: LLM response was: {response[:200]}...")
                return None

        except Exception as e:
            print(f"ERROR: Exception in LLM primary failure extraction: {e}")
            import traceback
            traceback.print_exc()
            return None
