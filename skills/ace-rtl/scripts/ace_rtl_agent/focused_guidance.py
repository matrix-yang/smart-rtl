# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused debugger implementation mixins."""

import os
import re
from typing import Dict, List, Optional, Tuple

from .focused_debugger_types import FixAttempt


class FocusedGuidanceMixin:
    def _sanitize_cvdp_feedback_guidance(self, text: str) -> str:
        """Remove over-specific CVDP repair hints that may encode private checker behavior."""
        if not text:
            return text
        lowered = text.lower()
        if self._is_default_output_feedback_guess(lowered):
            return self._default_output_functional_guidance()
        private_source_markers = (
            "reference model",
            "harness_library",
            "private reference",
            "hidden reference",
            "golden model",
            "checker/reference",
        )
        private_source_patterns = (
            r"\bprivate\s+(?:checker|reference|formula|model)\b",
            r"\b(?:checker|reference)\s+(?:formula|implementation|internals?)\b",
            r"\bfrom\s+the\s+(?:checker|reference|harness)\b",
        )
        prescriptive_markers = (
            "change the",
            "replace",
            "remove the",
            "add the",
            "assignment",
            "term",
        )
        references_private_source = any(marker in lowered for marker in private_source_markers) or any(
            re.search(pattern, lowered) for pattern in private_source_patterns
        )
        if references_private_source and any(
            marker in lowered for marker in prescriptive_markers
        ):
            return (
                "The candidate is failing a functional comparison after compilation, so repair it "
                "against the public specification: re-check byte ordering, arithmetic/matrix terms, "
                "key application, reset defaults, and the alignment of data, valid, and operation "
                "sideband pipelines. Use evaluator mismatches only to localize the affected public "
                "outputs; do not hardcode expected values or copy private checker/reference behavior."
            )
        return text

    def _default_output_functional_guidance(self) -> str:
        """Generic guidance for valid-cycle default-output functional failures."""
        return (
            "The candidate is failing a functional comparison after compilation with a checked "
            "data output still at its reset/default value. Repair it against the public "
            "specification by tracing the accepted transaction through data capture, "
            "mode/key sideband registers, result-register update timing, and output "
            "valid/done gating. Also check for default/idle branches that clear a registered "
            "result immediately before a separately pipelined valid/done flag exposes it. "
            "Do not assume a reset-only fix unless the simulator reports "
            "X/Z or uninitialized values, and do not hardcode expected outputs or copy "
            "private checker/reference behavior."
        )

    def _is_default_output_functional_mismatch(self, error_report: str) -> bool:
        """Detect functional checks where valid/done reached with default data."""
        lowered = (error_report or "").lower()
        if any(token in lowered for token in ("x/z", "uninitialized", "unable to accurately represent")):
            return False
        has_functional_compare = (
            "dut data output does not match" in lowered
            or "dut output data" in lowered
            or "data output is 0x0" in lowered
            or "actual output is 0" in lowered
        )
        has_default_actual = bool(
            re.search(r"(?:0x0+\b|actual(?: output)? is 0\b|actual\s*[:=]\s*0\b)", lowered)
        )
        has_nonzero_expected = bool(
            re.search(r"!=\s*0x[1-9a-f][0-9a-f]*|expected(?: model)?(?: output)?(?: is|:)?\s*0x[1-9a-f][0-9a-f]*", lowered)
        )
        return has_functional_compare and has_default_actual and has_nonzero_expected

    def _contains_reset_prescription(self, text: str) -> bool:
        """Return True when feedback prescribes reset changes as a fix."""
        lowered = (text or "").lower()
        return any(
            token in lowered
            for token in (
                "add `negedge",
                "add negedge",
                "add asynchronous reset",
                "add async reset",
                "reset all",
                "reset branch",
                "reset coverage",
                "reset behavior",
                "reset initialization",
                "initialize all",
                "full reset",
            )
        )

    def _is_default_output_feedback_guess(self, lowered_text: str) -> bool:
        """Detect unsupported reset-only guidance for all-zero output mismatches."""
        if any(token in lowered_text for token in ("x/z", "uninitialized", "unknown-state", "unknown state")):
            return False
        has_default_output = any(
            token in lowered_text
            for token in (
                "output data is 0x0",
                "actual output is 0",
                "dut outputs all zeros",
                "output `o_data` is `0x0`",
                "output data is `0x0`",
                "actual `o_data` is `0x0`",
                "0x0 while",
            )
        )
        has_reset_prescription = self._contains_reset_prescription(lowered_text)
        return has_default_output and has_reset_prescription

    def _simple_error_reasoning(self, focused_error: str) -> str:
        """Fallback simple reasoning if LLM call fails."""
        if "Expected `1`, got `0`" in focused_error:
            return "The signal is reading as 0 when it should be 1. This typically means: (1) the signal is not being assigned, (2) the assignment condition is not being met, or (3) the signal is being reset or overwritten."
        elif "Expected `0`, got `1`" in focused_error:
            return "The signal is 1 when it should be 0. This typically means: (1) the signal is not being cleared properly, (2) reset logic is missing, or (3) the signal state persists when it shouldn't."
        else:
            return "Analyze the error message and buggy code to understand what logic error is causing this failure."

    def _extract_and_reason_about_error(self, error_report: str) -> Tuple[str, str]:
        """
        Extract ONE specific error and provide simple reasoning.

        Used when we don't have buggy code available (e.g., in llm_centric_debugger metadata extraction).

        Returns:
            (focused_error, simple_reasoning)
        """
        focused_error = self._extract_focused_error_only(error_report)
        simple_reasoning = self._simple_error_reasoning(focused_error)
        return focused_error, simple_reasoning

    def extract_guidance_from_response(self, response: str) -> str:
        """
        Extract the GUIDANCE section from LLM response.

        This will be recorded in history for future iterations.
        """
        # Look for GUIDANCE section
        guidance_match = re.search(r'###\s*GUIDANCE\s*\n(.+?)(?=###|\Z)', response, re.DOTALL | re.IGNORECASE)
        if guidance_match:
            guidance = guidance_match.group(1).strip()
            # Return full guidance - no truncation
            return guidance

        # Fallback: Look for analysis or fix strategy
        for section_name in ['FIX STRATEGY', 'ANALYSIS', 'ROOT CAUSE']:
            match = re.search(rf'###?\s*{section_name}\s*\n(.+?)(?=###|\Z)', response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return "No guidance extracted from response"

    def extract_implementation_changes(self, code_before: str, code_after: str,
                                      guidance: str, focused_error: str,
                                      new_error_report: str) -> Tuple[str, str]:
        """
        LLM Call 3: Retrospective analysis

        Analyzes:
        1. What implementation changes were made (compare codes)
        2. Whether the fix actually solved the target error (compare error reports)

        Returns:
            Tuple of (implementation_changes, fix_effectiveness)
        """
        # If codes are too similar or identical, return simple message
        if code_before == code_after:
            return "No code changes detected", "No changes made"

        truncated_error_report = self._prepare_error_report_for_prompt(new_error_report)

        # Create retrospective analysis prompt
        changes_prompt = f"""Analyze the code changes and their effectiveness at fixing the target error.

TARGET ERROR THAT WAS FOCUSED ON:
{focused_error}

FIX GUIDANCE THAT WAS APPLIED:
{guidance}

BUGGY CODE (BEFORE FIX):
```systemverilog
{code_before}
```

CORRECTED CODE (AFTER FIX):
```systemverilog
{code_after}
```

NEW ERROR REPORT (after applying fix):
```
{truncated_error_report}
```

YOUR TASK - Provide TWO brief analyses:

### 1. IMPLEMENTATION CHANGES (1-2 sentences)
What specific changes were made to the code?
- Reference actual signal names, registers, logic blocks
- Describe what was added, modified, or removed
- Example: "Added fee_ready_pulse register in EXIT_PROCESSING state that sets signal high then clears it on next cycle"

### 2. FIX EFFECTIVENESS (1-2 sentences)
Did the fix solve the target error? Did it introduce new errors?
- Check if the SPECIFIC target error is still in the new error report
- Check if NEW or DIFFERENT errors appeared (regression detection)
- Be factual: "Solved the target error" OR "Did not solve - error still present" OR "Solved but introduced new error X" OR "Different error now appears"
- Example 1: "The target error 'Fee not marked as ready' is no longer present - solved successfully"
- Example 2: "The target error 'Load error in hours' is solved, but a NEW error 'Full rollover failed' now appears, suggesting the fix broke countdown functionality"

OUTPUT FORMAT:
### IMPLEMENTATION CHANGES
<1-2 sentences>

### FIX EFFECTIVENESS
<1 sentence>"""

        try:
            # Use ACE-RTL temperature if stage 1 uses ACE-RTL, otherwise use 0.1
            temp = self.ace_rtl_temperature if 1 in self.ace_rtl_stages else 0.1
            changes_response = self.llm_client.call_llm(changes_prompt, temperature=temp)
            if changes_response:
                # Extract IMPLEMENTATION CHANGES section
                changes_match = re.search(r'###\s*(?:1\.)?\s*IMPLEMENTATION CHANGES\s*\n(.+?)(?=###|$)',
                                         changes_response, re.DOTALL | re.IGNORECASE)
                impl_changes = changes_match.group(1).strip() if changes_match else changes_response.strip()

                # Extract FIX EFFECTIVENESS section
                effectiveness_match = re.search(r'###\s*(?:2\.)?\s*FIX EFFECTIVENESS\s*\n(.+?)(?=###|$)',
                                               changes_response, re.DOTALL | re.IGNORECASE)
                fix_effectiveness = effectiveness_match.group(1).strip() if effectiveness_match else "Effectiveness not analyzed"

                # Clean up - remove hedging language
                impl_changes = re.sub(r'\b(likely|probably|possibly|may have|might have)\b', '', impl_changes, flags=re.IGNORECASE).strip()
                fix_effectiveness = re.sub(r'\b(likely|probably|possibly|may have|might have)\b', '', fix_effectiveness, flags=re.IGNORECASE).strip()

                return impl_changes, fix_effectiveness
        except Exception as e:
            self.logger.warning(f"Failed to extract implementation changes via LLM: {e}")

        # Fallback: Simple diff analysis
        simple_changes = self._simple_diff_analysis(code_before, code_after)
        return simple_changes, "Effectiveness not analyzed"

    def _prepare_error_report_for_prompt(self, error_report: str, max_chars: Optional[int] = None) -> str:
        """Keep LLM prompts focused while preserving the full report on disk."""
        if not error_report:
            return ""

        if max_chars is None:
            try:
                max_chars = int(os.environ.get("ACE_RTL_ERROR_REPORT_PROMPT_CHARS", "90000"))
            except ValueError:
                max_chars = 90000

        if len(error_report) <= max_chars:
            return error_report

        truncated = self._intelligently_truncate_error_report(error_report, max_chars=max_chars)
        self.logger.info(
            f"Prepared compact error report for prompt: {len(error_report)} chars -> {len(truncated)} chars"
        )
        return truncated

    def _intelligently_truncate_error_report(self, error_report: str, max_chars: int = 90000) -> str:
        """
        Intelligently truncate very large error reports while preserving important information.

        Strategy:
        1. Keep first N lines (test setup, first failures)
        2. Extract key error lines (AssertionError, FAILED, traceback)
        3. Keep last N lines (summary, final results)
        4. Target: ~90KB max by default
        """
        lines = error_report.split('\n')
        total_lines = len(lines)

        if len(error_report) <= max_chars:
            return error_report

        # Calculate how many lines we can keep
        avg_line_length = len(error_report) / total_lines if total_lines > 0 else 100
        target_lines = int(max_chars / avg_line_length)

        # Preserve structure: top + key errors + bottom
        keep_top = int(target_lines * 0.3)  # 30% for beginning
        keep_bottom = int(target_lines * 0.2)  # 20% for end
        keep_middle = target_lines - keep_top - keep_bottom  # 50% for key errors

        # Part 1: Keep first lines (test setup, initial failures)
        top_lines = lines[:keep_top]

        # Part 2: Extract ALL key error lines from middle section
        middle_section = lines[keep_top:-keep_bottom] if keep_bottom > 0 else lines[keep_top:]

        # Critical patterns that MUST be preserved. CVDP native reports mix
        # pytest, Cadence logs, mutation results, and coverage summaries.
        critical_patterns = [
            'AssertionError', 'FAILED', 'ERROR', 'Traceback', 'FAIL:',
            'BUG_', 'Simulation ended without error', 'Expected', 'Got=',
            'Got ', 'mismatch', 'Mismatch', '$error', 'ncvlog:', 'ncelab:',
            'xrun:', '*E,', '*F,', 'coverage', 'Coverage', 'TARGET',
            'returned non-zero', 'Syntax error', 'parse error',
        ]

        # Find ALL lines with critical patterns (with context)
        critical_indices = set()
        for i, line in enumerate(middle_section):
            if any(pattern in line for pattern in critical_patterns):
                # Include context: 2 lines before and after
                for j in range(max(0, i-2), min(len(middle_section), i+3)):
                    critical_indices.add(j)

        # Extract critical lines in order
        key_lines = [middle_section[i] for i in sorted(critical_indices)]

        # If we have too many critical lines, prioritize by pattern importance
        if len(key_lines) > keep_middle:
            # First pass: Keep all AssertionError, FAILED, ERROR lines with minimal context
            priority_indices = set()
            for i, line in enumerate(middle_section):
                if any(pattern in line for pattern in critical_patterns):
                    # Just +/-1 line context instead of +/-2
                    for j in range(max(0, i-1), min(len(middle_section), i+2)):
                        priority_indices.add(j)

            key_lines = [middle_section[i] for i in sorted(priority_indices)]

            # If still too many, take first occurrences
            if len(key_lines) > keep_middle:
                key_lines = key_lines[:keep_middle]

        # Part 3: Keep last lines (test summary, final results)
        bottom_lines = lines[-keep_bottom:] if keep_bottom > 0 else []

        # Combine all parts
        truncated_lines = (
            top_lines +
            [f"\n... (middle section truncated - showing key errors only) ...\n"] +
            key_lines +
            [f"\n... (showing final {keep_bottom} lines) ...\n"] +
            bottom_lines
        )

        truncated_report = '\n'.join(truncated_lines)

        if len(truncated_report) > max_chars:
            marker = "\n... [TRUNCATED TO ERROR REPORT PROMPT CAP] ...\n"
            budget = max_chars - len(marker)
            head = max(0, int(budget * 0.55))
            tail = max(0, budget - head)
            truncated_report = truncated_report[:head] + marker + truncated_report[-tail:]

        self.logger.info(f"Intelligent truncation: {total_lines} lines -> {len(truncated_lines)} lines, "
                        f"{len(error_report)} chars -> {len(truncated_report)} chars")

        return truncated_report

    def _simple_diff_analysis(self, code_before: str, code_after: str) -> str:
        """Simple fallback for describing code changes."""
        # Count lines added/removed
        lines_before = code_before.split('\n')
        lines_after = code_after.split('\n')

        if len(lines_after) > len(lines_before) + 5:
            return "Added significant new logic to the code"
        elif len(lines_before) > len(lines_after) + 5:
            return "Removed significant portions of code"
        else:
            # Look for specific changes
            added_text = '\n'.join([l for l in lines_after if l not in lines_before])

            if 'always' in added_text:
                return "Modified or added always block logic"
            elif '<=' in added_text or '=' in added_text:
                signals = re.findall(r'(\w+)\s*[<]?=', added_text)
                if signals:
                    unique_sigs = list(set(signals))[:3]
                    return f"Modified assignments for: {', '.join(unique_sigs)}"

            return "Made targeted modifications to the logic"

    def parse_test_results(self, error_report: str) -> Tuple[int, int]:
        """
        Parse test results from error report.

        Returns:
            (num_passing, num_total)
        """
        # Look for test summary patterns
        patterns = [
            r'(\d+)\s+passed.*?(\d+)\s+total',
            r'(\d+)/(\d+)\s+tests?\s+passed',
            r'(\d+)\s+passed.*?(\d+)\s+failed'
        ]

        for pattern in patterns:
            match = re.search(pattern, error_report, re.IGNORECASE)
            if match:
                passed = int(match.group(1))
                if 'total' in pattern:
                    total = int(match.group(2))
                elif 'failed' in pattern:
                    failed = int(match.group(2))
                    total = passed + failed
                else:
                    total = int(match.group(2))
                return passed, total

        # Fallback: Count individual test results
        passed = len(re.findall(r'\bpassed\b', error_report, re.IGNORECASE))
        failed = len(re.findall(r'\bfailed\b', error_report, re.IGNORECASE))

        if passed > 0 or failed > 0:
            return passed, passed + failed

        return 0, 0
