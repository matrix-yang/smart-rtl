# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused debugger implementation mixins."""

import os
import re
from typing import Dict, List, Optional, Tuple

from .focused_debugger_types import FixAttempt


class FocusedHistoryMixin:
    def record_attempt(self,
                      iteration: int,
                      focused_error: str,
                      error_reasoning: str,
                      guidance: str,
                      implementation_changes: str,
                      fix_effectiveness: str,
                      test_results: str,
                      outcome: str,
                      test_passed: bool = False,
                      mismatch_count: int = -1):
        """
        Record a fix attempt for history tracking.

        Args:
            iteration: Iteration number
            focused_error: The error message focused on
            error_reasoning: Reasoning about what the error means
            guidance: The guidance/fix approach tried
            implementation_changes: Actual code changes made (HOW the guidance was implemented)
            fix_effectiveness: Whether the fix solved the target error
            test_results: Test results after fix (e.g., "3/10 tests passed")
            outcome: "improved", "no_change", "regressed", "solved"
            mismatch_count: Numeric mismatch count from simulation
        """
        attempt = FixAttempt(
            iteration=iteration,
            focused_error=focused_error,
            error_reasoning=error_reasoning,
            guidance_tried=guidance,
            implementation_changes=implementation_changes,
            fix_effectiveness=fix_effectiveness,
            test_results=test_results,
            outcome=outcome,
            mismatch_count=mismatch_count
        )

        self.fix_history.append(attempt)

        # Check if current error was solved - with regression detection
        # 1. Check if LLM says the specific error was solved
        llm_says_solved = fix_effectiveness and any(keyword in fix_effectiveness.lower() for keyword in
                                     ['solved', 'no longer present', 'fixed', 'resolved', 'eliminated', 'successfully'])

        # 2. Check if LLM explicitly says it's NOT solved
        llm_says_not_solved = fix_effectiveness and any(keyword in fix_effectiveness.lower() for keyword in
                                     ['still present', 'not solve', 'did not solve', 'failed to solve', 'remains', 'persists'])

        # 3. Check if we introduced NEW errors (regression detection)
        introduced_new_errors = self._check_if_new_errors_introduced(fix_effectiveness)

        if llm_says_not_solved:
            # LLM explicitly confirms error still present
            self.error_solved = False
            self.logger.info(f"LLM confirms error NOT solved in iteration {iteration} - continuing to focus on it")
        elif llm_says_solved and introduced_new_errors:
            if (
                self._is_structural_or_build_error(focused_error)
                or self._is_structural_or_build_error(fix_effectiveness)
            ):
                # Once a structural/build error is fixed, move on to the real
                # simulator or assertion failure instead of repeatedly targeting
                # an obsolete missing-module/port/signal problem. If a fix
                # introduces a new structural failure, switch to that immediately
                # because the public contract must be restored before functional
                # debugging is meaningful.
                self.error_solved = True
                self.logger.info(f"Structural/build issue changed in iteration {iteration}; switching focus to the current failure")
            else:
                # Error "solved" but broke something else - don't mark as solved yet
                self.error_solved = False
                self.logger.info(f"Current error appears solved but introduced NEW errors in iteration {iteration}")
                self.logger.info(f"   Keeping focus on original error to find solution that doesn't break other functionality")
        elif llm_says_solved:
            # LLM confirms the specific error is gone and no new errors introduced
            self.error_solved = True
            self.logger.info(f"Current error SOLVED in iteration {iteration} - specific error resolved without breaking other functionality")
        else:
            # Unclear - be conservative and keep focusing
            self.error_solved = False
            self.logger.info(f"Unclear if error solved in iteration {iteration} - continuing to focus on same error")

        self.logger.info(f"Recorded iteration {iteration}: {outcome} - {test_results}")

    def _check_if_new_errors_introduced(self, fix_effectiveness: str) -> bool:
        """
        Check if the fix introduced new errors while solving the current one.

        General approach (no problem-specific logic):
        - Look for keywords indicating new/different errors appeared
        - Detect if LLM mentions breaking other functionality
        - Identify regression patterns

        Returns:
            True if new errors were introduced, False otherwise
        """
        if not fix_effectiveness:
            return False

        effectiveness_lower = fix_effectiveness.lower()

        # Keywords indicating new errors introduced
        new_error_indicators = [
            'new error', 'different error', 'different failure', 'now shows',
            'instead shows', 'but now', 'however', 'broke', 'regressed',
            'introduced', 'caused', 'opposite problem', 'reverse problem',
            'now fails', 'new failure'
        ]

        # Check if any indicator is present
        has_new_error = any(indicator in effectiveness_lower for indicator in new_error_indicators)

        if has_new_error:
            self.logger.info(f"   Detected new error introduction in fix effectiveness analysis")

        return has_new_error

    def _is_structural_or_build_error(self, focused_error: str) -> bool:
        """Return True for compile/elaboration/interface failures that should not stay sticky after they disappear."""
        if not focused_error:
            return False

        error_text = focused_error.lower()
        structural_indicators = [
            'unable to find the root module',
            'root module',
            'module not found',
            'unknown module type',
            'module definition',
            'contains no object named',
            'no object named',
            'port mismatch',
            'no port named',
            'parameter not found',
            'not found in module',
            'has no member',
            'undeclared',
            'not declared',
            'syntax error',
            'parse error',
            'compilation failed',
            'compilation error',
            'elaboration failed',
            'iverilog',
            'cvdp_rtl_semantic_guard',
            'semantic guard',
            'public cvdp harness contract',
            'harness contract',
            'simulator-visible clocked-block temporary',
            'generated rtl has simulator-visible',
            'returned non-zero',
            'exit status',
        ]
        if any(indicator in error_text for indicator in structural_indicators):
            return True

        return re.search(
            r"\b(module|port|signal|parameter|wire|reg|interface)\b.*\b(missing|not found|unknown|undeclared|not declared)\b",
            error_text,
        ) is not None

    def _error_focus_kind(self, error_text: str) -> str:
        """Classify the active failure so obsolete structural focuses do not hide runtime failures."""
        if not error_text:
            return "unknown"

        text = error_text.lower()
        semantic_guard_indicators = [
            'cvdp_rtl_semantic_guard',
            'semantic guard',
            'public cvdp harness contract',
            'harness contract',
            'simulator-visible clocked-block temporary',
            'generated rtl has simulator-visible',
        ]
        if any(indicator in text for indicator in semantic_guard_indicators):
            return "structural"

        runtime_indicators = [
            'assertionerror',
            'expected',
            'got',
            'mismatch',
            'bit error',
            'test failed',
            'simulation ended with error',
            '$error',
            '$fatal',
            'timeout',
            'timed out',
            'cocotb',
            '[check]',
            'running sum',
            'long string of zeros',
        ]
        if any(indicator in text for indicator in runtime_indicators):
            return "runtime"

        optimization_indicators = [
            'yosys',
            'number of cells',
            'number of wires',
            'synthesis',
            'area target',
            'wire target',
        ]
        if any(indicator in text for indicator in optimization_indicators):
            return "optimization"

        coverage_indicators = [
            'coverage',
            'imc',
            'uncovered',
        ]
        if any(indicator in text for indicator in coverage_indicators):
            return "coverage"
        if self._is_structural_or_build_error(text):
            return "structural"

        return "unknown"

    def _build_history_trajectory(self) -> str:
        """
        Build concise history showing what was tried and what happened.
        Includes mismatch trend line for objective progress tracking.
        """
        if not self.fix_history:
            return ""

        try:
            max_attempts = int(os.environ.get("ACE_RTL_HISTORY_MAX_ATTEMPTS", "8"))
        except ValueError:
            max_attempts = 8
        max_attempts = max(1, max_attempts)

        try:
            max_field_chars = int(os.environ.get("ACE_RTL_HISTORY_FIELD_CHARS", "1200"))
        except ValueError:
            max_field_chars = 1200
        max_field_chars = max(200, max_field_chars)

        def compact(text: str) -> str:
            text = (text or "").strip()
            if len(text) <= max_field_chars:
                return text
            marker = " ... [truncated] ... "
            keep = max_field_chars - len(marker)
            head = int(keep * 0.65)
            tail = keep - head
            return text[:head].rstrip() + marker + text[-tail:].lstrip()

        history_parts = []

        # Mismatch trend line (objective progress indicator)
        counts = [a.mismatch_count for a in self.fix_history if a.mismatch_count >= 0]
        if counts:
            trend = " -> ".join(str(c) for c in counts)
            best = min(counts)
            current = counts[-1]
            history_parts.append(f"**MISMATCH TREND**: {trend}")
            history_parts.append(f"  Best achieved: {best} mismatches | Current: {current} mismatches")

            oscillation = self._detect_oscillation()
            if oscillation:
                history_parts.append(f"  WARNING: {oscillation}")
            history_parts.append("")

        history_parts.append("**Previous Attempts:**\n")

        shown_attempts = self.fix_history[-max_attempts:]
        omitted = len(self.fix_history) - len(shown_attempts)
        if omitted > 0:
            history_parts.append(f"Showing the most recent {len(shown_attempts)} attempts; {omitted} older attempts omitted to keep the prompt within model context.")
            history_parts.append("")

        for attempt in shown_attempts:
            outcome_emoji = {
                "solved": "SOLVED",
                "improved": "IMPROVED",
                "no_change": "NO CHANGE",
                "regressed": "REGRESSED"
            }.get(attempt.outcome, attempt.outcome.upper())

            mismatch_str = f" ({attempt.mismatch_count} mismatches)" if attempt.mismatch_count >= 0 else ""
            history_parts.append(f"**Iteration {attempt.iteration}**: {outcome_emoji}{mismatch_str}")
            history_parts.append(f"  - **Error focused on**: {compact(attempt.focused_error)}")
            history_parts.append(f"  - **Fix guidance tried**: {compact(attempt.guidance_tried)}")
            history_parts.append(f"  - **Implementation changes made**: {compact(attempt.implementation_changes)}")
            history_parts.append(f"  - **Did it solve the error?**: {compact(attempt.fix_effectiveness)}")
            history_parts.append(f"  - **Test results**: {compact(attempt.test_results)}")
            history_parts.append("")

        return "\n".join(history_parts)

    def _is_stalled(self) -> bool:
        """Check if mismatch count hasn't improved by >20% over the last 5 iterations."""
        counts = [a.mismatch_count for a in self.fix_history if a.mismatch_count >= 0]
        if len(counts) < 5:
            return False
        recent = counts[-5:]
        earliest_in_window = recent[0]
        best_in_window = min(recent)
        if earliest_in_window == 0:
            return False
        improvement = (earliest_in_window - best_in_window) / earliest_in_window
        return improvement < 0.2

    def _detect_oscillation(self) -> Optional[str]:
        """Detect if mismatch counts are oscillating without progress."""
        counts = [a.mismatch_count for a in self.fix_history if a.mismatch_count >= 0]
        if len(counts) < 6:
            return None

        recent = counts[-6:]
        best_recent = min(recent)
        worst_recent = max(recent)

        # Check if spread is small (stuck in a band)
        if worst_recent > 0 and (worst_recent - best_recent) / worst_recent < 0.15:
            return (f"Mismatch count stuck in range [{best_recent}-{worst_recent}] for last 6 iterations. "
                    f"Current approach is not converging. Try a fundamentally different architecture or algorithm.")

        # Check for alternating pattern (A, B, A, B)
        if len(counts) >= 4:
            diffs = [counts[i+1] - counts[i] for i in range(len(counts)-1)]
            sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
            if sign_changes >= len(diffs) - 1 and sign_changes >= 3:
                return ("Mismatch count is oscillating (alternating increase/decrease). "
                        "Fixes are likely trading one bug for another. "
                        "Focus on preserving ALL currently-passing signals while fixing the target.")

        return None
