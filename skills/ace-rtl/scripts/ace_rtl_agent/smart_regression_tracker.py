# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Smart Regression Tracker - Tracks code changes, progress, and repeated mistakes.

Replaces SimpleRegressionTracker with more intelligent tracking.
"""

import re
import difflib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class IterationResult:
    """Detailed tracking of iteration results."""
    iteration: int
    test_passed: bool
    error_report: str
    code: str
    tests_passed_count: int = 0
    tests_failed_count: int = 0
    improvements_noted: List[str] = field(default_factory=list)
    regressions_noted: List[str] = field(default_factory=list)
    code_changes: List[str] = field(default_factory=list)  # Descriptions of what changed
    fix_attempted: Optional[str] = None  # What fix was attempted


class SmartRegressionTracker:
    """
    Smart regression tracking that learns from iteration history.

    Features:
    1. Track code diffs to see what changed
    2. Measure progress (test pass count)
    3. Detect repeated attempts (same fix tried multiple times)
    4. Identify what's working vs what's broken
    """

    def __init__(self, logger):
        self.logger = logger
        self.iteration_history: List[IterationResult] = []
        self.attempted_fixes: Dict[str, List[int]] = {}  # fix_description -> [iteration numbers]
        self.working_features: List[str] = []  # Features that are working
        self.broken_features: List[str] = []  # Features that remain broken

    def record_iteration(self, iteration: int, test_passed: bool,
                        error_report: str, code: str,
                        fix_description: Optional[str] = None):
        """Record an iteration with detailed tracking."""

        # Count passing/failing tests
        tests_passed, tests_failed = self._count_test_results(error_report)

        # Detect improvements/regressions vs previous iteration
        improvements = []
        regressions = []

        if self.iteration_history:
            prev_result = self.iteration_history[-1]

            # Compare test counts
            if tests_passed > prev_result.tests_passed_count:
                improvements.append(f"Tests improved: {prev_result.tests_passed_count} -> {tests_passed} passing")
            elif tests_passed < prev_result.tests_passed_count:
                regressions.append(f"Tests regressed: {prev_result.tests_passed_count} -> {tests_passed} passing")

            # Detect specific test changes
            prev_passing_tests = self._extract_test_names(prev_result.error_report, passed=True)
            prev_failing_tests = self._extract_test_names(prev_result.error_report, passed=False)
            curr_passing_tests = self._extract_test_names(error_report, passed=True)
            curr_failing_tests = self._extract_test_names(error_report, passed=False)

            # New passes
            newly_passing = curr_passing_tests - prev_passing_tests
            if newly_passing:
                improvements.append(f"Newly passing tests: {', '.join(list(newly_passing)[:3])}")

            # New failures
            newly_failing = curr_failing_tests - prev_failing_tests
            if newly_failing:
                regressions.append(f"Newly failing tests: {', '.join(list(newly_failing)[:3])}")

        # Analyze code changes
        code_changes = []
        if self.iteration_history:
            prev_code = self.iteration_history[-1].code
            code_changes = self._analyze_code_diff(prev_code, code)

        # Record the iteration
        result = IterationResult(
            iteration=iteration,
            test_passed=test_passed,
            error_report=error_report,
            code=code,
            tests_passed_count=tests_passed,
            tests_failed_count=tests_failed,
            improvements_noted=improvements,
            regressions_noted=regressions,
            code_changes=code_changes,
            fix_attempted=fix_description
        )

        self.iteration_history.append(result)

        # Track attempted fixes
        if fix_description:
            fix_key = self._normalize_fix_description(fix_description)
            if fix_key not in self.attempted_fixes:
                self.attempted_fixes[fix_key] = []
            self.attempted_fixes[fix_key].append(iteration)

        # Update working/broken features
        self._update_feature_status(error_report, test_passed)

        self.logger.info(f"Iteration {iteration} recorded: {tests_passed} pass, {tests_failed} fail")

    def generate_context_for_llm(self, current_iteration: int) -> str:
        """
        Generate smart context showing progress, changes, and warnings.
        """
        if not self.iteration_history:
            return "No previous iterations to reference."

        context = ["## ITERATION HISTORY & PROGRESS", ""]

        # Show recent history with details
        recent_history = self.iteration_history[-4:]  # Last 4 iterations

        for result in recent_history:
            status = "PASSED" if result.test_passed else "FAILED"
            context.append(f"### Iteration {result.iteration}: {status}")
            context.append(f"- **Test Results**: {result.tests_passed_count} passing, {result.tests_failed_count} failing")

            # Show what changed
            if result.code_changes:
                context.append(f"- **Code Changes**: {result.code_changes[0]}")  # Show primary change
                if len(result.code_changes) > 1:
                    context.append(f"  (and {len(result.code_changes) - 1} more changes)")

            # Show what fix was attempted
            if result.fix_attempted:
                context.append(f"- **Fix Attempted**: {result.fix_attempted}")

            # Show improvements/regressions
            if result.improvements_noted:
                context.append(f"- **Improvements**: {'; '.join(result.improvements_noted[:2])}")
            if result.regressions_noted:
                context.append(f"- **Regressions**: {'; '.join(result.regressions_noted[:2])}")

            context.append("")

        # Show progress trend
        context.append("### Progress Trend:")
        progress_summary = self._analyze_progress_trend()
        context.append(progress_summary)
        context.append("")

        # Warn about repeated attempts
        repeated_attempts = self._detect_repeated_attempts()
        if repeated_attempts:
            context.append("### REPEATED ATTEMPTS DETECTED:")
            for attempt in repeated_attempts[:3]:
                context.append(f"- {attempt}")
            context.append("")

        # Show what's working vs broken
        if self.working_features or self.broken_features:
            context.append("### Feature Status:")
            if self.working_features:
                context.append(f"- **Working**: {', '.join(self.working_features[:3])}")
            if self.broken_features:
                context.append(f"- **Still Broken**: {', '.join(self.broken_features[:3])}")
            context.append("")

        # Strategic recommendation
        context.append("### Strategic Recommendation:")
        recommendation = self._generate_strategic_recommendation(current_iteration)
        context.append(recommendation)

        return "\n".join(context)

    def get_progress_stats(self) -> Dict[str, any]:
        """Get statistics about progress."""
        if not self.iteration_history:
            return {}

        latest = self.iteration_history[-1]
        first = self.iteration_history[0]

        return {
            'total_iterations': len(self.iteration_history),
            'current_passing': latest.tests_passed_count,
            'current_failing': latest.tests_failed_count,
            'initial_passing': first.tests_passed_count,
            'initial_failing': first.tests_failed_count,
            'improvement': latest.tests_passed_count - first.tests_passed_count,
            'any_iteration_passed': any(r.test_passed for r in self.iteration_history)
        }

    def _count_test_results(self, error_report: str) -> Tuple[int, int]:
        """Count how many tests passed vs failed."""
        passed = len(re.findall(r'\bpassed\b', error_report, re.IGNORECASE))
        failed = len(re.findall(r'\bfailed\b', error_report, re.IGNORECASE))

        # Sometimes format is "X/Y tests passed"
        match = re.search(r'(\d+)/(\d+)\s+tests?\s+passed', error_report, re.IGNORECASE)
        if match:
            passed = int(match.group(1))
            total = int(match.group(2))
            failed = total - passed

        # Or "X passed, Y failed"
        match = re.search(r'(\d+)\s+passed.*?(\d+)\s+failed', error_report, re.IGNORECASE)
        if match:
            passed = int(match.group(1))
            failed = int(match.group(2))

        return passed, failed

    def _extract_test_names(self, error_report: str, passed: bool) -> set:
        """Extract test names that passed or failed."""
        status_word = 'passed' if passed else 'failed'
        pattern = rf'([\w\.]+)\s+{status_word}'
        matches = re.finditer(pattern, error_report, re.IGNORECASE)
        return {match.group(1) for match in matches}

    def _analyze_code_diff(self, prev_code: str, curr_code: str) -> List[str]:
        """
        Analyze code changes and describe them in human-readable form.
        """
        changes = []

        # Use difflib to find differences
        prev_lines = prev_code.split('\n')
        curr_lines = curr_code.split('\n')

        diff = list(difflib.unified_diff(prev_lines, curr_lines, lineterm='', n=0))

        added_lines = []
        removed_lines = []

        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                added_lines.append(line[1:].strip())
            elif line.startswith('-') and not line.startswith('---'):
                removed_lines.append(line[1:].strip())

        # Characterize the changes
        if added_lines:
            # Look for specific patterns
            added_text = ' '.join(added_lines)

            if 'always' in added_text.lower():
                changes.append("Added new always block")
            if 'if' in added_text.lower() or 'case' in added_text.lower():
                changes.append("Added/modified conditional logic")
            if '<=' in added_text or '=' in added_text:
                # Extract signal names
                signal_matches = re.findall(r'(\w+)\s*[<]?=', added_text)
                if signal_matches:
                    # Convert to list before subscripting (defensive programming)
                    unique_signals = list(set(signal_matches))[:3]
                    changes.append(f"Modified assignment to: {', '.join(unique_signals)}")
            if 'state' in added_text.lower():
                changes.append("Modified state machine logic")
            if not changes:
                changes.append(f"Added {len(added_lines)} lines of code")

        if removed_lines:
            if not changes:  # Don't double-report if we already noted additions
                changes.append(f"Removed/replaced {len(removed_lines)} lines of code")

        # Detect refactoring (many lines changed)
        if len(added_lines) > 10 and len(removed_lines) > 10:
            changes = ["Major refactoring/rewrite"]

        return changes if changes else ["Minor code modifications"]

    def _normalize_fix_description(self, fix_description: str) -> str:
        """Normalize fix description for comparison."""
        # Convert to lowercase, remove extra whitespace
        normalized = re.sub(r'\s+', ' ', fix_description.lower().strip())

        # Extract key concepts
        key_words = ['add', 'remove', 'fix', 'change', 'modify', 'reset', 'signal', 'state', 'always', 'assign']
        concepts = []
        for word in key_words:
            if word in normalized:
                concepts.append(word)

        return ' '.join(concepts) if concepts else normalized[:50]

    def _update_feature_status(self, error_report: str, test_passed: bool):
        """Update what features are working vs broken."""
        # Extract test names
        passing_tests = self._extract_test_names(error_report, passed=True)
        failing_tests = self._extract_test_names(error_report, passed=False)

        # Update working features
        for test in passing_tests:
            if test not in self.working_features:
                self.working_features.append(test)
            if test in self.broken_features:
                self.broken_features.remove(test)

        # Update broken features
        for test in failing_tests:
            if test not in self.broken_features:
                self.broken_features.append(test)
            if test in self.working_features:
                self.working_features.remove(test)

    def _analyze_progress_trend(self) -> str:
        """Analyze whether we're making progress."""
        if len(self.iteration_history) < 2:
            return "Insufficient history to determine trend."

        # Look at test pass counts over last few iterations
        recent_counts = [r.tests_passed_count for r in self.iteration_history[-4:]]

        if len(recent_counts) >= 3:
            # Check if improving
            if recent_counts[-1] > recent_counts[0]:
                return f"**Improving**: Test passes increased from {recent_counts[0]} to {recent_counts[-1]} over recent iterations"
            elif recent_counts[-1] < recent_counts[0]:
                return f"**Declining**: Test passes decreased from {recent_counts[0]} to {recent_counts[-1]} - recent changes made things worse"
            else:
                # Check if stuck
                if len(set(recent_counts)) == 1:
                    return f"**Stuck**: Test passes remain at {recent_counts[0]} for {len(recent_counts)} iterations - need different approach"
                else:
                    return f"**Fluctuating**: Test passes varying between {min(recent_counts)} and {max(recent_counts)} - unstable changes"

        return "Monitoring progress..."

    def _detect_repeated_attempts(self) -> List[str]:
        """Detect if same fix has been tried multiple times."""
        repeated = []

        for fix_key, iterations in self.attempted_fixes.items():
            if len(iterations) >= 2:
                # Check if it worked
                worked = False
                for iter_num in iterations:
                    # Find the result
                    result = next((r for r in self.iteration_history if r.iteration == iter_num), None)
                    if result and result.improvements_noted:
                        worked = True
                        break

                if not worked:
                    repeated.append(f"Attempted '{fix_key}' in iterations {iterations} but did not improve results")

        return repeated

    def _generate_strategic_recommendation(self, current_iteration: int) -> str:
        """Generate a strategic recommendation based on history."""
        if not self.iteration_history:
            return "Begin debugging with systematic analysis of the primary failure."

        stats = self.get_progress_stats()

        # If improving, continue current approach
        if stats['improvement'] > 0:
            return "Current approach is working - continue with incremental fixes to remaining issues"

        # If stuck for many iterations
        if stats['total_iterations'] >= 5 and stats['improvement'] == 0:
            return "No progress in 5+ iterations - consider:\n" \
                   "  1. Re-read specification for misunderstood requirements\n" \
                   "  2. Complete rewrite of problematic module\n" \
                   "  3. Different implementation approach (e.g., FSM vs combinational)"

        # If regressing
        if stats['improvement'] < 0:
            return "Recent changes made things worse - consider reverting to earlier iteration and trying different fix"

        # If some tests pass but others fail
        if stats['current_passing'] > 0 and stats['current_failing'] > 0:
            return "Partial success - focus on fixing failing tests without breaking passing ones (check for edge cases)"

        # If all tests fail
        if stats['current_passing'] == 0:
            return "All tests failing - likely fundamental issue with:\n" \
                   "  1. Module interface (input/output signals)\n" \
                   "  2. Basic logic implementation\n" \
                   "  3. Specification understanding"

        return "Continue debugging systematically."
