# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Iteration Memory - learns from previous failed attempts.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import json

@dataclass
class FailedAttempt:
    """Record of a failed debugging attempt."""
    iteration: int
    error_type: str
    attempted_fix: str
    reasoning: str
    result: str  # "failed" or reason for failure
    code_before: str
    code_after: str

class IterationMemory:
    """
    Maintains memory of previous attempts to avoid repeating mistakes.

    It learns from what did not work and adjusts the next approach.
    """

    def __init__(self, logger):
        self.logger = logger
        self.failed_attempts: List[FailedAttempt] = []
        self.avoided_patterns: Set[str] = set()
        self.successful_patterns: Set[str] = set()

    def record_failed_attempt(self, iteration: int, error_type: str,
                            attempted_fix: str, reasoning: str,
                            code_before: str, code_after: str, result: str):
        """Record a failed attempt for future learning."""

        attempt = FailedAttempt(
            iteration=iteration,
            error_type=error_type,
            attempted_fix=attempted_fix,
            reasoning=reasoning,
            result=result,
            code_before=code_before,
            code_after=code_after
        )

        self.failed_attempts.append(attempt)
        self._extract_pattern_to_avoid(attempted_fix, reasoning)

        self.logger.info(f"Recorded failed attempt {iteration}: {reasoning}")

    def record_successful_attempt(self, attempted_fix: str, reasoning: str):
        """Record a successful pattern for future use."""
        pattern = self._extract_pattern(attempted_fix, reasoning)
        self.successful_patterns.add(pattern)
        self.logger.info(f"Recorded successful pattern: {pattern}")

    def _extract_pattern_to_avoid(self, attempted_fix: str, reasoning: str):
        """Extract patterns that should be avoided in future."""
        pattern = self._extract_pattern(attempted_fix, reasoning)
        self.avoided_patterns.add(pattern)

    def _extract_pattern(self, attempted_fix: str, reasoning: str) -> str:
        """Extract a general pattern from the fix and reasoning."""
        # Simple pattern extraction - could be enhanced
        if "output" in attempted_fix.lower() and "reset" in reasoning.lower():
            return "reset_output_modification"
        elif "always_ff" in attempted_fix and "timing" in reasoning.lower():
            return "clocked_logic_timing"
        elif re.search(r'\b\w+\s*<=\s*\w+', attempted_fix):
            return "register_assignment_change"
        else:
            return "generic_fix_attempt"

    def should_avoid_approach(self, potential_fix: str, reasoning: str) -> bool:
        """Check if this approach has been tried and failed before."""
        pattern = self._extract_pattern(potential_fix, reasoning)

        if pattern in self.avoided_patterns:
            self.logger.warning(f"Avoiding previously failed pattern: {pattern}")
            return True

        # Check for similar failed attempts
        for attempt in self.failed_attempts:
            if (self._similarity_score(attempt.attempted_fix, potential_fix) > 0.7 or
                self._similarity_score(attempt.reasoning, reasoning) > 0.7):
                self.logger.warning(f"Similar to failed attempt {attempt.iteration}")
                return True

        return False

    def _similarity_score(self, text1: str, text2: str) -> float:
        """Simple similarity scoring between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from previous attempts for next iteration."""
        if not self.failed_attempts:
            return {"message": "No previous attempts to learn from"}

        insights = {
            "total_attempts": len(self.failed_attempts),
            "avoided_patterns": list(self.avoided_patterns),
            "successful_patterns": list(self.successful_patterns),
            "common_error_types": self._get_common_error_types(),
            "recommendations": self._generate_recommendations()
        }

        return insights

    def _get_common_error_types(self) -> Dict[str, int]:
        """Get counts of common error types."""
        error_counts = {}
        for attempt in self.failed_attempts:
            error_type = attempt.error_type
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on learning."""
        recommendations = []

        if len(self.failed_attempts) >= 3:
            recommendations.append("Consider a different debugging approach - multiple similar attempts have failed")

        if "reset_output_modification" in self.avoided_patterns:
            recommendations.append("Avoid modifying output assignments in reset - this pattern has failed before")

        if "register_assignment_change" in self.avoided_patterns:
            recommendations.append("Register assignment changes have failed - consider timing or logic issues instead")

        return recommendations

    def get_context_for_llm(self) -> str:
        """Get context string to include in LLM prompts."""
        if not self.failed_attempts:
            return "No previous attempts to learn from."

        context_parts = [f"LEARNING FROM {len(self.failed_attempts)} PREVIOUS FAILED ATTEMPTS:"]

        # Include all failed attempts for complete learning context
        recent_attempts = self.failed_attempts

        for i, attempt in enumerate(recent_attempts, 1):
            context_parts.append(f"\nAttempt {attempt.iteration}: {attempt.reasoning}")
            context_parts.append(f"Result: {attempt.result}")

        if self.avoided_patterns:
            context_parts.append(f"\nAVOID these approaches that failed: {', '.join(self.avoided_patterns)}")

        if self.successful_patterns:
            context_parts.append(f"\nSUCCESSFUL approaches to consider: {', '.join(self.successful_patterns)}")

        recommendations = self._generate_recommendations()
        if recommendations:
            context_parts.append(f"\nRECOMMENDATIONS: {'; '.join(recommendations)}")

        return "\n".join(context_parts)

    def save_to_file(self, filepath: str):
        """Save iteration memory to file for persistence."""
        data = {
            "failed_attempts": [asdict(attempt) for attempt in self.failed_attempts],
            "avoided_patterns": list(self.avoided_patterns),
            "successful_patterns": list(self.successful_patterns)
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str):
        """Load iteration memory from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.failed_attempts = [FailedAttempt(**attempt) for attempt in data.get("failed_attempts", [])]
            self.avoided_patterns = set(data.get("avoided_patterns", []))
            self.successful_patterns = set(data.get("successful_patterns", []))

        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Could not load iteration memory: {e}")
