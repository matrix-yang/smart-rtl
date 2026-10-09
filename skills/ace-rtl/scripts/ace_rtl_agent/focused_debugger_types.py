# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused debugger shared data types."""

from dataclasses import dataclass


@dataclass
class FixAttempt:
    """Records a fix attempt and its outcome."""
    iteration: int
    focused_error: str  # The specific error focused on
    error_reasoning: str  # Deep reasoning about what the error means
    guidance_tried: str  # The specific guidance/hypothesis that was tried
    implementation_changes: str  # Actual code changes made to implement the guidance
    fix_effectiveness: str  # Whether the fix solved the target error
    test_results: str  # Test results after this fix (e.g., "0/10 passed", "3/10 passed")
    outcome: str  # "improved", "no_change", "regressed", "solved"
    mismatch_count: int = -1  # Numeric mismatch count from simulation (-1 = unknown)
