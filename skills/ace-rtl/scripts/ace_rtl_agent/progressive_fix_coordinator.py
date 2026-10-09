# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Progressive Fix Coordinator - Accumulates fixes across iterations instead of replacing them
"""

import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum


class FixCategory(Enum):
    """Categories of fixes that can be applied."""
    RESET_LOGIC = "reset_logic"
    HANDSHAKING = "handshaking"
    DATA_FORMATTING = "data_formatting"
    TIMING_CONTROL = "timing_control"
    STATE_MACHINE = "state_machine"
    SIGNAL_DECLARATIONS = "signal_declarations"
    COMPILATION_FIX = "compilation_fix"


@dataclass
class AppliedFix:
    """Represents a fix that has been successfully applied."""
    category: FixCategory
    description: str
    iteration_applied: int
    code_pattern: str  # The code pattern that implements this fix
    test_improvement: str  # What test improvement this fix achieved
    dependencies: List[str] = field(default_factory=list)  # Other fixes this depends on


@dataclass
class FixConflict:
    """Represents a conflict between fixes."""
    fix1: AppliedFix
    fix2: AppliedFix
    conflict_description: str
    resolution_strategy: str


class ProgressiveFixCoordinator:
    """
    Coordinates the accumulation of fixes across iterations.

    This ensures that when a new fix is applied, it doesn't break previously
    working fixes. Instead of replacing code, it merges fixes intelligently.
    """

    def __init__(self, logger):
        self.logger = logger
        self.applied_fixes: Dict[str, AppliedFix] = {}
        self.fix_history: List[AppliedFix] = []

    def analyze_new_fix_requirements(self, error_report: str, current_code: str,
                                   iteration: int) -> List[Dict[str, Any]]:
        """
        Analyze what new fixes are needed based on the current error report.

        Returns a list of required fixes with their categories and priorities.
        """
        required_fixes = []

        # Analyze compilation errors first (highest priority)
        compilation_fixes = self._analyze_compilation_fixes(error_report)
        required_fixes.extend(compilation_fixes)

        # Analyze reset-related issues
        reset_fixes = self._analyze_reset_fixes(error_report, current_code)
        required_fixes.extend(reset_fixes)

        # Analyze handshaking issues
        handshaking_fixes = self._analyze_handshaking_fixes(error_report, current_code)
        required_fixes.extend(handshaking_fixes)

        # Analyze data formatting issues
        data_fixes = self._analyze_data_formatting_fixes(error_report, current_code)
        required_fixes.extend(data_fixes)

        # Analyze timing issues
        timing_fixes = self._analyze_timing_fixes(error_report, current_code)
        required_fixes.extend(timing_fixes)

        # Sort by priority
        required_fixes.sort(key=lambda x: x.get('priority', 5))

        self.logger.info(f"Identified {len(required_fixes)} required fixes for iteration {iteration}")

        return required_fixes

    def generate_progressive_fix_strategy(self, required_fixes: List[Dict[str, Any]],
                                        current_code: str) -> str:
        """
        Generate a strategy for applying fixes progressively without breaking existing ones.
        """
        strategy = ["## PROGRESSIVE FIX STRATEGY", ""]

        # Show what's currently working
        if self.applied_fixes:
            strategy.append("### CURRENTLY WORKING (Must Maintain):")
            for fix_id, fix in self.applied_fixes.items():
                strategy.append(f"- {fix.category.value}: {fix.description}")
                strategy.append(f"  - Applied in iteration {fix.iteration_applied}")
                strategy.append(f"  - Test improvement: {fix.test_improvement}")
            strategy.append("")

        # Show what needs to be added
        if required_fixes:
            strategy.append("### NEW FIXES REQUIRED:")
            for i, fix in enumerate(required_fixes[:3]):  # Limit to top 3
                strategy.append(f"{i+1}. **{fix['category']}** (Priority {fix.get('priority', 'N/A')})")
                strategy.append(f"   - Issue: {fix['description']}")
                strategy.append(f"   - Strategy: {fix['strategy']}")
                if fix.get('conflicts'):
                    strategy.append(f"   - Conflicts: {fix['conflicts']}")
                strategy.append("")

        # Generate integration strategy
        strategy.extend(self._generate_integration_strategy(required_fixes, current_code))

        strategy.append("**CRITICAL RULE**: Each fix must be added WITHOUT removing existing working fixes.")
        strategy.append("If a fix would conflict with existing code, find a way to merge them.")

        return "\n".join(strategy)

    def record_successful_fix(self, fix_category: FixCategory, description: str,
                            code_pattern: str, test_improvement: str, iteration: int):
        """
        Record a fix that was successfully applied and improved the test results.
        """
        fix_id = f"{fix_category.value}_{iteration}"

        applied_fix = AppliedFix(
            category=fix_category,
            description=description,
            iteration_applied=iteration,
            code_pattern=code_pattern,
            test_improvement=test_improvement
        )

        self.applied_fixes[fix_id] = applied_fix
        self.fix_history.append(applied_fix)

        self.logger.info(f"Recorded successful fix: {description}")

    def check_fix_conflicts(self, new_fix_requirements: List[Dict[str, Any]]) -> List[FixConflict]:
        """
        Check if new fix requirements conflict with existing fixes.
        """
        conflicts = []

        for new_fix in new_fix_requirements:
            new_category = new_fix.get('category')

            for existing_fix in self.applied_fixes.values():
                conflict = self._detect_fix_conflict(new_fix, existing_fix)
                if conflict:
                    conflicts.append(conflict)

        return conflicts

    def generate_code_merge_instructions(self, current_code: str,
                                       new_fixes: List[Dict[str, Any]]) -> str:
        """
        Generate specific instructions for merging new fixes with existing code.
        """
        instructions = ["## CODE MERGE INSTRUCTIONS", ""]

        # Analyze current code structure
        code_sections = self._analyze_code_sections(current_code)

        instructions.append("### Current Code Structure:")
        for section, content in code_sections.items():
            instructions.append(f"- {section}: {'Present' if content else 'Missing'}")
        instructions.append("")

        # Generate merge instructions for each fix
        for fix in new_fixes:
            instructions.extend(self._generate_fix_merge_instructions(fix, code_sections))

        # Add preservation instructions
        instructions.append("### PRESERVATION REQUIREMENTS:")
        for fix in self.applied_fixes.values():
            instructions.append(f"- MUST preserve: {fix.description}")
            if fix.code_pattern:
                instructions.append(f"  - Pattern: {fix.code_pattern}")

        return "\n".join(instructions)

    def _analyze_compilation_fixes(self, error_report: str) -> List[Dict[str, Any]]:
        """Analyze compilation errors and suggest fixes."""
        fixes = []

        # Check for common compilation errors
        if "syntax error" in error_report.lower():
            fixes.append({
                'category': FixCategory.COMPILATION_FIX,
                'description': 'Fix syntax errors',
                'priority': 1,
                'strategy': 'Check semicolons, parentheses, and keyword usage',
                'conflicts': None
            })

        if "undeclared" in error_report.lower() or "not declared" in error_report.lower():
            fixes.append({
                'category': FixCategory.SIGNAL_DECLARATIONS,
                'description': 'Add missing signal declarations',
                'priority': 1,
                'strategy': 'Declare all used signals with proper types',
                'conflicts': None
            })

        return fixes

    def _analyze_reset_fixes(self, error_report: str, current_code: str) -> List[Dict[str, Any]]:
        """Analyze reset-related issues."""
        fixes = []

        # Check if reset is mentioned in errors
        if "reset" in error_report.lower():
            # Check if there's already a reset fix applied
            has_reset_fix = any(fix.category == FixCategory.RESET_LOGIC
                              for fix in self.applied_fixes.values())

            if not has_reset_fix:
                # Look for specific reset issues
                if "s_axis_ready" in error_report and "expected.*0.*got.*1" in error_report:
                    fixes.append({
                        'category': FixCategory.RESET_LOGIC,
                        'description': 'Add reset protection for ready signals',
                        'priority': 2,
                        'strategy': 'Ensure ready signals are 0 immediately after reset',
                        'conflicts': 'May conflict with handshaking logic'
                    })

        return fixes

    def _analyze_handshaking_fixes(self, error_report: str, current_code: str) -> List[Dict[str, Any]]:
        """Analyze handshaking protocol issues."""
        fixes = []

        # Look for handshaking-related errors
        handshaking_signals = ['valid', 'ready', 'data']
        if any(signal in error_report.lower() for signal in handshaking_signals):

            # Check if handshaking is already implemented
            has_handshaking_fix = any(fix.category == FixCategory.HANDSHAKING
                                    for fix in self.applied_fixes.values())

            if not has_handshaking_fix:
                if "m_axis_valid" in error_report and "expected.*1.*got.*0" in error_report:
                    fixes.append({
                        'category': FixCategory.HANDSHAKING,
                        'description': 'Implement proper AXI Stream handshaking',
                        'priority': 3,
                        'strategy': 'Ensure valid/ready handshaking works correctly',
                        'conflicts': 'May interact with reset logic'
                    })

        return fixes

    def _analyze_data_formatting_fixes(self, error_report: str, current_code: str) -> List[Dict[str, Any]]:
        """Analyze data formatting issues."""
        fixes = []

        # Look for data formatting errors
        if "m_axis_data" in error_report and "not matching" in error_report.lower():
            has_data_fix = any(fix.category == FixCategory.DATA_FORMATTING
                             for fix in self.applied_fixes.values())

            if not has_data_fix:
                fixes.append({
                    'category': FixCategory.DATA_FORMATTING,
                    'description': 'Fix data formatting logic',
                    'priority': 4,
                    'strategy': 'Correct bit field assignments and data transformation',
                    'conflicts': None
                })

        return fixes

    def _analyze_timing_fixes(self, error_report: str, current_code: str) -> List[Dict[str, Any]]:
        """Analyze timing-related issues."""
        fixes = []

        # Look for timing-related patterns
        timing_patterns = [r'\d+\.?\d*ns', 'clock', 'cycle']
        if any(re.search(pattern, error_report, re.IGNORECASE) for pattern in timing_patterns):

            has_timing_fix = any(fix.category == FixCategory.TIMING_CONTROL
                               for fix in self.applied_fixes.values())

            if not has_timing_fix:
                fixes.append({
                    'category': FixCategory.TIMING_CONTROL,
                    'description': 'Add timing control logic',
                    'priority': 5,
                    'strategy': 'Add proper timing delays or state management',
                    'conflicts': 'May interact with reset and handshaking'
                })

        return fixes

    def _detect_fix_conflict(self, new_fix: Dict[str, Any], existing_fix: AppliedFix) -> Optional[FixConflict]:
        """Detect if a new fix conflicts with an existing one."""
        new_category = new_fix.get('category')

        # Define known conflicts
        conflicts = {
            FixCategory.RESET_LOGIC: [FixCategory.HANDSHAKING, FixCategory.TIMING_CONTROL],
            FixCategory.HANDSHAKING: [FixCategory.RESET_LOGIC],
            FixCategory.TIMING_CONTROL: [FixCategory.RESET_LOGIC, FixCategory.HANDSHAKING],
        }

        if new_category in conflicts and existing_fix.category in conflicts[new_category]:
            return FixConflict(
                fix1=existing_fix,
                fix2=AppliedFix(new_category, new_fix.get('description', ''), 0, '', ''),
                conflict_description=f"{new_category.value} may conflict with {existing_fix.category.value}",
                resolution_strategy="Merge the fixes instead of replacing"
            )

        return None

    def _generate_integration_strategy(self, required_fixes: List[Dict[str, Any]],
                                     current_code: str) -> List[str]:
        """Generate strategy for integrating new fixes with existing code."""
        strategy = ["### INTEGRATION STRATEGY:"]

        if not required_fixes:
            strategy.append("- No new fixes required")
            return strategy

        # Group fixes by category
        by_category = {}
        for fix in required_fixes:
            category = fix.get('category')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(fix)

        # Generate integration steps
        strategy.append("1. **Preserve existing working code**")
        strategy.append("2. **Add new functionality incrementally**:")

        for category, fixes in by_category.items():
            strategy.append(f"   - {category.value}: {len(fixes)} fix(es)")
            for fix in fixes:
                strategy.append(f"     - {fix['description']}")

        strategy.append("3. **Test integration at each step**")
        strategy.append("4. **Verify no regressions occur**")
        strategy.append("")

        return strategy

    def _analyze_code_sections(self, code: str) -> Dict[str, bool]:
        """Analyze which sections are present in the current code."""
        sections = {
            'module_declaration': bool(re.search(r'module\s+\w+', code)),
            'signal_declarations': bool(re.search(r'(logic|reg|wire)\s+', code)),
            'combinational_logic': bool(re.search(r'always_comb|always\s*@\(\*\)', code)),
            'sequential_logic': bool(re.search(r'always_ff|always\s*@\(posedge', code)),
            'reset_logic': bool(re.search(r'if\s*\(\s*~?\w*reset', code, re.IGNORECASE)),
            'handshaking_logic': bool(re.search(r'(valid|ready)', code, re.IGNORECASE)),
        }

        return sections

    def _generate_fix_merge_instructions(self, fix: Dict[str, Any],
                                       code_sections: Dict[str, bool]) -> List[str]:
        """Generate specific merge instructions for a fix."""
        instructions = [f"### Merge Instructions for {fix['category'].value}:"]

        category = fix['category']

        if category == FixCategory.RESET_LOGIC:
            if code_sections.get('sequential_logic'):
                instructions.append("- ADD reset condition to existing always_ff block")
                instructions.append("- DO NOT replace the entire block")
            else:
                instructions.append("- ADD new always_ff block with reset logic")

        elif category == FixCategory.HANDSHAKING:
            if code_sections.get('combinational_logic'):
                instructions.append("- EXTEND existing combinational logic for handshaking")
            else:
                instructions.append("- ADD new combinational logic for handshaking")

        elif category == FixCategory.DATA_FORMATTING:
            if code_sections.get('combinational_logic'):
                instructions.append("- MODIFY data formatting within existing combinational block")
            else:
                instructions.append("- ADD data formatting logic")

        else:
            instructions.append(f"- ADD {category.value} logic without removing existing code")

        instructions.append("")
        return instructions
