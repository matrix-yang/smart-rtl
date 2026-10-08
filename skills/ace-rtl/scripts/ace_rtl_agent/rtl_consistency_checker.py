# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RTL Consistency Checker - Ensures consistent RTL coding style and prevents common syntax issues
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class IssueType(Enum):
    """Types of RTL consistency issues."""
    SYNTAX_ERROR = "syntax_error"
    STYLE_INCONSISTENCY = "style_inconsistency"
    LOGIC_ANTIPATTERN = "logic_antipattern"
    TIMING_ISSUE = "timing_issue"
    RESET_PATTERN = "reset_pattern"


@dataclass
class ConsistencyIssue:
    """Represents a consistency issue found in RTL code."""
    issue_type: IssueType
    severity: str  # HIGH, MEDIUM, LOW
    description: str
    line_number: Optional[int]
    suggestion: str
    code_snippet: str


@dataclass
class RTLTemplate:
    """Template for consistent RTL structure."""
    module_header: str
    signal_declarations: str
    combinational_logic: str
    sequential_logic: str
    reset_pattern: str


class RTLConsistencyChecker:
    """
    Checks RTL code for consistency issues and provides templates for common patterns.

    This prevents the agent from introducing syntax errors or inconsistent coding styles
    that cause compilation failures or make the code harder to debug.
    """

    def __init__(self, logger):
        self.logger = logger
        self.preferred_style = "systemverilog"  # vs "verilog"

    def check_rtl_consistency(self, rtl_code: str) -> List[ConsistencyIssue]:
        """
        Check RTL code for consistency issues.

        Returns a list of issues found, prioritized by severity.
        """
        issues = []

        # Check syntax consistency
        issues.extend(self._check_syntax_consistency(rtl_code))

        # Check signal declaration consistency
        issues.extend(self._check_signal_declarations(rtl_code))

        # Check always block patterns
        issues.extend(self._check_always_block_patterns(rtl_code))

        # Check reset patterns
        issues.extend(self._check_reset_patterns(rtl_code))

        # Check common antipatterns
        issues.extend(self._check_common_antipatterns(rtl_code))

        # Sort by severity
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        issues.sort(key=lambda x: severity_order.get(x.severity, 3))

        return issues

    def generate_rtl_template(self, module_name: str, interface_signals: Dict[str, Any]) -> RTLTemplate:
        """
        Generate a consistent RTL template based on the interface requirements.

        This helps ensure the agent uses consistent patterns.
        """
        # Generate module header
        module_header = self._generate_module_header(module_name, interface_signals)

        # Generate signal declarations
        signal_declarations = self._generate_signal_declarations(interface_signals)

        # Generate combinational logic template
        combinational_logic = self._generate_combinational_template()

        # Generate sequential logic template
        sequential_logic = self._generate_sequential_template()

        # Generate reset pattern
        reset_pattern = self._generate_reset_pattern()

        return RTLTemplate(
            module_header=module_header,
            signal_declarations=signal_declarations,
            combinational_logic=combinational_logic,
            sequential_logic=sequential_logic,
            reset_pattern=reset_pattern
        )

    def suggest_fixes(self, issues: List[ConsistencyIssue]) -> str:
        """
        Generate specific fix suggestions for the found issues.
        """
        if not issues:
            return "No consistency issues found. RTL code follows good practices."

        suggestions = ["## RTL CONSISTENCY FIXES REQUIRED", ""]

        high_priority = [issue for issue in issues if issue.severity == "HIGH"]
        medium_priority = [issue for issue in issues if issue.severity == "MEDIUM"]
        low_priority = [issue for issue in issues if issue.severity == "LOW"]

        if high_priority:
            suggestions.append("### HIGH PRIORITY FIXES (Must fix before compilation):")
            for issue in high_priority:
                suggestions.append(f"- **{issue.issue_type.value}**: {issue.description}")
                suggestions.append(f"  - Fix: {issue.suggestion}")
                if issue.code_snippet:
                    suggestions.append(f"  - Code: `{issue.code_snippet}`")
                suggestions.append("")

        if medium_priority:
            suggestions.append("### MEDIUM PRIORITY FIXES (Recommended):")
            for issue in medium_priority:
                suggestions.append(f"- **{issue.issue_type.value}**: {issue.description}")
                suggestions.append(f"  - Fix: {issue.suggestion}")
                suggestions.append("")

        if low_priority:
            suggestions.append("### LOW PRIORITY FIXES (Style improvements):")
            for issue in low_priority:
                suggestions.append(f"- **{issue.issue_type.value}**: {issue.description}")
                suggestions.append(f"  - Fix: {issue.suggestion}")
                suggestions.append("")

        suggestions.append("**IMPORTANT**: Fix HIGH priority issues first as they will cause compilation failures.")

        return "\n".join(suggestions)

    def _check_syntax_consistency(self, rtl_code: str) -> List[ConsistencyIssue]:
        """Check for basic syntax consistency issues."""
        issues = []
        lines = rtl_code.split('\n')

        # Check for mixed logic/reg/wire usage
        has_logic = bool(re.search(r'\blogic\b', rtl_code))
        has_reg = bool(re.search(r'\breg\b', rtl_code))
        has_wire = bool(re.search(r'\bwire\b', rtl_code))

        if has_logic and (has_reg or has_wire):
            issues.append(ConsistencyIssue(
                issue_type=IssueType.STYLE_INCONSISTENCY,
                severity="MEDIUM",
                description="Mixed usage of 'logic' and 'reg/wire' declarations",
                line_number=None,
                suggestion="Use 'logic' consistently for SystemVerilog or 'reg/wire' for Verilog",
                code_snippet=""
            ))

        # Check for always_comb with reg declarations
        if re.search(r'always_comb', rtl_code) and has_reg:
            issues.append(ConsistencyIssue(
                issue_type=IssueType.SYNTAX_ERROR,
                severity="HIGH",
                description="Using 'always_comb' with 'reg' declarations causes compilation errors",
                line_number=None,
                suggestion="Use 'logic' declarations with 'always_comb' or 'always @(*)' with 'reg'",
                code_snippet=""
            ))

        # Check for missing semicolons
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if (line_stripped and
                not line_stripped.startswith('//') and
                not line_stripped.endswith(';') and
                not line_stripped.endswith('{') and
                not line_stripped.endswith('}') and
                not line_stripped.startswith('end') and
                not line_stripped.startswith('begin') and
                re.search(r'(assign|logic|reg|wire|input|output)', line_stripped)):

                issues.append(ConsistencyIssue(
                    issue_type=IssueType.SYNTAX_ERROR,
                    severity="HIGH",
                    description=f"Missing semicolon on line {i+1}",
                    line_number=i+1,
                    suggestion="Add semicolon at end of statement",
                    code_snippet=line_stripped
                ))

        return issues

    def _check_signal_declarations(self, rtl_code: str) -> List[ConsistencyIssue]:
        """Check signal declaration consistency."""
        issues = []

        # Check for consistent bit width declarations
        signal_patterns = [
            r'(logic|reg|wire)\s+(\[[\d:]+\])?\s*(\w+)',
            r'(input|output)\s+(logic|reg|wire)?\s*(\[[\d:]+\])?\s*(\w+)'
        ]

        declared_signals = {}

        for pattern in signal_patterns:
            matches = re.finditer(pattern, rtl_code)
            for match in matches:
                signal_name = match.group(-1)  # Last group is signal name
                bit_width = match.group(-2) if len(match.groups()) > 2 else None
                declared_signals[signal_name] = bit_width

        # Check for undeclared signals used in assignments
        assignment_patterns = [
            r'assign\s+(\w+)',
            r'(\w+)\s*<=',
            r'(\w+)\s*='
        ]

        used_signals = set()
        for pattern in assignment_patterns:
            matches = re.finditer(pattern, rtl_code)
            for match in matches:
                used_signals.add(match.group(1))

        # Find signals used but not declared
        undeclared = used_signals - set(declared_signals.keys())
        for signal in undeclared:
            if not signal.isdigit():  # Skip numeric constants
                issues.append(ConsistencyIssue(
                    issue_type=IssueType.SYNTAX_ERROR,
                    severity="HIGH",
                    description=f"Signal '{signal}' used but not declared",
                    line_number=None,
                    suggestion=f"Add declaration: logic {signal};",
                    code_snippet=f"{signal} <="
                ))

        return issues

    def _check_always_block_patterns(self, rtl_code: str) -> List[ConsistencyIssue]:
        """Check always block patterns for consistency."""
        issues = []

        # Check for proper always block usage
        always_blocks = re.finditer(r'always(_ff|_comb)?\s*@\(([^)]+)\)', rtl_code)

        for match in always_blocks:
            block_type = match.group(1)
            sensitivity = match.group(2)

            # Check combinational blocks
            if block_type == '_comb' or sensitivity == '*':
                # Should not have clock edges
                if re.search(r'posedge|negedge', sensitivity):
                    issues.append(ConsistencyIssue(
                        issue_type=IssueType.LOGIC_ANTIPATTERN,
                        severity="HIGH",
                        description="Combinational always block should not use clock edges",
                        line_number=None,
                        suggestion="Use always_comb or always @(*) for combinational logic",
                        code_snippet=match.group(0)
                    ))

            # Check sequential blocks
            elif block_type == '_ff' or 'posedge' in sensitivity:
                # Should have proper reset pattern
                block_content = self._extract_always_block_content(rtl_code, match.end())
                if 'reset' in rtl_code.lower() and 'if' not in block_content:
                    issues.append(ConsistencyIssue(
                        issue_type=IssueType.RESET_PATTERN,
                        severity="MEDIUM",
                        description="Sequential block missing reset condition",
                        line_number=None,
                        suggestion="Add proper reset condition: if (~resetn) begin ... end",
                        code_snippet=""
                    ))

        return issues

    def _check_reset_patterns(self, rtl_code: str) -> List[ConsistencyIssue]:
        """Check reset pattern consistency."""
        issues = []

        # Check for consistent reset polarity
        reset_patterns = [
            r'if\s*\(\s*~?(\w*reset\w*)\s*\)',
            r'(\w*reset\w*)\s*==\s*[01]',
        ]

        reset_signals = []
        for pattern in reset_patterns:
            matches = re.finditer(pattern, rtl_code, re.IGNORECASE)
            for match in matches:
                reset_signals.append(match.group(1))

        # Check for mixed reset polarities
        active_high = any('~' not in usage for usage in reset_signals)
        active_low = any('~' in usage for usage in reset_signals)

        if active_high and active_low:
            issues.append(ConsistencyIssue(
                issue_type=IssueType.RESET_PATTERN,
                severity="MEDIUM",
                description="Mixed reset polarities detected",
                line_number=None,
                suggestion="Use consistent reset polarity (recommend active-low: ~resetn)",
                code_snippet=""
            ))

        return issues

    def _check_common_antipatterns(self, rtl_code: str) -> List[ConsistencyIssue]:
        """Check for common RTL antipatterns."""
        issues = []

        # Check for blocking assignments in sequential blocks
        sequential_blocks = re.finditer(r'always_ff.*?end(?:module|\s+always)', rtl_code, re.DOTALL)
        for block in sequential_blocks:
            block_content = block.group(0)
            if re.search(r'\w+\s*=\s*[^=]', block_content):  # Blocking assignment
                issues.append(ConsistencyIssue(
                    issue_type=IssueType.LOGIC_ANTIPATTERN,
                    severity="HIGH",
                    description="Blocking assignment (=) in sequential always block",
                    line_number=None,
                    suggestion="Use non-blocking assignment (<=) in sequential blocks",
                    code_snippet=""
                ))

        # Check for non-blocking assignments in combinational blocks
        comb_blocks = re.finditer(r'always_comb.*?end(?:module|\s+always)', rtl_code, re.DOTALL)
        for block in comb_blocks:
            block_content = block.group(0)
            if '<=' in block_content:
                issues.append(ConsistencyIssue(
                    issue_type=IssueType.LOGIC_ANTIPATTERN,
                    severity="HIGH",
                    description="Non-blocking assignment (<=) in combinational block",
                    line_number=None,
                    suggestion="Use blocking assignment (=) in combinational blocks",
                    code_snippet=""
                ))

        return issues

    def _generate_module_header(self, module_name: str, interface_signals: Dict[str, Any]) -> str:
        """Generate consistent module header."""
        lines = [f"module {module_name} ("]

        # Add clock and reset first (if present)
        if 'clk' in interface_signals:
            lines.append("    input  logic        clk,")
        if 'resetn' in interface_signals or 'reset' in interface_signals:
            reset_name = 'resetn' if 'resetn' in interface_signals else 'reset'
            lines.append(f"    input  logic        {reset_name},")

        # Add other signals
        for signal_name, signal_info in interface_signals.items():
            if signal_name not in ['clk', 'resetn', 'reset']:
                direction = signal_info.get('direction', 'input')
                width = signal_info.get('width', '')
                width_str = f" {width}" if width else "        "
                lines.append(f"    {direction}  logic{width_str} {signal_name},")

        # Remove last comma and close
        if lines[-1].endswith(','):
            lines[-1] = lines[-1][:-1]
        lines.append(");")

        return "\n".join(lines)

    def _generate_signal_declarations(self, interface_signals: Dict[str, Any]) -> str:
        """Generate internal signal declarations."""
        return """    // Internal signals
    logic        internal_valid;
    logic [31:0] internal_data;

    // Add other internal signals as needed"""

    def _generate_combinational_template(self) -> str:
        """Generate combinational logic template."""
        return """    // Combinational logic
    always_comb begin
        // Add combinational assignments here
        // Use blocking assignments (=)
    end"""

    def _generate_sequential_template(self) -> str:
        """Generate sequential logic template."""
        return """    // Sequential logic
    always_ff @(posedge clk) begin
        if (~resetn) begin
            // Reset assignments
            // Use non-blocking assignments (<=)
        end else begin
            // Normal operation
            // Use non-blocking assignments (<=)
        end
    end"""

    def _generate_reset_pattern(self) -> str:
        """Generate consistent reset pattern."""
        return """    // Reset pattern (active-low synchronous)
    if (~resetn) begin
        signal <= 1'b0;
        data   <= '0;
    end else begin
        // Normal operation
    end"""

    def _extract_always_block_content(self, rtl_code: str, start_pos: int) -> str:
        """Extract the content of an always block starting at position."""
        # Simple extraction - find the matching 'end'
        lines = rtl_code[start_pos:].split('\n')
        content = []
        begin_count = 0

        for line in lines:
            content.append(line)
            begin_count += line.count('begin')
            begin_count -= line.count('end')
            if begin_count <= 0 and 'end' in line:
                break

        return '\n'.join(content)
