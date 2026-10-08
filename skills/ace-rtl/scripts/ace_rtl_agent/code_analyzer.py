# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Code Analyzer - deep understanding of RTL code structure and logic.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re

@dataclass
class SignalFlow:
    """Represents how a signal flows through the design."""
    signal_name: str
    sources: List[str]  # Where signal is driven from
    destinations: List[str]  # Where signal is used
    timing: str  # "combinational", "clocked", "mixed"

@dataclass
class LogicBlock:
    """Represents a logical block of RTL code."""
    block_type: str  # "always_ff", "always_comb", "assign"
    signals_read: List[str]
    signals_written: List[str]
    conditions: List[str]  # if conditions, case statements
    timing_domain: str

class CodeAnalyzer:
    """
    Deep RTL code analysis.

    Understands RTL code structure, data flow, timing relationships,
    and logic dependencies - essential for making targeted fixes.
    """

    def __init__(self, logger):
        self.logger = logger

    def analyze_rtl_code(self, rtl_code: str) -> Dict[str, Any]:
        """
        Perform comprehensive RTL code analysis.

        Returns deep understanding of code structure, timing, and logic flow.
        """
        self.logger.info("Performing deep RTL code analysis")

        analysis = {
            "module_interface": self._analyze_module_interface(rtl_code),
            "signal_declarations": self._analyze_signal_declarations(rtl_code),
            "logic_blocks": self._analyze_logic_blocks(rtl_code),
            "signal_flow": self._analyze_signal_flow(rtl_code),
            "timing_analysis": self._analyze_timing_domains(rtl_code),
            "reset_analysis": self._analyze_reset_behavior(rtl_code),
            "potential_issues": self._identify_potential_issues(rtl_code)
        }

        return analysis

    def _analyze_module_interface(self, rtl_code: str) -> Dict[str, Any]:
        """Analyze module interface - ports and their characteristics."""

        interface = {
            "module_name": self._extract_module_name(rtl_code),
            "inputs": [],
            "outputs": [],
            "port_widths": {}
        }

        # Extract input ports with widths
        input_pattern = r"input\s+(?:logic\s+)?(?:\[(\d+):(\d+)\])?\s*(\w+)"
        for match in re.finditer(input_pattern, rtl_code):
            msb, lsb, name = match.groups()
            interface["inputs"].append(name)
            if msb and lsb:
                interface["port_widths"][name] = int(msb) - int(lsb) + 1
            else:
                interface["port_widths"][name] = 1

        # Extract output ports with widths
        output_pattern = r"output\s+(?:logic\s+)?(?:\[(\d+):(\d+)\])?\s*(\w+)"
        for match in re.finditer(output_pattern, rtl_code):
            msb, lsb, name = match.groups()
            interface["outputs"].append(name)
            if msb and lsb:
                interface["port_widths"][name] = int(msb) - int(lsb) + 1
            else:
                interface["port_widths"][name] = 1

        return interface

    def _analyze_signal_declarations(self, rtl_code: str) -> Dict[str, Any]:
        """Analyze internal signal declarations."""

        signals = {
            "registers": [],
            "wires": [],
            "signal_widths": {}
        }

        # Find logic declarations
        logic_pattern = r"logic\s+(?:\[(\d+):(\d+)\])?\s*(\w+)"
        for match in re.finditer(logic_pattern, rtl_code):
            msb, lsb, name = match.groups()

            # Skip input/output ports
            if re.search(rf"(?:input|output)\s+.*{name}", rtl_code):
                continue

            signals["registers"].append(name)
            if msb and lsb:
                signals["signal_widths"][name] = int(msb) - int(lsb) + 1
            else:
                signals["signal_widths"][name] = 1

        return signals

    def _analyze_logic_blocks(self, rtl_code: str) -> List[LogicBlock]:
        """Analyze logic blocks (always blocks, assignments)."""
        blocks = []

        # Find always_ff blocks
        always_ff_pattern = r"always_ff\s*@\([^)]+\)\s*begin(.*?)end"
        for match in re.finditer(always_ff_pattern, rtl_code, re.DOTALL):
            block_content = match.group(1)

            block = LogicBlock(
                block_type="always_ff",
                signals_read=self._extract_signals_read(block_content),
                signals_written=self._extract_signals_written(block_content),
                conditions=self._extract_conditions(block_content),
                timing_domain="clocked"
            )
            blocks.append(block)

        # Find always_comb blocks
        always_comb_pattern = r"always_comb\s*begin(.*?)end"
        for match in re.finditer(always_comb_pattern, rtl_code, re.DOTALL):
            block_content = match.group(1)

            block = LogicBlock(
                block_type="always_comb",
                signals_read=self._extract_signals_read(block_content),
                signals_written=self._extract_signals_written(block_content),
                conditions=self._extract_conditions(block_content),
                timing_domain="combinational"
            )
            blocks.append(block)

        return blocks

    def _extract_signals_read(self, block_content: str) -> List[str]:
        """Extract signals that are read in this block."""
        # Look for signal names on right-hand side of assignments
        read_signals = set()

        # Find in arithmetic expressions
        assignment_pattern = r"<=\s*([^;]+);"
        for match in re.finditer(assignment_pattern, block_content):
            rhs = match.group(1)
            # Extract identifiers (signal names)
            identifiers = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", rhs)
            read_signals.update(identifiers)

        # Find in conditions
        condition_pattern = r"if\s*\(\s*([^)]+)\)"
        for match in re.finditer(condition_pattern, block_content):
            condition = match.group(1)
            identifiers = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", condition)
            read_signals.update(identifiers)

        return list(read_signals)

    def _extract_signals_written(self, block_content: str) -> List[str]:
        """Extract signals that are written in this block."""
        written_signals = set()

        # Non-blocking assignments
        nb_pattern = r"(\w+)\s*<="
        written_signals.update(re.findall(nb_pattern, block_content))

        # Blocking assignments
        b_pattern = r"(\w+)\s*="
        written_signals.update(re.findall(b_pattern, block_content))

        return list(written_signals)

    def _extract_conditions(self, block_content: str) -> List[str]:
        """Extract conditional statements."""
        conditions = []

        # if conditions
        if_pattern = r"if\s*\(\s*([^)]+)\)"
        conditions.extend(re.findall(if_pattern, block_content))

        # case statements
        case_pattern = r"case\s*\(\s*([^)]+)\)"
        conditions.extend(re.findall(case_pattern, block_content))

        return conditions

    def _analyze_signal_flow(self, rtl_code: str) -> List[SignalFlow]:
        """Analyze how signals flow through the design."""
        # This is a simplified version - could be much more sophisticated
        flows = []

        # Extract all signal names
        all_signals = set()
        signal_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"
        all_signals.update(re.findall(signal_pattern, rtl_code))

        # For each signal, try to trace its flow
        for signal in all_signals:
            if signal in ['begin', 'end', 'if', 'else', 'case', 'logic', 'always_ff', 'always_comb']:
                continue

            sources = self._find_signal_sources(rtl_code, signal)
            destinations = self._find_signal_destinations(rtl_code, signal)
            timing = self._determine_signal_timing(rtl_code, signal)

            if sources or destinations:  # Only include if signal is actually used
                flows.append(SignalFlow(
                    signal_name=signal,
                    sources=sources,
                    destinations=destinations,
                    timing=timing
                ))

        return flows

    def _find_signal_sources(self, rtl_code: str, signal: str) -> List[str]:
        """Find where a signal is driven from."""
        sources = []

        # Look for assignments to this signal
        assignment_pattern = rf"{signal}\s*<=\s*([^;]+);"
        for match in re.finditer(assignment_pattern, rtl_code):
            rhs = match.group(1).strip()
            sources.append(f"assigned_from: {rhs}")

        return sources

    def _find_signal_destinations(self, rtl_code: str, signal: str) -> List[str]:
        """Find where a signal is used."""
        destinations = []

        # Look for uses in assignments (RHS)
        usage_pattern = rf"\w+\s*<=\s*[^;]*\b{signal}\b[^;]*;"
        matches = re.findall(usage_pattern, rtl_code)
        if matches:
            destinations.extend([f"used_in_assignment: {match}" for match in matches])

        return destinations

    def _determine_signal_timing(self, rtl_code: str, signal: str) -> str:
        """Determine if signal is clocked, combinational, etc."""
        if re.search(rf"{signal}\s*<=", rtl_code):
            return "clocked"
        elif re.search(rf"{signal}\s*=", rtl_code):
            return "combinational"
        else:
            return "unknown"

    def _analyze_timing_domains(self, rtl_code: str) -> Dict[str, Any]:
        """Analyze timing domains and clock relationships."""
        timing = {
            "clock_signals": [],
            "reset_signals": [],
            "has_async_reset": False,
            "has_sync_reset": False
        }

        # Find clock signals
        clock_pattern = r"@\s*\(\s*posedge\s+(\w+)"
        timing["clock_signals"] = list(set(re.findall(clock_pattern, rtl_code)))

        # Find reset signals
        reset_pattern = r"\b(rst|reset)\b"
        timing["reset_signals"] = list(set(re.findall(reset_pattern, rtl_code, re.IGNORECASE)))

        # Determine reset type
        if "always_ff" in rtl_code and "rst" in rtl_code:
            timing["has_sync_reset"] = True

        return timing

    def _analyze_reset_behavior(self, rtl_code: str) -> Dict[str, Any]:
        """Analyze reset behavior in detail."""
        reset_analysis = {
            "reset_signals": [],
            "reset_values": {},
            "reset_coverage": []
        }

        # Find reset blocks
        reset_block_pattern = r"if\s*\(\s*(rst|reset)\s*\)\s*begin(.*?)end"
        for match in re.finditer(reset_block_pattern, rtl_code, re.DOTALL | re.IGNORECASE):
            reset_signal, reset_content = match.groups()
            reset_analysis["reset_signals"].append(reset_signal)

            # Extract reset values
            reset_assignments = re.findall(r"(\w+)\s*<=\s*([^;]+);", reset_content)
            for signal, value in reset_assignments:
                reset_analysis["reset_values"][signal] = value.strip()
                reset_analysis["reset_coverage"].append(signal)

        return reset_analysis

    def _identify_potential_issues(self, rtl_code: str) -> List[str]:
        """Identify potential issues in the code."""
        issues = []

        # Check for common RTL issues
        if "always_comb" in rtl_code and "<=" in rtl_code:
            issues.append("Non-blocking assignment in combinational block")

        if "always_ff" in rtl_code and "=" in rtl_code and "<=" in rtl_code:
            issues.append("Mixed blocking and non-blocking assignments")

        # Check reset coverage
        reset_analysis = self._analyze_reset_behavior(rtl_code)
        interface = self._analyze_module_interface(rtl_code)

        for output in interface["outputs"]:
            if output not in reset_analysis["reset_coverage"]:
                issues.append(f"Output '{output}' not explicitly reset")

        return issues

    def _extract_module_name(self, rtl_code: str) -> str:
        """Extract module name."""
        match = re.search(r"module\s+(\w+)", rtl_code)
        return match.group(1) if match else "unknown"

    def get_fix_suggestions(self, analysis: Dict[str, Any], error_context: Dict[str, Any]) -> List[str]:
        """Generate general suggestions based on code analysis - no hardcoded patterns."""
        suggestions = []

        # General code quality issues only
        potential_issues = analysis.get("potential_issues", [])
        if potential_issues:
            suggestions.extend(potential_issues)

        # Basic structural observations
        reset_analysis = analysis.get("reset_analysis", {})
        if not reset_analysis.get("has_reset", False):
            suggestions.append("Consider if reset logic is properly implemented")

        return suggestions
