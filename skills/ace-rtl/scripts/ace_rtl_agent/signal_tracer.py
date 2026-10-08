# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Signal Tracer - Analyzes RTL code to trace signal paths and identify issues.

This is a general-purpose RTL analyzer that works across different design types.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class SignalInfo:
    """Information about a signal in RTL code."""
    name: str
    signal_type: str  # 'input', 'output', 'reg', 'wire', 'logic'
    bit_width: Optional[str] = None
    assigned_in_lines: List[int] = None
    depends_on: List[str] = None  # Signals this depends on
    assignment_type: Optional[str] = None  # 'blocking', 'nonblocking', 'continuous'
    in_always_block: bool = False
    always_block_sensitivity: Optional[str] = None

    def __post_init__(self):
        if self.assigned_in_lines is None:
            self.assigned_in_lines = []
        if self.depends_on is None:
            self.depends_on = []


@dataclass
class AlwaysBlockInfo:
    """Information about an always block."""
    start_line: int
    end_line: int
    sensitivity_list: str
    block_type: str  # 'sequential', 'combinational', 'latch'
    signals_assigned: List[str]
    signals_read: List[str]


class SignalTracer:
    """
    Traces signal paths through RTL code to help identify issues.

    General approach (not design-specific):
    1. Parse signal declarations
    2. Track assignments and dependencies
    3. Identify control flow (always blocks, state machines)
    4. Trace paths from inputs to outputs
    """

    def __init__(self, logger):
        self.logger = logger
        self.signals: Dict[str, SignalInfo] = {}
        self.always_blocks: List[AlwaysBlockInfo] = []
        self.state_machine_detected = False
        self.state_signal: Optional[str] = None
        self.states: Set[str] = set()

    def analyze_rtl(self, rtl_code: str) -> Dict:
        """
        Analyze RTL code structure.

        Returns a general analysis applicable to any RTL design.
        """
        self.signals = {}
        self.always_blocks = []
        self.state_machine_detected = False
        self.state_signal = None
        self.states = set()

        lines = rtl_code.split('\n')

        # Step 1: Parse signal declarations
        self._parse_signal_declarations(lines)

        # Step 2: Identify always blocks
        self._identify_always_blocks(lines)

        # Step 3: Trace signal assignments
        self._trace_assignments(lines)

        # Step 4: Detect state machines (common in RTL)
        self._detect_state_machine(lines)

        # Step 5: Build dependency graph
        self._build_dependencies(lines)

        return {
            'signals': self.signals,
            'always_blocks': self.always_blocks,
            'state_machine_detected': self.state_machine_detected,
            'state_signal': self.state_signal,
            'states': list(self.states)
        }

    def trace_signal_path(self, signal_name: str, rtl_code: str) -> str:
        """
        Trace how a specific signal is computed.

        General approach works for any signal in any design.
        """
        if signal_name not in self.signals:
            return f"Signal '{signal_name}' not found in RTL code."

        signal_info = self.signals[signal_name]
        lines = rtl_code.split('\n')

        trace_parts = [f"## Signal Trace for '{signal_name}'"]

        # Basic info
        decl_str = f"\n**Declaration**: {signal_info.signal_type}"
        if signal_info.bit_width:
            decl_str += f" {signal_info.bit_width}"
        trace_parts.append(decl_str)

        # Where it's assigned
        if signal_info.assigned_in_lines:
            trace_parts.append(f"\n**Assigned at lines**: {', '.join(map(str, signal_info.assigned_in_lines))}")
            trace_parts.append(f"**Assignment type**: {signal_info.assignment_type or 'Unknown'}")

            # Show the actual assignments
            trace_parts.append("\n**Assignment statements**:")
            for line_num in signal_info.assigned_in_lines[:3]:  # Show first 3
                if 0 <= line_num - 1 < len(lines):
                    trace_parts.append(f"  Line {line_num}: {lines[line_num - 1].strip()}")
        else:
            trace_parts.append(f"\n**WARNING**: '{signal_name}' is declared but never assigned!")

        # What it depends on
        if signal_info.depends_on:
            trace_parts.append(f"\n**Depends on signals**: {', '.join(signal_info.depends_on)}")

            # Recursive trace (one level) for key dependencies
            key_deps = signal_info.depends_on[:3]  # Limit to avoid overwhelming output
            for dep in key_deps:
                if dep in self.signals and self.signals[dep].signal_type not in ['input', 'parameter']:
                    dep_info = self.signals[dep]
                    if dep_info.assigned_in_lines:
                        trace_parts.append(f"  -> '{dep}' assigned at lines: {', '.join(map(str, dep_info.assigned_in_lines[:2]))}")

        # Control flow context
        if signal_info.in_always_block:
            trace_parts.append(f"\n**Control context**: Inside always block")
            if signal_info.always_block_sensitivity:
                trace_parts.append(f"  Sensitivity: {signal_info.always_block_sensitivity}")

        return "\n".join(trace_parts)

    def identify_potential_issues(self, rtl_code: str, failing_signal: Optional[str] = None) -> List[str]:
        """
        Identify potential issues in RTL code.

        General checks applicable to any RTL design:
        - Signals declared but never assigned
        - Signals assigned but never used
        - Missing reset logic in sequential blocks
        - Combinational logic in sequential blocks
        - Potential latch inference
        """
        issues = []
        lines = rtl_code.split('\n')

        # Check 1: Outputs never assigned
        for sig_name, sig_info in self.signals.items():
            if sig_info.signal_type == 'output' and not sig_info.assigned_in_lines:
                issues.append(f"CRITICAL: Output '{sig_name}' is never assigned (will read as 'x' or '0')")

        # Check 2: Registers not initialized/reset
        for sig_name, sig_info in self.signals.items():
            if sig_info.signal_type in ['reg', 'logic'] and sig_info.in_always_block:
                # Check if reset logic exists
                has_reset = self._check_reset_logic(sig_name, lines, sig_info.assigned_in_lines)
                if not has_reset and sig_info.always_block_sensitivity and 'posedge' in sig_info.always_block_sensitivity:
                    issues.append(f"WARNING: Register '{sig_name}' in sequential logic but no reset detected")

        # Check 3: State machine issues
        if self.state_machine_detected and self.state_signal:
            state_issues = self._check_state_machine_issues(lines)
            issues.extend(state_issues)

        # Check 4: Incomplete case/if statements (latch inference)
        latch_issues = self._check_incomplete_assignments(lines)
        issues.extend(latch_issues)

        # Check 5: If focusing on a specific failing signal, analyze it deeply
        if failing_signal and failing_signal in self.signals:
            specific_issues = self._analyze_failing_signal(failing_signal, lines)
            issues.extend(specific_issues)

        return issues

    def _parse_signal_declarations(self, lines: List[str]):
        """Parse input, output, reg, wire, logic declarations."""
        signal_patterns = [
            (r'\b(input|output|inout)\s+(?:(logic|reg|wire)\s+)?(?:\[([^\]]+)\]\s+)?(\w+)', 'port'),
            (r'\b(reg|wire|logic)\s+(?:\[([^\]]+)\]\s+)?(\w+)', 'internal')
        ]

        for line_num, line in enumerate(lines, 1):
            # Remove comments
            line_clean = re.sub(r'//.*$', '', line)
            line_clean = re.sub(r'/\*.*?\*/', '', line_clean)

            for pattern, decl_type in signal_patterns:
                matches = re.finditer(pattern, line_clean)
                for match in matches:
                    if decl_type == 'port':
                        port_dir = match.group(1)  # input/output/inout
                        sig_type = match.group(2) or 'wire'  # logic/reg/wire
                        bit_width = match.group(3)  # [7:0]
                        sig_name = match.group(4)

                        self.signals[sig_name] = SignalInfo(
                            name=sig_name,
                            signal_type=port_dir,
                            bit_width=f"[{bit_width}]" if bit_width else None
                        )
                    else:  # internal
                        sig_type = match.group(1)
                        bit_width = match.group(2)
                        sig_name = match.group(3)

                        if sig_name not in self.signals:  # Don't overwrite port declarations
                            self.signals[sig_name] = SignalInfo(
                                name=sig_name,
                                signal_type=sig_type,
                                bit_width=f"[{bit_width}]" if bit_width else None
                            )

    def _identify_always_blocks(self, lines: List[str]):
        """Identify always blocks and their characteristics."""
        in_always = False
        always_start = 0
        sensitivity = ""
        brace_count = 0
        signals_assigned = []
        signals_read = []

        for line_num, line in enumerate(lines, 1):
            line_clean = re.sub(r'//.*$', '', line).strip()

            # Detect always block start
            if re.search(r'\balways\s*@', line_clean) or re.search(r'\balways_comb\b', line_clean) or re.search(r'\balways_ff\b', line_clean):
                in_always = True
                always_start = line_num

                # Extract sensitivity list
                sens_match = re.search(r'always\s*@\s*\(([^)]+)\)', line_clean)
                if sens_match:
                    sensitivity = sens_match.group(1).strip()
                elif 'always_comb' in line_clean:
                    sensitivity = '*'
                elif 'always_ff' in line_clean:
                    sensitivity = 'posedge clk'

                signals_assigned = []
                signals_read = []
                brace_count = 0

            if in_always:
                brace_count += line_clean.count('{') - line_clean.count('}')

                # Track assigned signals
                assign_match = re.findall(r'(\w+)\s*[<]?=', line_clean)
                signals_assigned.extend(assign_match)

                # Track read signals (rough heuristic)
                read_match = re.findall(r'[=+\-*/&|^]\s*(\w+)', line_clean)
                signals_read.extend(read_match)

                # Detect block end
                if brace_count <= 0 and ('}' in line_clean or ';' in line_clean):
                    # Determine block type
                    if 'posedge' in sensitivity or 'negedge' in sensitivity:
                        block_type = 'sequential'
                    elif sensitivity == '*' or 'always_comb' in lines[always_start - 1]:
                        block_type = 'combinational'
                    else:
                        block_type = 'latch'

                    self.always_blocks.append(AlwaysBlockInfo(
                        start_line=always_start,
                        end_line=line_num,
                        sensitivity_list=sensitivity,
                        block_type=block_type,
                        signals_assigned=list(set(signals_assigned)),
                        signals_read=list(set(signals_read))
                    ))

                    # Update signal info
                    for sig in signals_assigned:
                        if sig in self.signals:
                            self.signals[sig].in_always_block = True
                            self.signals[sig].always_block_sensitivity = sensitivity

                    in_always = False

    def _trace_assignments(self, lines: List[str]):
        """Trace where each signal is assigned."""
        for line_num, line in enumerate(lines, 1):
            line_clean = re.sub(r'//.*$', '', line).strip()

            # Look for assignments
            # Blocking: signal = value
            blocking_matches = re.finditer(r'(\w+)\s*=\s*', line_clean)
            for match in blocking_matches:
                sig_name = match.group(1)
                if sig_name in self.signals:
                    self.signals[sig_name].assigned_in_lines.append(line_num)
                    if not self.signals[sig_name].assignment_type:
                        self.signals[sig_name].assignment_type = 'blocking'

            # Non-blocking: signal <= value
            nonblocking_matches = re.finditer(r'(\w+)\s*<=\s*', line_clean)
            for match in nonblocking_matches:
                sig_name = match.group(1)
                if sig_name in self.signals:
                    if line_num not in self.signals[sig_name].assigned_in_lines:
                        self.signals[sig_name].assigned_in_lines.append(line_num)
                    self.signals[sig_name].assignment_type = 'nonblocking'

            # Continuous: assign signal = value
            assign_matches = re.finditer(r'assign\s+(\w+)\s*=', line_clean)
            for match in assign_matches:
                sig_name = match.group(1)
                if sig_name in self.signals:
                    self.signals[sig_name].assigned_in_lines.append(line_num)
                    self.signals[sig_name].assignment_type = 'continuous'

    def _detect_state_machine(self, lines: List[str]):
        """Detect if there's a state machine (common pattern in RTL)."""
        # Look for state-related patterns
        state_patterns = [
            r'\b(state|current_state|next_state|cs|ns|fsm_state)\b',
            r'\bparameter\s+\w*STATE\w*\s*=',
            r'\blocalparam\s+\w*STATE\w*\s*=',
            r'\btypedef\s+enum'
        ]

        for line in lines:
            line_clean = re.sub(r'//.*$', '', line).lower()
            for pattern in state_patterns:
                if re.search(pattern, line_clean):
                    self.state_machine_detected = True

                    # Try to identify the state signal
                    state_match = re.search(r'\b(state|current_state|cs|fsm_state)\b', line_clean)
                    if state_match:
                        self.state_signal = state_match.group(1)

                    # Extract state names
                    state_name_match = re.finditer(r'\b([A-Z_]+STATE[A-Z_]*|IDLE|COMPUTE|DONE|INIT|WAIT)\b', line.upper())
                    for match in state_name_match:
                        self.states.add(match.group(1))

                    break

    def _build_dependencies(self, lines: List[str]):
        """Build dependency graph showing what each signal depends on."""
        for sig_name in self.signals:
            if not self.signals[sig_name].assigned_in_lines:
                continue

            dependencies = set()

            # Check assignment lines for dependencies
            for line_num in self.signals[sig_name].assigned_in_lines:
                if 0 <= line_num - 1 < len(lines):
                    line = lines[line_num - 1]
                    line_clean = re.sub(r'//.*$', '', line)

                    # Extract RHS of assignment
                    if '=' in line_clean:
                        rhs = line_clean.split('=', 1)[1] if '=' in line_clean else ""

                        # Find signal names on RHS
                        potential_signals = re.findall(r'\b(\w+)\b', rhs)
                        for pot_sig in potential_signals:
                            if pot_sig in self.signals and pot_sig != sig_name:
                                dependencies.add(pot_sig)

            self.signals[sig_name].depends_on = list(dependencies)

    def _check_reset_logic(self, sig_name: str, lines: List[str], assigned_lines: List[int]) -> bool:
        """Check if a signal has reset logic."""
        for line_num in assigned_lines:
            if 0 <= line_num - 1 < len(lines):
                line = lines[line_num - 1].lower()
                if 'rst' in line or 'reset' in line:
                    return True
        return False

    def _check_state_machine_issues(self, lines: List[str]) -> List[str]:
        """Check for common state machine issues."""
        issues = []

        if not self.states:
            issues.append("State machine detected but no states found (may be incomplete)")

        if self.state_signal and self.state_signal in self.signals:
            state_info = self.signals[self.state_signal]
            if not state_info.assigned_in_lines:
                issues.append(f"CRITICAL: State signal '{self.state_signal}' never assigned (FSM won't work)")

        return issues

    def _check_incomplete_assignments(self, lines: List[str]) -> List[str]:
        """Check for incomplete case/if statements that could infer latches."""
        issues = []

        # Look for case statements without default
        in_case = False
        has_default = False
        case_signal = ""

        for line in lines:
            line_clean = re.sub(r'//.*$', '', line).strip().lower()

            if 'case' in line_clean and '(' in line_clean:
                in_case = True
                has_default = False
                case_match = re.search(r'case\s*\((\w+)\)', line_clean)
                if case_match:
                    case_signal = case_match.group(1)

            if in_case:
                if 'default' in line_clean:
                    has_default = True
                if 'endcase' in line_clean:
                    if not has_default:
                        issues.append(f"WARNING: case({case_signal}) statement without 'default' (may infer latches)")
                    in_case = False

        return issues

    def _analyze_failing_signal(self, signal_name: str, lines: List[str]) -> List[str]:
        """Deep analysis of a specific failing signal."""
        issues = []
        sig_info = self.signals[signal_name]

        # Check if it's assigned at all
        if not sig_info.assigned_in_lines:
            issues.append(f"ROOT CAUSE: Signal '{signal_name}' is never assigned in the RTL")
            return issues

        # Check if dependencies exist
        if sig_info.depends_on:
            for dep in sig_info.depends_on:
                if dep in self.signals:
                    dep_info = self.signals[dep]
                    if not dep_info.assigned_in_lines and dep_info.signal_type not in ['input', 'parameter']:
                        issues.append(f"DEPENDENCY ISSUE: '{signal_name}' depends on '{dep}', but '{dep}' is never assigned")

        # Check assignment context
        if sig_info.in_always_block and sig_info.always_block_sensitivity:
            # Check if it's in a conditional that might not execute
            for line_num in sig_info.assigned_in_lines[:3]:
                if 0 <= line_num - 1 < len(lines):
                    # Look for conditionals around this line
                    context_start = max(0, line_num - 5)
                    context_end = min(len(lines), line_num + 2)
                    context = '\n'.join(lines[context_start:context_end])

                    if_count = context.lower().count('if ')
                    if if_count > 0:
                        issues.append(f"HINT: '{signal_name}' assigned inside conditional at line {line_num} - check if condition is met")

        return issues
