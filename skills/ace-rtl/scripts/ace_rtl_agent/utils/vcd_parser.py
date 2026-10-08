# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple VCD (Value Change Dump) Parser for extracting signal values at specific times.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class VCDParser:
    """
    Lightweight VCD parser to extract signal values at specific time points.
    Focused on debugging assistance rather than full VCD functionality.
    """

    def __init__(self, vcd_file: Path):
        """
        Initialize VCD parser with a VCD file.

        Args:
            vcd_file: Path to VCD file
        """
        self.vcd_file = vcd_file
        self.signals = {}  # identifier -> signal_info
        self.time_values = {}  # time -> {identifier -> value}
        self.parsed = False

    def parse(self):
        """Parse the VCD file and extract signal definitions and values."""
        if not self.vcd_file.exists():
            raise FileNotFoundError(f"VCD file not found: {self.vcd_file}")

        with open(self.vcd_file, 'r') as f:
            lines = f.readlines()

        # Parse in two phases: declarations and values
        in_declaration = True
        current_time = 0

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # End of declaration section
            if line.startswith('$enddefinitions'):
                in_declaration = False
                i += 1
                continue

            if in_declaration:
                # Parse signal declarations: $var wire 1 ! clk $end
                if line.startswith('$var'):
                    self._parse_var_declaration(line)
            else:
                # Parse value changes
                if line.startswith('#'):
                    # Time stamp
                    current_time = int(line[1:])
                    if current_time not in self.time_values:
                        self.time_values[current_time] = {}
                elif line and not line.startswith('$'):
                    # Value change
                    self._parse_value_change(line, current_time)

            i += 1

        self.parsed = True

    def _parse_var_declaration(self, line: str):
        """
        Parse a $var declaration line.
        Example: $var wire 1 ! clk $end
        Format: $var <type> <width> <identifier> <reference> $end
        """
        parts = line.split()
        if len(parts) >= 5 and parts[0] == '$var':
            var_type = parts[1]
            width = int(parts[2])
            identifier = parts[3]
            reference = parts[4]

            self.signals[identifier] = {
                'name': reference,
                'type': var_type,
                'width': width
            }

    def _parse_value_change(self, line: str, current_time: int):
        """
        Parse a value change line.
        Examples:
          - "0!" (single bit value 0 for identifier !)
          - "b10101010 #" (multi-bit value for identifier #)
        """
        if line[0] in ['0', '1', 'x', 'z', 'X', 'Z']:
            # Single bit value
            value = line[0]
            identifier = line[1:]
            if identifier:
                self.time_values[current_time][identifier] = value
        elif line[0] == 'b':
            # Multi-bit value: b10101010 #
            parts = line.split()
            if len(parts) >= 2:
                value = parts[0][1:]  # Remove 'b' prefix
                identifier = parts[1]
                self.time_values[current_time][identifier] = value

    def get_signal_value(self, signal_name: str, time: int) -> Optional[str]:
        """
        Get the value of a signal at a specific time.

        Args:
            signal_name: Name of the signal (e.g., 'clk', 'q', 'd')
            time: Time point

        Returns:
            Signal value as string, or None if not found
        """
        if not self.parsed:
            self.parse()

        # Find identifier for this signal name
        identifier = None
        for ident, info in self.signals.items():
            if info['name'] == signal_name:
                identifier = ident
                break

        if not identifier:
            return None

        # Find the most recent value at or before the requested time
        relevant_times = sorted([t for t in self.time_values.keys() if t <= time], reverse=True)

        for t in relevant_times:
            if identifier in self.time_values[t]:
                return self.time_values[t][identifier]

        return None

    def get_all_signals_at_time(self, time: int, module_prefix: str = None) -> Dict[str, str]:
        """
        Get values of all signals at a specific time.

        Args:
            time: Time point
            module_prefix: Module prefix to filter signals (optional, None = all signals)

        Returns:
            Dictionary of signal_name -> value
        """
        if not self.parsed:
            self.parse()

        result = {}

        # Get all values up to this time
        for identifier, info in self.signals.items():
            signal_name = info['name']

            # Filter by module if specified
            if module_prefix and not signal_name.startswith(module_prefix):
                continue

            # Get the most recent value
            # Try both with and without module hierarchy
            value = None
            relevant_times = sorted([t for t in self.time_values.keys() if t <= time], reverse=True)

            for t in relevant_times:
                if identifier in self.time_values[t]:
                    value = self.time_values[t][identifier]
                    break

            if value is not None:
                # Use short name (without module prefix)
                short_name = signal_name.split('.')[-1] if '.' in signal_name else signal_name
                result[short_name] = value

        return result

    def get_signal_trace(self, signal_name: str, start_time: int, end_time: int) -> List[Tuple[int, str]]:
        """
        Get all value changes for a signal in a time range.

        Args:
            signal_name: Name of the signal
            start_time: Start time
            end_time: End time

        Returns:
            List of (time, value) tuples
        """
        if not self.parsed:
            self.parse()

        # Find identifier for this signal
        identifier = None
        for ident, info in self.signals.items():
            if info['name'].endswith(signal_name):
                identifier = ident
                break

        if not identifier:
            return []

        trace = []
        for time in sorted(self.time_values.keys()):
            if start_time <= time <= end_time:
                if identifier in self.time_values[time]:
                    value = self.time_values[time][identifier]
                    trace.append((time, value))

        return trace

    def get_signals_around_time(self, time: int, before: int = 10, after: int = 10) -> Dict[int, Dict[str, str]]:
        """
        Get signal values in a time window around a specific time.
        Useful for showing context around a mismatch.

        Args:
            time: Center time point
            before: Time units before (default: 10)
            after: Time units after (default: 10)

        Returns:
            Dictionary of time -> {signal_name -> value}
        """
        if not self.parsed:
            self.parse()

        result = {}
        start_time = max(0, time - before)
        end_time = time + after

        # Get all relevant time points in range
        relevant_times = sorted([t for t in self.time_values.keys()
                                if start_time <= t <= end_time])

        for t in relevant_times:
            result[t] = self.get_all_signals_at_time(t)

        return result

    def format_waveform_text(self, signal_names: List[str], start_time: int, end_time: int,
                            mark_time: Optional[int] = None) -> str:
        """
        Create a text-based waveform display for specified signals.

        Args:
            signal_names: List of signal names to display
            start_time: Start time
            end_time: End time
            mark_time: Optional time to mark (e.g., mismatch time)

        Returns:
            Formatted text waveform
        """
        if not self.parsed:
            self.parse()

        lines = []
        lines.append("=" * 80)
        lines.append("WAVEFORM EXCERPT")
        lines.append("=" * 80)

        # Get all time points in range
        relevant_times = sorted([t for t in self.time_values.keys()
                                if start_time <= t <= end_time])

        if not relevant_times:
            return "No waveform data available in specified time range"

        # For each signal, show its values
        for signal_name in signal_names:
            lines.append(f"\nSignal: {signal_name}")
            lines.append("-" * 40)

            trace = []
            for t in relevant_times:
                value = self.get_signal_value(signal_name, t)
                if value is not None:
                    marker = " <-- MISMATCH" if mark_time and t == mark_time else ""
                    trace.append(f"  t={t:6d}ps: {value:>16s}{marker}")

            if trace:
                lines.extend(trace[:20])  # Limit to 20 entries per signal
                if len(trace) > 20:
                    lines.append(f"  ... ({len(trace) - 20} more entries)")
            else:
                lines.append("  (no values in range)")

        lines.append("=" * 80)
        return "\n".join(lines)
