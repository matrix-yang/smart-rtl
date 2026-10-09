# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RTL Validator - Pre-test validation system for RTL code
Provides comprehensive validation without overfitting to specific problems
"""

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result from RTL validation."""
    passed: bool
    error_type: str = ""
    error_message: str = ""
    warnings: List[str] = None
    validation_time: float = 0.0
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.details is None:
            self.details = {}


class RTLValidator:
    """
    Comprehensive RTL validation system.

    Performs multiple validation checks before expensive testbench runs:
    1. Structural validation (syntax, completeness)
    2. Compilation validation (iverilog/other tools)
    3. Semantic validation (basic RTL principles)
    4. Completeness validation (all required constructs present)
    """

    def __init__(self, logger, timeout: int = 30):
        """
        Initialize RTL validator.

        Args:
            logger: Logger instance
            timeout: Timeout for validation operations in seconds
        """
        self.logger = logger
        self.timeout = timeout

        # Removed compilation tool checking - CVDP testbench handles compilation validation
        self.logger.debug(f"RTL Validator initialized (structural + semantic validation only)")

    # Removed _check_available_tools method - no longer needed

    def validate_rtl(self, rtl_code: str, module_name: Optional[str] = None) -> ValidationResult:
        """
        Perform comprehensive RTL validation.

        Args:
            rtl_code: RTL code to validate
            module_name: Expected top module name (optional)

        Returns:
            ValidationResult with detailed validation information
        """
        start_time = time.time()

        try:
            # Phase 1: Structural validation
            structural_result = self._validate_structure(rtl_code, module_name)
            if not structural_result.passed:
                return ValidationResult(
                    passed=False,
                    error_type="structural",
                    error_message=structural_result.error_message,
                    warnings=structural_result.warnings,
                    validation_time=time.time() - start_time,
                    details=structural_result.details
                )

            # Phase 2: Semantic validation (skip compilation - CVDP testbench will catch compilation errors)
            semantic_result = self._validate_semantics(rtl_code)

            # Combine results
            all_warnings = structural_result.warnings + semantic_result.warnings
            all_details = {**structural_result.details, **semantic_result.details}

            return ValidationResult(
                passed=True,
                error_type="",
                error_message="",
                warnings=all_warnings,
                validation_time=time.time() - start_time,
                details=all_details
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                error_type="validation_error",
                error_message=f"Validation process failed: {str(e)}",
                validation_time=time.time() - start_time
            )

    def _validate_structure(self, rtl_code: str, expected_module: Optional[str] = None) -> ValidationResult:
        """Validate RTL structural correctness."""
        warnings = []
        details = {}

        if not rtl_code or not rtl_code.strip():
            return ValidationResult(
                passed=False,
                error_message="Empty or whitespace-only RTL code"
            )

        # Check for basic RTL structure (more precise regex to avoid false positives)
        module_declarations = re.findall(r'^\s*module\s+(\w+)', rtl_code, re.IGNORECASE | re.MULTILINE)
        if not module_declarations:
            return ValidationResult(
                passed=False,
                error_message="No module declarations found"
            )

        details['modules_found'] = module_declarations

        # Check module/endmodule balance (more precise regex)
        endmodule_count = len(re.findall(r'^\s*endmodule\s*$', rtl_code, re.IGNORECASE | re.MULTILINE))
        if len(module_declarations) != endmodule_count:
            return ValidationResult(
                passed=False,
                error_message=f"Unbalanced module/endmodule: {len(module_declarations)} modules, {endmodule_count} endmodules"
            )

        # Check for expected module name if provided
        if expected_module and expected_module not in module_declarations:
            warnings.append(f"Expected module '{expected_module}' not found. Found: {module_declarations}")

        # Check for reasonable content
        code_lines = [line.strip() for line in rtl_code.split('\n') if line.strip() and not line.strip().startswith('//')]
        if len(code_lines) < 10:  # Very basic threshold
            warnings.append("RTL code appears to be very short")

        # Check for balanced delimiters
        if not self._check_balanced_delimiters(rtl_code):
            return ValidationResult(
                passed=False,
                error_message="Unbalanced parentheses, brackets, or braces"
            )

        # Check for basic RTL constructs
        required_constructs = {
            'ports': r'\b(input|output|inout)\b',
            'signals': r'\b(wire|reg|logic)\b',
            'logic': r'\b(always|assign)\b'
        }

        missing_constructs = []
        for construct_name, pattern in required_constructs.items():
            if not re.search(pattern, rtl_code, re.IGNORECASE):
                missing_constructs.append(construct_name)

        if missing_constructs:
            warnings.append(f"Missing typical RTL constructs: {missing_constructs}")

        details['code_lines'] = len(code_lines)
        details['missing_constructs'] = missing_constructs

        return ValidationResult(
            passed=True,
            warnings=warnings,
            details=details
        )

    # Removed compilation validation methods - CVDP testbench handles compilation checking

    def _validate_semantics(self, rtl_code: str) -> ValidationResult:
        """Validate RTL semantic correctness (general principles)."""
        warnings = []
        details = {}

        # Check for common RTL issues (general patterns, not problem-specific)

        # 1. Clock/reset patterns
        has_clock = bool(re.search(r'\b(clk|clock)\b', rtl_code, re.IGNORECASE))
        has_reset = bool(re.search(r'\b(rst|reset)\b', rtl_code, re.IGNORECASE))
        has_always = bool(re.search(r'\balways\b', rtl_code, re.IGNORECASE))

        if has_always and not (has_clock or has_reset):
            warnings.append("Sequential logic found but no clear clock/reset signals")

        # 2. Port direction consistency
        port_directions = re.findall(r'\b(input|output|inout)\s+(?:\w+\s+)*(\w+)', rtl_code, re.IGNORECASE)
        details['port_count'] = len(port_directions)

        if len(port_directions) == 0:
            warnings.append("No ports found - module may not be usable")

        # 3. Signal usage patterns
        wire_declarations = re.findall(r'\bwire\s+(?:\[.*?\]\s+)?(\w+)', rtl_code, re.IGNORECASE)
        reg_declarations = re.findall(r'\breg\s+(?:\[.*?\]\s+)?(\w+)', rtl_code, re.IGNORECASE)

        details['wire_count'] = len(wire_declarations)
        details['reg_count'] = len(reg_declarations)

        # 4. Always block patterns
        always_blocks = re.findall(r'always\s*@\s*\([^)]+\)', rtl_code, re.IGNORECASE)
        details['always_block_count'] = len(always_blocks)

        # 5. Basic completeness check
        has_logic_content = bool(re.search(r'\b(assign|always|generate)\b', rtl_code, re.IGNORECASE))
        if not has_logic_content:
            warnings.append("No logic content found (assign, always, or generate blocks)")

        return ValidationResult(
            passed=True,  # Semantic validation rarely fails, mostly warnings
            warnings=warnings,
            details=details
        )

    def _check_balanced_delimiters(self, code: str) -> bool:
        """Check if parentheses, brackets, and braces are balanced."""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}

        # Simple state machine to skip string literals
        in_string = False
        in_comment = False
        escape_next = False
        i = 0

        while i < len(code):
            char = code[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == '\\':
                escape_next = True
                i += 1
                continue

            # Handle single-line comments
            if not in_string and i < len(code) - 1 and code[i:i+2] == '//':
                # Skip to end of line
                while i < len(code) and code[i] != '\n':
                    i += 1
                continue

            # Handle multi-line comments
            if not in_string and i < len(code) - 1 and code[i:i+2] == '/*':
                in_comment = True
                i += 2
                continue

            if in_comment and i < len(code) - 1 and code[i:i+2] == '*/':
                in_comment = False
                i += 2
                continue

            if in_comment:
                i += 1
                continue

            # Handle strings
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif in_string:
                i += 1
                continue

            # Check delimiters
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                last_open = stack.pop()
                if pairs[last_open] != char:
                    return False

            i += 1

        return len(stack) == 0 and not in_string and not in_comment
