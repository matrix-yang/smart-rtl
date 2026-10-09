# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLM-Centric RTL Debugger - Focused, iterative debugging with clear guidance
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import re
from .focused_debugger import FocusedDebugger
from .testbench_analyzer import TestbenchAnalyzer
from .utils.llm_client import LLMClient
from .utils.ace_model_client import AceRTLClient


class LLMCentricDebugger:
    """
    Focused LLM-centric RTL debugger.

    Philosophy:
    - Provide minimal but critical information
    - Focus on ONE specific error at a time
    - Track what was tried and what worked
    - Give clear, actionable guidance
    """

    def __init__(self, model_name: str, logger, api_key: Optional[str] = None,
                 max_retries: int = 3, initial_backoff: float = 5.0,
                 llm_context_limit: int = 300000, ace_rtl_stages: set = None,
                 ace_rtl_temperature: float = 0.2):
        self.model_name = model_name
        self.logger = logger
        self.max_retries = max_retries
        self.ace_rtl_stages = ace_rtl_stages if ace_rtl_stages is not None else set()
        self.ace_rtl_temperature = ace_rtl_temperature

        # Core components - streamlined
        self.focused_debugger = FocusedDebugger(logger, model_name,
                                                ace_rtl_stages=self.ace_rtl_stages,
                                                ace_rtl_temperature=self.ace_rtl_temperature,
                                                api_key=api_key,
                                                max_retries=max_retries,
                                                initial_backoff=initial_backoff)
        self.testbench_analyzer = TestbenchAnalyzer(logger)

        # Backward compatibility - keep error_analyzer for agent.py
        from .deep_error_analyzer import DeepErrorAnalyzer
        self.error_analyzer = DeepErrorAnalyzer(logger, model_name, llm_context_limit=llm_context_limit)

        # LLM client for stage 2 (code correction)
        # Use ACE-RTL if stage 2 is in ace_rtl_stages, otherwise use LLM
        if 2 in self.ace_rtl_stages:
            self.logger.info("Stage 2 (code correction): Using ACE-RTL API client")
            self.client = AceRTLClient(
                logger=logger,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                temperature=self.ace_rtl_temperature
            )
        else:
            self.logger.debug("Stage 2 (code correction): Using LLM model")
            self.client = LLMClient(
                model=model_name,
                api_key=api_key,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                logger=logger
            )

        self.logger.debug("Initialized Focused RTL Debugger")

    def clear_history_for_restart(self):
        """Clear debugging history - called when doing a fresh start."""
        self.focused_debugger.clear_history()
        self.logger.info("Cleared debugging history for fresh start")

    def _is_cvdp_coverage_context(
        self, specification: str, current_code: str, testbench_code: Optional[str]
    ) -> bool:
        """Return True for coverage testbench generation."""
        text = f"{specification}\n{current_code}\n{testbench_code or ''}"
        low = text.lower()
        return (
            "# === cvdp harness context ===" in low
            and re.search(r"\btarget\s*=", text, re.IGNORECASE) is not None
            and ("coverage.cmd" in low or "test_coverage" in low or "test_report" in low)
            and re.search(r"\bnum_bugs\s*=", text, re.IGNORECASE) is None
            and "bug_" not in low
        )

    def debug_rtl_with_llm_reasoning(self, specification: str, current_code: str,
                                   error_report: str, iteration: int,
                                   harness_dir: Optional[Any] = None,
                                   is_restart: bool = False,
                                   multi_file_info: dict = None,
                                   testbench_context: Optional[str] = None,
                                   per_signal_mismatches: Optional[list] = None) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        Focused debugging method with minimal but critical information.

        Returns:
            Tuple of (corrected_code, full_prompt, llm_response, metadata)

        Process:
        1. Extract test status before this iteration
        2. Get testbench code if available
        3. Create focused prompt (code + error + focus + history + spec)
        4. Get LLM response with guidance
        5. Extract corrected code
        6. Extract guidance for recording
        7. Return for testing
        """
        print(f"   Using Focused Debugger (streamlined prompts with clear guidance)")

        # Step 3: Load context files for multi-file datapoints
        context_code = None
        target_guidance = ""
        if multi_file_info and multi_file_info.get('context_files'):
            try:
                context_parts = []
                for context_file in multi_file_info['context_files']:
                    context_path = multi_file_info['rtl_dir'] / context_file
                    with open(context_path, 'r') as f:
                        content = f.read()
                    context_parts.append(f"// ========== FILE: {context_file} (helper module - do not modify) ==========\n{content}")

                context_code = '\n\n'.join(context_parts)
                print(f"   Loaded {len(multi_file_info['context_files'])} context files (helpers, won't be modified)")
            except Exception as e:
                self.logger.warning(f"Could not load context files: {e}")

        if multi_file_info:
            target_files = multi_file_info.get('target_files') or []
            primary_target = multi_file_info.get('buggy_file')
            additional_targets = multi_file_info.get('additional_target_files') or []
            if primary_target or target_files:
                target_guidance = "\n\nCVDP GENERATED TARGET FILES:\n"
                if primary_target:
                    target_guidance += f"- Current primary target file under repair: `{primary_target}`.\n"
                if target_files:
                    target_guidance += "- Declared generated RTL target file(s): " + ", ".join(
                        f"`{name}`" for name in target_files
                    ) + ".\n"
                if additional_targets:
                    target_guidance += (
                        "- The other declared target file(s) are also generated deliverables, not immutable helpers. "
                        "For cross-file/interface fixes, return complete replacements for all declared target files so wrappers, cores, parameters, and instance APIs stay consistent.\n"
                    )
                if len(target_files) > 1:
                    target_guidance += (
                        "- For multi-file generated RTL, emit one file section per declared target using this exact header before each file: "
                        "`// TARGET_FILE: <target filename or rtl/path.sv>`. "
                        "Include unchanged target files too if they are needed to preserve cross-file compatibility.\n"
                    )
                target_guidance += (
                    "- Do not emit replacement modules for helper/context RTL files that are not listed as generated target files; instantiate or reference them as immutable dependencies.\n"
                )

        # Step 1: Get testbench code if available (including dependencies)
        testbench_code = None
        if harness_dir:
            try:
                harness_path = Path(harness_dir)

                def read_context_file(path: Path, max_chars: int = 50000) -> str:
                    content = path.read_text(encoding='utf-8', errors='replace')
                    if len(content) <= max_chars:
                        return content
                    keep = max_chars - len("\n... [FILE TRUNCATED] ...\n")
                    head = int(keep * 0.65)
                    tail = keep - head
                    return content[:head] + "\n... [FILE TRUNCATED] ...\n" + content[-tail:]

                def add_context_file(parts: list, path: Path, label: str) -> None:
                    try:
                        rel = path.relative_to(harness_path)
                    except ValueError:
                        rel = path
                    parts.append(f"# File: {rel} ({label})")
                    parts.append(read_context_file(path))
                    parts.append("")

                src_dir = harness_path / "src"
                harness_parts = ["# === CVDP HARNESS CONTEXT ==="]
                included = 0

                if src_dir.exists():
                    for pattern, label in [
                        ('.env', 'CVDP env/scoring knobs'),
                        ('*.cmd', 'tool command file'),
                        ('*.py', 'CVDP pytest runner/checker'),
                        ('*.sv', 'immutable source/UUT HDL'),
                        ('*.v', 'immutable source/UUT HDL'),
                        ('*.svh', 'source include'),
                        ('*.vh', 'source include'),
                    ]:
                        for path in sorted(src_dir.glob(pattern)):
                            if path.name in ['test_runner.py', '__init__.py'] or path.is_dir():
                                continue
                            add_context_file(harness_parts, path, label)
                            included += 1

                # Some CVDP items keep immutable helper RTL under rtl/. Include
                # it only when it is not the same generated file already shown
                # as the code under repair.
                rtl_dir = harness_path / "rtl"
                if rtl_dir.exists():
                    current_norm = ''.join(current_code.split())
                    for pattern in ('*.sv', '*.v', '*.svh', '*.vh'):
                        for path in sorted(rtl_dir.glob(pattern)):
                            content = read_context_file(path)
                            if current_norm and ''.join(content.split()) == current_norm:
                                continue
                            try:
                                rel = path.relative_to(harness_path)
                            except ValueError:
                                rel = path
                            harness_parts.append(f"# File: {rel} (immutable/helper RTL context)")
                            harness_parts.append(content)
                            harness_parts.append("")
                            included += 1

                if included:
                    testbench_code = "\n".join(harness_parts)
                    print(f"   CVDP harness context loaded: {included} files ({len(testbench_code)} chars)")
                else:
                    print(f"   No CVDP harness context files found in {harness_path}")

                # For coverage debugging, the latest coverage
                # summary is often more useful than another long simulator log.
                if testbench_code and self._is_cvdp_coverage_context(specification, current_code, testbench_code):
                    rundir = harness_path / "rundir"
                    coverage_parts = []
                    for name in ("coverage.log", "imc.log"):
                        path = rundir / name
                        if path.exists() and path.is_file():
                            content = path.read_text(encoding='utf-8', errors='replace')
                            if len(content) > 20000:
                                content = content[:8000] + "\n... [RUN ARTIFACT TRUNCATED] ...\n" + content[-8000:]
                            coverage_parts.append(f"# File: rundir/{name} (latest coverage/run artifact)")
                            coverage_parts.append(content)
                    if coverage_parts:
                        testbench_code += "\n\n# === CVDP LATEST COVERAGE ARTIFACTS ===\n" + "\n".join(coverage_parts)
                        print(f"   Added latest coverage artifacts ({sum(len(p) for p in coverage_parts)} chars)")
            except Exception as e:
                self.logger.warning(f"Could not load harness context: {e}")
                print(f"   Harness context loading failed: {e}")

        # Use sanitized testbench context for VerilogEval/ChipBench when harness_dir is not available
        if testbench_code is None and testbench_context:
            testbench_code = testbench_context
            print(f"   Using sanitized testbench context ({len(testbench_code)} chars)")

        # Step 2: Create focused prompt (LLM Call 1 happens inside this)
        # Returns: prompt for LLM Call 2, focused_error, and fix_guidance from LLM Call 1
        prompt, focused_error, fix_guidance = self.focused_debugger.create_focused_prompt(
            buggy_code=current_code,
            error_report=error_report,
            iteration=iteration,
            specification=specification + target_guidance,
            testbench_code=testbench_code,
            context_code=context_code,
            per_signal_mismatches=per_signal_mismatches
        )

        # Step 3: LLM Call 2 - Implementation (take guidance and implement it)
        try:
            # Use ACE-RTL temperature if stage 2 uses ACE-RTL, otherwise use default
            temp = self.ace_rtl_temperature if 2 in self.ace_rtl_stages else 0.2
            response = self.client.call_llm(prompt, temperature=temp)
            if not response:
                raise ValueError("LLM returned empty response")
        except Exception as e:
            print(f"   ERROR: LLM call failed: {e}")
            raise

        # Step 4: Extract corrected code
        try:
            corrected_code = self._extract_rtl_code(response)

            # For multi-file datapoints, ensure filename is at the top for CVDP
            if multi_file_info and multi_file_info.get('buggy_file'):
                buggy_filename = multi_file_info['buggy_file']
                target_files = multi_file_info.get('target_files') or []
                has_any_marker = bool(re.search(r"(?im)^[ \t]*//[ \t]*TARGET_FILE[ \t]*:", corrected_code))
                # For single-target jobs, a marker helps the CVDP injector.
                # For multi-target jobs, do not add a default single marker:
                # that would collapse a multi-module payload into the primary
                # file. The prompt already asks the LLM to mark each file.
                if len(target_files) <= 1 and not has_any_marker and f"// TARGET_FILE: {buggy_filename}" not in corrected_code[:200]:
                    corrected_code = f"// TARGET_FILE: {buggy_filename}\n{corrected_code}"
                    self.logger.info(f"Added TARGET_FILE marker: {buggy_filename}")

        except Exception as e:
            print(f"   ERROR: RTL code extraction failed: {e}")
            raise ValueError(f"RTL extraction failed: {e}")

        # Metadata includes guidance from LLM Call 1
        metadata = {
            'approach': 'focused',
            'iteration': iteration,
            'guidance': fix_guidance,  # From LLM Call 1
            'focused_error': focused_error,
            'error_reasoning': fix_guidance,  # Store guidance as reasoning for now
            'has_testbench': testbench_code is not None
        }

        return corrected_code, prompt, response, metadata

    def record_iteration_result(self, iteration: int, error_report: str,
                               metadata: Dict, test_passed: bool,
                               code_before: str = "", code_after: str = "",
                               mismatch_count: int = -1):
        """
        Record the result of this iteration for history tracking.

        Args:
            iteration: Iteration number
            error_report: Error report after running test
            metadata: Metadata from debug call (contains guidance, focused_error, etc.)
            test_passed: Whether all tests passed
            code_before: Code before the fix
            code_after: Code after the fix
        """
        # Extract test results
        num_passing, num_total = self.focused_debugger.parse_test_results(error_report)
        test_results_str = f"{num_passing}/{num_total} tests passed"

        # Determine outcome using mismatch count (more reliable than parsed test results)
        if test_passed:
            outcome = "solved"
        elif iteration == 1:
            outcome = "no_change"  # First iteration baseline
        elif mismatch_count >= 0 and self.focused_debugger.fix_history:
            prev_attempt = self.focused_debugger.fix_history[-1]
            if prev_attempt.mismatch_count >= 0:
                if mismatch_count < prev_attempt.mismatch_count:
                    outcome = "improved"
                elif mismatch_count > prev_attempt.mismatch_count:
                    outcome = "regressed"
                else:
                    outcome = "no_change"
            else:
                outcome = "no_change"
        else:
            # Fallback to parsed test results
            if self.focused_debugger.fix_history:
                prev_attempt = self.focused_debugger.fix_history[-1]
                try:
                    prev_passing = int(prev_attempt.test_results.split('/')[0])
                    if num_passing > prev_passing:
                        outcome = "improved"
                    elif num_passing < prev_passing:
                        outcome = "regressed"
                    else:
                        outcome = "no_change"
                except (ValueError, IndexError):
                    outcome = "no_change"
            else:
                outcome = "no_change"

        # LLM Call 3: Extract implementation changes AND analyze fix effectiveness
        guidance = metadata.get('guidance', 'No guidance extracted')
        focused_error = metadata.get('focused_error', 'Unknown error')

        if code_before and code_after:
            implementation_changes, fix_effectiveness = self.focused_debugger.extract_implementation_changes(
                code_before=code_before,
                code_after=code_after,
                guidance=guidance,
                focused_error=focused_error,
                new_error_report=error_report
            )
        else:
            implementation_changes = "Code changes not tracked"
            fix_effectiveness = "Effectiveness not analyzed"

        # Record the attempt
        self.focused_debugger.record_attempt(
            iteration=iteration,
            focused_error=focused_error,
            error_reasoning=metadata.get('error_reasoning', ''),
            guidance=guidance,
            implementation_changes=implementation_changes,
            fix_effectiveness=fix_effectiveness,
            test_results=test_results_str,
            outcome=outcome,
            test_passed=test_passed,
            mismatch_count=mismatch_count
        )

    def _extract_rtl_code(self, response: str) -> str:
        """Extract RTL code from LLM response using simple token-based approach."""
        import re
        from typing import Optional

        self.logger.debug("Starting token-based RTL extraction from LLM response")

        # Method 1: Extract using <code>...</code> tokens (primary method)
        extracted_code = self._extract_with_code_tokens(response)
        if extracted_code:
            self.logger.debug("Successfully extracted RTL using <code> tokens")
            return extracted_code

        # Method 2: Fallback to markdown code blocks
        extracted_code = self._extract_from_markdown_blocks(response)
        if extracted_code:
            self.logger.debug("Successfully extracted RTL from markdown code blocks")
            return extracted_code

        # Method 3: Check if entire response is valid RTL (last resort)
        if self._is_valid_rtl_structure(response.strip()):
            self.logger.debug("Entire response appears to be valid RTL")
            return response.strip()

        # If all methods fail, this indicates LLM didn't follow instructions
        self.logger.error("Failed to extract RTL code - LLM may not have followed format instructions")
        self.logger.error(f"Response length: {len(response)} chars, has <code>: {'<code>' in response}, has ```: {'```' in response}")
        raise ValueError("LLM response does not contain properly formatted RTL code. Please retry the LLM call.")

    def _extract_with_code_tokens(self, response: str) -> Optional[str]:
        """Extract RTL code using <code>...</code> tokens."""
        import re

        # Look for <code>...</code> tokens (case insensitive)
        pattern = r'<code>\s*(.*?)\s*</code>'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            code = match.strip()

            if self._is_valid_rtl_structure(code):
                return code

        # The configured reasoning model sometimes emits the opening token and omits the closing tag.
        # Recover that candidate, but trim after the last module before validation
        # so explanatory prose is not injected into the generated file.
        orphan_match = re.search(r'<code\b[^>]*>\s*(.*)', response, re.DOTALL | re.IGNORECASE)
        if orphan_match:
            code = orphan_match.group(1).strip()
            endmodule_matches = list(re.finditer(r'\bendmodule\b', code, re.IGNORECASE))
            if endmodule_matches:
                code = code[:endmodule_matches[-1].end()].strip()
            if self._is_valid_rtl_structure(code):
                self.logger.info("Recovered RTL from unterminated <code> block")
                return code

        return None

    def _extract_from_markdown_blocks(self, response: str) -> Optional[str]:
        """Fallback: Extract RTL from markdown code blocks."""
        import re

        # Try common RTL language tags
        language_tags = ['systemverilog', 'verilog', 'sv', 'v']

        for tag in language_tags:
            pattern = rf'```{tag}\s*\n(.*?)\n```'
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                code = match.strip()
                if self._is_valid_rtl_structure(code):
                    return code

        # Try generic code blocks that contain modules
        pattern = r'```\s*\n(.*?)\n```'
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            code = match.strip()
            if self._is_valid_rtl_structure(code):
                return code

        return None

    def _is_valid_rtl_structure(self, code: str) -> bool:
        """Comprehensive validation to check if extracted code has valid RTL structure."""
        import re

        if not code or not code.strip():
            self.logger.debug("Validation failed: empty code")
            return False

        code_clean = code.strip()

        # Remove block comments /* ... */ first
        code_no_block_comments = re.sub(r'/\*.*?\*/', '', code_clean, flags=re.DOTALL)

        # Remove line comments // ... and empty lines
        lines_clean = []
        for line in code_no_block_comments.split('\n'):
            line_no_comment = re.sub(r'//.*', '', line).strip()
            if line_no_comment:
                lines_clean.append(line_no_comment)

        if not lines_clean:
            self.logger.debug("Validation failed: no content after removing comments")
            return False

        code_no_comments = '\n'.join(lines_clean)

        # Check 1: Must contain at least one module declaration
        module_declarations = re.findall(r'\bmodule\s+(\w+)', code_no_comments, re.IGNORECASE)
        if not module_declarations:
            self.logger.debug("Validation failed: no module declarations found")
            return False

        # Check 2: Each module must have corresponding endmodule
        endmodule_count = len(re.findall(r'\bendmodule\b', code_no_comments, re.IGNORECASE))
        if len(module_declarations) != endmodule_count:
            # Warning but don't reject - let compiler catch syntax errors
            self.logger.warning(f"Module/endmodule mismatch ({len(module_declarations)} modules, {endmodule_count} endmodules) - accepting anyway")
            # return False  # Comment out - be lenient

        # Check 3: Must have reasonable content (not just module/endmodule)
        if len(code_no_comments) < 50:  # Minimum reasonable module size
            self.logger.debug("Validation failed: code too short")
            return False

        # Check 4: Should contain typical RTL constructs
        rtl_constructs = [
            r'\b(input|output|inout)\b',  # Port declarations
            r'\b(wire|reg|logic)\b',      # Signal declarations
            r'\b(always|assign)\b',       # RTL logic
            r'\b(initial|task|function)\b', # Testbench/verification logic
            r'\b(parameter|localparam)\b' # Parameters
        ]

        construct_found = any(re.search(pattern, code_no_comments, re.IGNORECASE)
                            for pattern in rtl_constructs)

        if not construct_found:
            self.logger.debug("Validation failed: no RTL constructs found")
            return False

        # Check 5: Balanced parentheses and brackets (basic syntax check)
        if not self._check_balanced_delimiters(code_no_comments):
            self.logger.debug("Validation failed: unbalanced delimiters")
            return False

        self.logger.debug(f"Validation passed: {len(module_declarations)} module(s) found")
        return True

    def _check_balanced_delimiters(self, code: str) -> bool:
        """Check if parentheses, brackets, and braces are balanced."""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}

        # Simple state machine to skip string literals
        in_string = False
        escape_next = False

        for char in code:
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not in_string:
                in_string = True
                continue
            elif char == '"' and in_string:
                in_string = False
                continue

            if in_string:
                continue

            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                last_open = stack.pop()
                if pairs[last_open] != char:
                    return False

        return len(stack) == 0

    def _add_format_instructions(self, prompt: str) -> str:
        """Add explicit format instructions to prompt for retry attempts."""
        format_instructions = """

CRITICAL FORMAT REQUIREMENT:
You MUST wrap your final SystemVerilog code with <code> and </code> tags exactly like this:

<code>
module YourModuleName (
    // your SystemVerilog code here
);
    // module implementation
endmodule
</code>

This format is REQUIRED for proper code extraction. Do not use markdown code blocks or any other format.
"""
        return prompt + format_instructions
