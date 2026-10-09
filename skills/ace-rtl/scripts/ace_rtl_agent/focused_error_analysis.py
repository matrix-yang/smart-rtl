# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused debugger implementation mixins."""

import os
import re
from typing import Dict, List, Optional, Tuple

from .focused_debugger_types import FixAttempt


class FocusedErrorAnalysisMixin:
    def _is_compilation_error(self, focused_error: str, error_report: str) -> bool:
        """
        Detect if the error is a compilation error (syntax/structure) vs runtime test failure.

        Compilation error indicators:
        - iverilog/verilator/vcs compilation failure
        - Exit status/return code errors
        - Syntax error messages
        - Missing module definitions
        - Port mismatch errors

        Returns:
            True if this is a compilation error
        """
        error_text = (focused_error + " " + error_report).lower()

        compilation_indicators = [
            'compilation failed',
            'compile error',
            'compilation error',
            'iverilog',
            'exit status',
            'return code',
            'syntax error',
            'parse error',
            'module not found',
            'module definition',
            'port mismatch',
            'undeclared',
            'subprocess.callederror',
            'returned non-zero',
            'compilation unsuccessful'
        ]

        return any(indicator in error_text for indicator in compilation_indicators)

    def _extract_focused_error_only(self, error_report: str) -> str:
        """
        Use LLM to extract and summarize the most important error from error report.

        Returns the focused error message with key details.
        """
        prompt_error_report = self._prepare_error_report_for_prompt(error_report)
        extraction_prompt = f"""Extract the MOST IMPORTANT error from this test error report.

ERROR REPORT:
```
{prompt_error_report}
```

YOUR TASK:
Identify the primary/first error that should be fixed. Provide a brief, focused summary.

The error may be:
- An assertion with expected/actual values (e.g., "Expected 5, got 3")
- A descriptive error message (e.g., "WB DATA MISMATCH", "Fee not marked as ready")
- A test failure description

Extract whatever information is available in the error report:

**Required**:
- **Error Message**: <clear description of what failed>

**Optional** (only include if present in the error report):
- **Expected vs Actual**: Expected `X`, got `Y` (if values are in the error)
- **Clock Cycle**: At cycle N (if cycle info present - more useful than ns timing)
- **Signals Involved**: <signal names> (if signals are mentioned)
- **Test Case**: <which test failed> (if multiple tests)

Be flexible - not all errors have all fields. Just extract what's actually in the report.
Keep it concise and factual.

EXAMPLE OUTPUT 1 (with expected/actual):
**Error Message**: Output mismatch for load instruction
**Expected vs Actual**: Expected `0x1234`, got `0x0000`
**Clock Cycle**: At cycle 15
**Signals Involved**: wb_data_o

EXAMPLE OUTPUT 2 (descriptive error only):
**Error Message**: WB DATA MISMATCH
**Test Case**: test_load_store[basic_load]

OUTPUT: Provide the focused error extraction in the format above."""

        try:
            # Use ACE-RTL temperature if stage 1 uses ACE-RTL, otherwise use 0.1
            temp = self.ace_rtl_temperature if 1 in self.ace_rtl_stages else 0.1
            focused_error = self.llm_client.call_llm(extraction_prompt, temperature=temp)
            if focused_error:
                return focused_error.strip()
        except Exception as e:
            self.logger.warning(f"Failed to extract focused error via LLM: {e}")

        # Fallback: Use hardcoded pattern matching
        return self._extract_error_with_patterns(error_report)

    def _extract_error_with_patterns(self, error_report: str) -> str:
        """Fallback pattern-based extraction if LLM fails."""
        lines = error_report.split('\n')

        # Find first AssertionError
        for line in lines:
            if 'AssertionError' in line and ':' in line:
                error_msg = line.split('AssertionError:')[-1].strip()
                return f"**Error Message**: {error_msg}"

        # Find any FAILED or ERROR line
        for line in lines:
            if 'FAILED' in line.upper() or 'ERROR' in line.upper():
                return f"**Error**: {line.strip()}"

        return "**Error**: Tests are failing - see error report"

    def _extract_and_analyze_error(self, error_report: str, buggy_code: str,
                                   specification: str = "", testbench_code: Optional[str] = None,
                                   rtl_guidance: str = "") -> Tuple[str, str, str]:
        """
        COMBINED: Extract focused error AND generate reasoning/guidance in ONE LLM call.

        Used when starting a new error (first iteration or previous error solved).

        Returns:
            Tuple of (focused_error, error_reasoning, fix_guidance)
        """
        prompt_error_report = self._prepare_error_report_for_prompt(error_report)
        is_assertion_insertion_task = self._is_cvdp_assertion_insertion_task(
            specification, buggy_code, testbench_code
        )
        is_verification_task = (
            self._is_verification_task(specification, buggy_code)
            and not is_assertion_insertion_task
        )
        if is_assertion_insertion_task:
            code_role = "SystemVerilog assertion code for the target RTL/verif file"
        elif is_verification_task:
            code_role = "SystemVerilog verification/testbench code"
        else:
            code_role = "SystemVerilog RTL code"
        is_coverage_task = self._is_cvdp_coverage_testbench_task(
            specification, buggy_code, testbench_code
        ) and not is_assertion_insertion_task
        is_bug_detection_task = self._is_cvdp_bug_detection_testbench_task(
            specification, buggy_code, testbench_code
        ) and not is_assertion_insertion_task
        task_guidance = ""
        if is_assertion_insertion_task:
            task_guidance = (
                "\nCVDP ASSERTION-INSERTION TASK GUIDANCE:\n"
                "- The generated target is the complete `rtl/` or `verif/` file for the assertion task, not a coverage-stimulus or bug-checker testbench.\n"
                "- For RTL targets, preserve original ports, parameters, assignments, latency, reset behavior, and outputs. Assertion code must observe only.\n"
                "- For separate assertion/verif modules, preserve the exact requested/existing checker module name/TOPLEVEL and DUT instantiation/reference style. Do not rename the checker module.\n"
                "- Prefer Cadence-compatible immediate assertions inside clocked procedural blocks; use simple clocked SVA only when clearly legal. Gate by reset/valid state and avoid `timescale.\n"
                "- For conditional checks, use procedural `if (cond) assert (expr) else $error(...)` or `assert (!cond || expr)`. Do not use the boolean implication token `->` inside assertion/property expressions.\n"
                "- Avoid old-Cadence hazards: no unclocked `assert property`, no property blocks without an explicit clock, no `->` boolean implication in assertions/properties, no `$past` outside a clocked block, no property-local variables inside sampled-value functions like `$past(array[idx])`, no `pure`/`virtual`, and no declarations after procedural statements.\n"
                "- For indexed/array temporal checks, use module-scope helper registers in `always_ff` blocks rather than SVA property-local variables in `$past`/`$stable` expressions.\n"
                "- If the clean/reference RTL is failing in all parameterizations, remove or weaken the failing internal/latency assertions instead of adding more complex checks. Preserve a clean simulation first, then keep only reset/valid/done-gated invariants that cannot fire in intermediate states.\n"
                "- If repeated reference failures remain, stop asserting algorithm content/latency and use local implementation invariants with unambiguous timing: reset/post-reset values, state/counter bounds, legal handshakes, done/valid stability, and known-output checks at stable states.\n"
                "- Return raw SystemVerilog only, with no markdown fences or prose.\n"
                "- If the reference RTL fails, the assertion is likely too strict or mistimed. If mutants are missed, add focused invariants using existing design signals.\n"
            )
        elif is_coverage_task:
            task_guidance = (
                "\nCVDP COVERAGE TASK GUIDANCE:\n"
                "- This is a coverage/stimulus testbench task, not a bug-detection checker task.\n"
                "- If the clean simulation fails because the testbench checker calls `$error`, remove/downgrade that checker to `$display` and keep the stimulus.\n"
                "- Never call `$coverage_control`, `$coverage_save`, `$coverage_get`, `$set_coverage_db_name`, or similar coverage system tasks; native xrun/imc coverage is controlled by the immutable harness.\n"
                "- Never use `force`, `release`, `bind`, hierarchical assignments to DUT internals, shell/system calls, or file writes; close coverage only through legal public DUT inputs.\n"
                "- Avoid `typedef struct packed` with `string` or other non-integral fields; use explicit task calls or simple parallel integral arrays for directed stimulus.\n"
                "- If coverage is below TARGET, add directed legal stimulus for uncovered state/branch/toggle paths.\n"
                "- Do not modify immutable src RTL and do not add assertions as a substitute for coverage.\n"
            )
        elif is_bug_detection_task:
            task_guidance = (
                "\nCVDP CHECKER TASK GUIDANCE:\n"
                "- The clean DUT must finish without `$error`/`$fatal`, while every injected `BUG_i` variant must make xrun return nonzero.\n"
                "- Build expected behavior from the specification and immutable DUT interface; do not use DUT outputs as expected values.\n"
                "- Add directed stimulus for the missed behavior, and fail only on real mismatches so the clean DUT still passes.\n"
                "- Do not modify immutable src RTL and do not suppress real checker failures.\n"
            )
        stage_guidance = self._failure_stage_guidance(
            prompt_error_report,
            is_coverage_task,
            is_bug_detection_task,
        )
        if is_assertion_insertion_task:
            stage_guidance = "\n\n".join(
                part for part in [stage_guidance, self._assertion_stage_guidance(prompt_error_report)] if part
            )
        is_cvdp_harness_context = bool(
            testbench_code and "# === CVDP HARNESS CONTEXT ===" in testbench_code
        )
        context_label = (
            "SANITIZED CVDP CONTEXT (runner/setup context only; no private checker or reference-model source)"
            if is_cvdp_harness_context else
            "TESTBENCH CODE (reference implementation)"
        )
        context_lang = "text" if is_cvdp_harness_context else "python"
        if is_assertion_insertion_task and is_cvdp_harness_context:
            reference_algorithm_note = "- Use the CVDP harness context to identify the target RTL file, immutable source behavior, ports, reset/clocking, and assertion scoring"
        else:
            reference_algorithm_note = (
                "- Use the public specification and evaluator messages for functional behavior; use sanitized CVDP context only for runner/setup constraints"
                if is_cvdp_harness_context else
                "- Check testbench reference algorithm (if provided) to see what's expected"
            )
        combined_prompt = f"""Analyze this {code_role} test failure and provide comprehensive debugging guidance.

ERROR REPORT:
```
{prompt_error_report}
```

BUGGY CODE:
```systemverilog
{buggy_code}
```

SPECIFICATION:
{specification}

{"SYSTEMVERILOG GUIDANCE (follow these design patterns when applicable):" if rtl_guidance else ""}
{rtl_guidance if rtl_guidance else ""}

{context_label + ":" if testbench_code else ""}
{f"```{context_lang}" if testbench_code else ""}
{testbench_code if testbench_code else ""}
{"```" if testbench_code else ""}

{task_guidance}
{stage_guidance}

YOUR TASK - Provide THREE sections:

### 1. FOCUSED ERROR (extract the most important error)
Extract the primary error that should be fixed:
- **Error Message**: <clear description>
- **Expected vs Actual**: (if present)
- **Clock Cycle**: (if present - more useful than ns)
- **Signals Involved**: (if mentioned)

### 2. ERROR ANALYSIS (2-3 sentences)
Analyze deeply:
- What the error means
- Point to specific code parts that cause this
{reference_algorithm_note}
- Identify any spec violations

### 3. FIX GUIDANCE (1 paragraph)
Provide specific, actionable guidance:
- What needs to change in the code
- Why this will work
- How it aligns with spec and testbench expectations
- For CVDP RTL repair, do not derive formula-level algorithm edits from private checker/reference internals; cite the public specification or buggy RTL when proposing functional changes

OUTPUT FORMAT:
### FOCUSED ERROR
<extracted error>

### ERROR ANALYSIS
<analysis>

### FIX GUIDANCE
<guidance>"""

        try:
            # Use ACE-RTL temperature if stage 1 uses ACE-RTL, otherwise use 0.1
            temp = self.ace_rtl_temperature if 1 in self.ace_rtl_stages else 0.1
            response = self.llm_client.call_llm(combined_prompt, temperature=temp)
            if response:
                # Extract all three sections
                focused_error = self._extract_section(response, "FOCUSED ERROR")
                error_analysis = self._extract_section(response, "ERROR ANALYSIS")
                fix_guidance = self._extract_section(response, "FIX GUIDANCE")
                if is_cvdp_harness_context:
                    error_analysis = self._sanitize_cvdp_feedback_guidance(error_analysis)
                    fix_guidance = self._sanitize_cvdp_feedback_guidance(fix_guidance)
                    if self._is_default_output_functional_mismatch(prompt_error_report) and self._contains_reset_prescription(fix_guidance):
                        fix_guidance = self._default_output_functional_guidance()

                return focused_error, error_analysis, fix_guidance
        except Exception as e:
            self.logger.warning(f"Failed combined error extraction/analysis: {e}")

        # Fallback: Use separate methods
        focused_error = self._extract_error_with_patterns(error_report)
        simple_reasoning = self._simple_error_reasoning(focused_error)
        return focused_error, simple_reasoning, "Fix the identified issue"

    def _extract_section(self, response: str, section_name: str) -> str:
        """Extract a specific section from LLM response."""
        pattern = rf'###\s*(?:\d+\.)?\s*{section_name}\s*\n(.+?)(?=###|$)'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _generate_code_related_reasoning(self, focused_error: str, buggy_code: str,
                                        error_report: str, specification: str = "",
                                        testbench_code: Optional[str] = None,
                                        rtl_guidance: str = "") -> Tuple[str, str]:
        """
        Use LLM to analyze error and generate both reasoning AND guidance for fixing.

        This is the "thinking" step that:
        - Analyzes what the error means
        - Identifies code issues and spec violations
        - Proposes specific guidance on how to fix it

        Returns:
            Tuple of (error_reasoning, fix_guidance)
        """
        # Detect compilation errors
        is_compilation_error = self._is_compilation_error(focused_error, error_report)
        prompt_error_report = self._prepare_error_report_for_prompt(error_report)
        is_assertion_insertion_task = self._is_cvdp_assertion_insertion_task(
            specification, buggy_code, testbench_code
        )
        is_coverage_task = self._is_cvdp_coverage_testbench_task(
            specification, buggy_code, testbench_code
        ) and not is_assertion_insertion_task
        is_bug_detection_task = self._is_cvdp_bug_detection_testbench_task(
            specification, buggy_code, testbench_code
        ) and not is_assertion_insertion_task

        # Build simple additional guidance
        additional_guidance = ""

        if is_compilation_error:
            additional_guidance = "\nNOTE: This is a COMPILATION ERROR (syntax issue). Focus on fixing the syntax errors while keeping the overall implementation logic intact.\n"

        additional_guidance += "\nSIMPLICITY PRINCIPLE: If you believe the buggy code is too complicated and it's becoming hopeless to get correct code from it, consider suggesting a simpler implementation approach.\n"
        if is_assertion_insertion_task:
            additional_guidance += (
                "\nCVDP ASSERTION-INSERTION TASK: Optimize for preserving the original RTL behavior while adding assertions that verify the requested properties. "
                "The output is the complete target `rtl/` or `verif/` file, not a coverage-stimulus or bug-checker testbench. "
                "For RTL targets, keep module ports/logic/reset/latency intact; for separate assertion modules, preserve the requested checker module and DUT instantiation/reference style. "
                "Use only existing signals or requested DUT hierarchy, gate assertions during reset/invalid cycles, and prefer procedural immediate assertions over fragile SVA; avoid unsupported SVA or `timescale. "
                "Do not rename separate checker modules; the module name must match the harness TOPLEVEL/file stem. "
                "For conditional checks, use procedural `if (cond) assert (expr) else $error(...)` or `assert (!cond || expr)`, never the boolean implication token `->` inside assertion/property expressions. "
                "Do not use unclocked `assert property`, property blocks without an explicit clock, `->` boolean implication in assertions/properties, `$past` outside a clocked block, property-local variables inside sampled-value functions like `$past(array[idx])`, `pure`/`virtual`, or declarations after procedural statements. "
                "For indexed/array temporal checks, capture indices/values in module-scope helper registers inside `always_ff` blocks instead of using local SVA variables in `$past` expressions. "
                "If the clean/reference RTL fails under all harness parameterizations, remove or weaken over-specific internal algorithm/latency assertions and keep only reset/valid/done-gated invariants that are guaranteed not to fire during intermediate states. "
                "If repeated reference failures remain, stop asserting algorithm content/latency and use local implementation invariants with unambiguous timing: reset/post-reset values, state/counter bounds, legal handshakes, done/valid stability, and known-output checks at stable states. "
                "Return raw SystemVerilog only, with no markdown fences or prose.\n"
            )
        elif is_coverage_task:
            additional_guidance += (
                "\nCVDP COVERAGE TASK: Optimize for clean simulation plus coverage TARGET. "
                "If checker/assert/$error logic fails on the clean DUT, remove or downgrade it. "
                "Do not call `$coverage_control`, `$coverage_save`, `$coverage_get`, `$set_coverage_db_name`, or any coverage system task; xrun/imc handles real coverage collection. "
                "Do not use `force`, `release`, `bind`, hierarchical assignments to DUT internals, shell/system calls, or file writes; coverage must come from public DUT inputs only. "
                "Avoid packed structs with `string` or other non-integral fields; Cadence requires packed struct fields to be integral. "
                "If coverage is low, add directed stimulus for uncovered branches/states/toggles; do not add more checkers.\n"
            )
        elif is_bug_detection_task:
            additional_guidance += (
                "\nCVDP CHECKER TASK: Optimize for clean DUT pass plus injected BUG_i detection. "
                "Keep real `$error`/`$fatal` checker failures for buggy variants, but fix timing/reference-model issues that fail the clean DUT. "
                "Add directed stimulus and expected-value checks for the missed mutant behavior; do not modify immutable src RTL.\n"
            )
        stage_guidance = self._failure_stage_guidance(
            prompt_error_report,
            is_coverage_task,
            is_bug_detection_task,
        )
        if is_assertion_insertion_task:
            stage_guidance = "\n\n".join(
                part for part in [stage_guidance, self._assertion_stage_guidance(prompt_error_report)] if part
            )
        is_cvdp_harness_context = bool(
            testbench_code and "# === CVDP HARNESS CONTEXT ===" in testbench_code
        )
        context_label = (
            "SANITIZED CVDP CONTEXT (runner/setup context only; no private checker or reference-model source)"
            if is_cvdp_harness_context else
            "TESTBENCH CODE (reference implementation - CHECK THIS to see expected algorithm)"
        )
        context_lang = "text" if is_cvdp_harness_context else "python"
        if is_assertion_insertion_task and is_cvdp_harness_context:
            reference_algorithm_note = "- Use the CVDP harness context to identify the target RTL file, immutable source behavior, ports, reset/clocking, and assertion scoring"
            fix_alignment_note = "- Preserve the immutable/source RTL behavior in the generated target file and align assertions with the prompt's verification requirements"
        else:
            reference_algorithm_note = (
                "- Use the public specification and evaluator messages for functional behavior; use sanitized CVDP context only for runner/setup constraints"
                if is_cvdp_harness_context else
                "- **IMPORTANT**: If testbench code is provided, check the reference algorithm to understand what's expected"
            )
            fix_alignment_note = (
                "- For CVDP RTL repair, do not derive formula-level algorithm edits from private checker/reference internals; cite the public specification or buggy RTL when proposing functional changes"
                if is_cvdp_harness_context else
                "- **CRITICAL**: If testbench has reference implementation, align your fix with that algorithm (don't assume standard implementations)"
            )

        is_verification_task = (
            self._is_verification_task(specification, buggy_code)
            and not is_assertion_insertion_task
        )
        if is_assertion_insertion_task:
            code_role = "SystemVerilog assertion code for the target RTL/verif file"
        elif is_verification_task:
            code_role = "SystemVerilog verification/testbench code"
        else:
            code_role = "SystemVerilog RTL code"
        reasoning_prompt = f"""Analyze this {code_role} error and provide both detailed reasoning and specific fix guidance.
{additional_guidance}

ERROR:
{focused_error}

BUGGY CODE:
```systemverilog
{buggy_code}
```

SPECIFICATION (original requirements):
{specification}

{"SYSTEMVERILOG GUIDANCE (follow these design patterns when applicable):" if rtl_guidance else ""}
{rtl_guidance if rtl_guidance else ""}

{context_label + ":" if testbench_code else ""}
{f"```{context_lang}" if testbench_code else ""}
{testbench_code if testbench_code else ""}
{"```" if testbench_code else ""}

ERROR REPORT (for additional context):
{prompt_error_report}

{stage_guidance}

YOUR TASK - Provide TWO sections:

### 1. ERROR ANALYSIS (2-3 sentences)
Analyze the error deeply:
- What does the error mean?
- Point to specific parts of the buggy code (signal names, initial/always blocks, tasks, line-level logic) that cause this
{reference_algorithm_note}
- Identify any specification violations or testbench-code mismatches

### 2. FIX GUIDANCE (1 paragraph)
Based on your analysis, provide specific, actionable guidance on how to fix this error:
- What needs to be changed in the code?
- Why will this fix work?
- What minimal changes are needed?
{fix_alignment_note}
- How does this align with the specification and testbench expectations?

Reference actual public signals, states, and assignments. Do not hardcode
expected outputs or emit task-specific formula patches unless the formula is
explicitly stated in the public specification.

OUTPUT FORMAT:
### ERROR ANALYSIS
<your analysis>

### FIX GUIDANCE
<your guidance>"""

        try:
            # Use ACE-RTL temperature if stage 1 uses ACE-RTL, otherwise use 0.1
            temp = self.ace_rtl_temperature if 1 in self.ace_rtl_stages else 0.1
            reasoning_response = self.llm_client.call_llm(reasoning_prompt, temperature=temp)
            if reasoning_response:
                # Extract both ERROR ANALYSIS and FIX GUIDANCE sections
                error_analysis = ""
                fix_guidance = ""

                # Extract ERROR ANALYSIS section
                analysis_match = re.search(r'###\s*(?:1\.)?\s*ERROR ANALYSIS\s*\n(.+?)(?=###|$)',
                                          reasoning_response, re.DOTALL | re.IGNORECASE)
                if analysis_match:
                    error_analysis = analysis_match.group(1).strip()

                # Extract FIX GUIDANCE section
                guidance_match = re.search(r'###\s*(?:2\.)?\s*FIX GUIDANCE\s*\n(.+?)(?=###|$)',
                                          reasoning_response, re.DOTALL | re.IGNORECASE)
                if guidance_match:
                    fix_guidance = guidance_match.group(1).strip()

                # If extraction failed, use full response as reasoning
                if not error_analysis:
                    error_analysis = reasoning_response.strip()
                if not fix_guidance:
                    fix_guidance = "Apply minimal fixes to address the error identified above"
                if is_cvdp_harness_context:
                    error_analysis = self._sanitize_cvdp_feedback_guidance(error_analysis)
                    fix_guidance = self._sanitize_cvdp_feedback_guidance(fix_guidance)
                    if self._is_default_output_functional_mismatch(prompt_error_report) and self._contains_reset_prescription(fix_guidance):
                        fix_guidance = self._default_output_functional_guidance()

                return error_analysis, fix_guidance
        except Exception as e:
            self.logger.warning(f"Failed to generate code-related reasoning: {e}")

        # Fallback to simple reasoning if LLM fails
        simple_reasoning = self._simple_error_reasoning(focused_error)
        return simple_reasoning, "Fix the identified issue in the code"
