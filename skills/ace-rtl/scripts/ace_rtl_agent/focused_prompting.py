# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused debugger implementation mixins."""

import os
import re
from typing import Dict, List, Optional, Tuple

from .focused_debugger_types import FixAttempt


class FocusedPromptingMixin:
    def create_focused_prompt(self,
                             buggy_code: str,
                             error_report: str,
                             iteration: int,
                             specification: str = "",
                             testbench_code: Optional[str] = None,
                             context_code: Optional[str] = None,
                             per_signal_mismatches: Optional[List[Dict]] = None) -> Tuple[str, str, str]:
        """
        Create minimal, focused prompt.

        Structure:
        1. Buggy code
        2. Error report (full)
        3. Specific error with reasoning (uses spec for analysis)
        4. History of attempts
        5. Testbench (optional)

        Note: Specification is now shown in prompt and used for error reasoning.

        Returns:
            Tuple of (prompt, focused_error, fix_guidance)
        """
        is_assertion_insertion_task = self._is_cvdp_assertion_insertion_task(
            specification, buggy_code, testbench_code
        )
        is_lint_optimization_task = self._is_cvdp_lint_optimization_task(
            specification, buggy_code, testbench_code
        ) and not is_assertion_insertion_task
        is_verification_task = (
            self._is_verification_task(specification, buggy_code)
            and not is_assertion_insertion_task
            and not is_lint_optimization_task
        )
        is_coverage_task = self._is_cvdp_coverage_testbench_task(
            specification, buggy_code, testbench_code
        ) and not is_assertion_insertion_task
        is_bug_detection_task = self._is_cvdp_bug_detection_testbench_task(
            specification, buggy_code, testbench_code
        ) and not is_assertion_insertion_task
        is_rtl_modification_task = (
            self._is_cvdp_rtl_modification_task(specification, buggy_code, testbench_code)
            and not is_assertion_insertion_task
            and not is_lint_optimization_task
            and not is_verification_task
        )
        spec_lower = specification.lower()
        requires_target_file_sections = (
            "// target_file:" in spec_lower
            or "return all generated target files" in spec_lower
            or "return all target files" in spec_lower
        )
        prompt_error_report = self._prepare_error_report_for_prompt(error_report)
        if is_assertion_insertion_task:
            code_role = "SystemVerilog assertion code for the target RTL/verif file"
            task_instruction = (
                "Generate the complete corrected target SystemVerilog file for this assertion-insertion task. "
                "If the target is an RTL file, insert the requested assertions while preserving module name, ports, parameters, existing logic, latency, reset behavior, and file role. "
                "If the prompt requests a separate assertion/verif module, preserve the exact existing checker module name/TOPLEVEL and instantiate/reference the DUT as specified. "
                "Add only Cadence-compatible assertion logic needed by the prompt. Return raw SystemVerilog only, with no markdown fences."
            )
        elif is_lint_optimization_task:
            code_role = "synthesizable RTL code for the lint/optimization target file"
            task_instruction = (
                "Generate the complete corrected target RTL file for this lint/optimization task. "
                "Preserve the exact module interface and externally visible behavior while fixing lint issues or reducing the requested synthesis metric(s). "
                "The result must pass both functional sanity simulation and Yosys synthesis/metric checks. Return raw synthesizable SystemVerilog only, with no markdown fences."
            )
        elif is_rtl_modification_task:
            code_role = "synthesizable RTL code for the CVDP target file"
            task_instruction = (
                "Generate the complete corrected RTL for this CVDP RTL implementation task. "
                "Preserve the exact module interface and harness-visible behavior. If the target needs helper modules, include each required synthesizable module exactly once. "
                "A focused top-level rewrite is allowed when the inherited FSM/microcode structure keeps failing the observable harness contract. "
                + (
                    "Return every declared target file as a separate `// TARGET_FILE: <filename>` section. "
                    if requires_target_file_sections
                    else ""
                )
                + "Return raw synthesizable SystemVerilog only, with no markdown fences."
            )
        elif is_verification_task:
            code_role = "SystemVerilog verification/testbench code"
            task_instruction = (
                "Generate the complete corrected SystemVerilog testbench/stimulus file. "
                "It may use initial blocks, tasks, $display/$error/$finish, delays, and other verification constructs "
                "needed by the harness. The testbench must always terminate with bounded waits/timeouts and $finish. "
                "Do not modify the immutable UUT RTL."
            )
        else:
            code_role = "synthesizable RTL code"
            task_instruction = "Generate the complete corrected synthesizable RTL module. Do not include verification-only constructs."
        verification_runtime_rules = ""
        if is_assertion_insertion_task:
            verification_runtime_rules = """
- CVDP assertion-insertion task: keep the original RTL functionality intact and add assertions inside the generated RTL file.
- For RTL target files, preserve module name, port list, parameters, combinational/sequential assignments, latency, reset behavior, and all existing outputs. Do not replace the design with an assertion-only wrapper.
- For separate `verif/` assertion modules, preserve the exact requested/existing checker module name, file role, and DUT instantiation/interface; do not modify immutable RTL or rename the checker module.
- Prefer simple Cadence-compatible inline assertions: inside a clocked always_ff/always block, use `if (!reset_condition) assert (condition) else $error("clear message ...");`.
- For conditional checks, use procedural `if (cond) assert (expr) else $error(...)` or `assert (!cond || expr)`. Do not use the boolean implication token `->` inside assertion/property expressions.
- Use only signals that exist in the target module or requested DUT hierarchy. If the prompt names a clock/reset, use those exact names. If no clock is available, use an always_comb block with immediate assertions for combinational invariants.
- Do not drive or assign design outputs from assertion blocks. Assertions must observe behavior only.
- For reset assertions, check post-reset state on the relevant clock edge and disable/skip functional assertions while reset is active.
- If the clean/reference RTL is failing assertions, make the generated assertions less strict before adding new ones. In particular, avoid exact internal algorithm-shape or cycle-count assertions for complex structures unless the signal timing is unambiguous under every parameterization.
- For complex data structures, prefer safe observable checks at `done`/valid states and simple one-cycle protocol/reset invariants over deep internal tree/array traversal properties that can fire during intermediate states.
- If reference RTL still fails after weakening functional/latency assertions, use local implementation invariants that cannot disturb behavior and are unambiguous under all parameters: reset/post-reset values, state/counter bounds, legal handshakes, done/valid stability, and known-output checks at stable states.
- Avoid unsupported/fragile forms unless the prompt explicitly requires them: no unclocked `assert property (@*)`, no property/endproperty blocks without an explicit clock, no `->` boolean implication in assertions/properties, no `$past` outside a clocked block, no property-local variables inside `$past`/`$rose`/`$fell`/`$stable`, no `pure` or `virtual`, no classes, no dynamic arrays, no declarations after procedural statements, and no `timescale.
- For array/indexed temporal checks, do not write `property ... int idx; ... $past(array[idx])`. Instead, capture needed indices/values in module-scope helper registers in an `always_ff` block, then assert the captured relationship procedurally.
- If helper values are needed, declare `logic`/`integer` helpers at module scope or at the very start of a procedural block before any assignment. Avoid declaring `int i` inside a `for` header if the generated block also needs old-Cadence compatibility.
- Return raw SystemVerilog only. Do not include markdown fences or explanatory text.
- Error messages should mention the violated requirement and key signal values.
"""
        if is_lint_optimization_task:
            verification_runtime_rules = """
- CVDP lint/optimization task: edit only the target RTL implementation. Do not generate a testbench, checker, assertions, markdown, or explanatory text.
- Preserve module name, parameter names/defaults, port names, widths, directions, reset polarity, clocking, valid/done protocol, output latency, and externally observable behavior unless the prompt explicitly requests a latency optimization.
- Always prioritize the `sanity` service over the `synth` service. If functional tests fail, repair behavior first even if area worsens.
- For lint cleanup, use explicit casts/slices/extension for width mismatches, give combinational outputs default assignments, fully cover case/if branches, remove unused declarations, and keep exactly one procedural/continuous driver per signal.
- For area/cell/wire optimization, reduce only requested metrics: inline one-use wires, remove duplicate registers, share common subexpressions, simplify constant conditions, replace large duplicated case trees with compact arithmetic or loops when synthesizable, and avoid unnecessary packed/unpacked arrays.
- If cocotb reports `ValueError: Unable to accurately represent` or X/Z values, fix deterministic reset/default assignments first; every output and state bit must be known after reset and on all branches.
- For Icarus warnings about constant/indexed selects in `always_*` processes, extract packed-vector subfields through continuous `assign` statements, wires, or helper logic before procedural use.
- For multi-cycle RTL, pipeline mode/op-select sideband controls with the accepted data and use the delayed controls for result selection; do not use live inputs to select delayed output behavior.
- For fixed-latency datapaths, align the result register and valid/done flag to the same public output cycle. If the checked output is still the reset/default value when valid/done is asserted, inspect off-by-one valid-pipe indexing or a result register updated one clock too late.
- For registered outputs, do not clear the result register in an idle/default branch that can run one cycle before a separately pipelined valid/done flag is observed. Hold the computed result or update/clear the result under the same pipeline stage that drives the public valid/done flag.
- For arithmetic specified by public constants or polynomial coefficients, derive terms from the bits explicitly set in the public coefficient. For finite-field or polynomial constant multipliers, include the shifted/multiplied term for each set coefficient bit and omit every unset-bit term. Do not invent extra terms and do not copy private checker/reference formulas.
- For pattern/tolerance matchers, count Hamming distance over exactly the active WIDTH, clear the count for every candidate pattern, assert match for exact/within-tolerance cases, and deassert match for one bit beyond tolerance.
- For serial, PWM, sigma-delta, VGA, and cache controllers, preserve cycle-level protocol timing: idle levels, valid/ready/done behavior, output latency, and one-clock counter/FSM advancement.
- Keep synthesis-friendly SystemVerilog: no initial blocks, procedural/assignment delays, $display/$error/$fatal/$finish, randomization, force/release, classes, dynamic arrays, DPI, bind, or SVA. Legal `module name #(parameter ...)` headers and `u_mod #(...)` parameter overrides are allowed.
- Avoid broad algorithm rewrites unless the current structure cannot meet the requested metric. Prefer minimal behavior-preserving changes with clear reset/default handling.
"""
        if is_rtl_modification_task:
            verification_runtime_rules = f"""
- CVDP RTL implementation task: edit only the target RTL implementation. Do not generate a testbench, checker, assertions, markdown, or explanatory text.
- Preserve the exact module name, parameters, ports, widths, reset polarity, clocking, and checked output timing.
- If multiple generated RTL files/modules are required by the prompt or compile command, provide each required module exactly once using the natural module names.
- If the specification/contract lists multiple target files, provide one complete section per target headed by `// TARGET_FILE: <filename>`; keep wrappers and core modules mutually consistent.
- Legal parameterized module headers and instance parameter overrides are allowed; do not use procedural/assignment delay controls such as `#5`, standalone `#(...)`, or `<= #1`.
- Icarus Verilog rejects procedural declarations after executable statements. Declare temporary `logic`/`integer` variables at module scope or at the top of an `always`/`always_ff`/`always_comb` block before any assignment, `if`, or `case`.
- Icarus warns on constant/indexed selects in `always_*` processes. Avoid direct packed-vector constant/indexed part-selects there; extract bytes/fields with continuous `assign` statements, wires, or helper combinational logic first.
- For multi-cycle datapaths, register mode/op-select sideband controls alongside accepted data and use the delayed controls at the output stage.
- For fixed-latency datapaths, make the result register update no later than the public valid/done cycle; an all-zero/default output with valid/done asserted usually means the valid pipe and result pipe are off by one cycle.
- If a functional mismatch reports valid/done at the checked cycle with a reset/default data value, do not focus only on adding reset initialization unless the simulator reports X/Z or uninitialized state. Prioritize aligning result-register updates, output gating/clearing, accepted-data storage, and delayed operation/key sideband controls.
- If the checked result is a register, avoid default/idle branches that clear it immediately before a delayed valid/done flag exposes it. Hold the computed result until the visible transaction completes, or gate result clearing with the same pipeline stage as the public valid/done signal.
- For public coefficient-based arithmetic, derive included terms from set bits in the public coefficient and omit unset-bit terms. For finite-field or polynomial constant multipliers, include one term per set coefficient bit only; do not use private checker/reference formulas.
- If repeated local changes do not satisfy the harness, replace the failing internal FSM/microcode with a compact synthesizable implementation that directly matches the CVDP harness-visible contract.
- Every output and state bit must be deterministic after reset and on every branch; avoid X/Z from uninitialized memories, arrays, counters, or partial assignments.
"""
        if is_verification_task:
            verification_runtime_rules = """
- Keep stimulus finite and deterministic; avoid unbounded `forever` loops or `wait` statements without a watchdog timeout.
- After all directed checks and coverage sampling complete, call `$finish`; on failure, call `$error` and then finish.
- Preserve the existing UUT instantiation, port names, clock/reset sequencing, and harness-visible file/module name.
- This CVDP task scores generated verification code. The clean DUT must run without `$error`, injected `BUG_i` DUT variants must trigger `$error`, and coverage tasks must reach the `TARGET` value from `.env`.
- Prefer Cadence Incisive-compatible immediate checks: `if (actual !== expected) $error(...)`. Do not use `assert property (@*)` or other unclocked SVA forms that Incisive rejects.
- In procedural blocks and tasks, declare variables at the start of the block before statements; avoid declaration-with-initializer after procedural statements.
- Do not mask failures by finishing before checkers run. Keep a watchdog timeout, but only call `$finish` after all expected checks and coverage-driving sequences have completed.
"""
        if is_bug_detection_task and not is_coverage_task:
            task_instruction = (
                "Generate the complete corrected SystemVerilog checker testbench file. "
                "This checker task is scored by passing the clean DUT and detecting every injected `BUG_i` mutant. "
                "Preserve and extend the provided stimulus, build a reference model from the specification, and fail the simulation only on real DUT/spec mismatches."
            )
            verification_runtime_rules = """
- CVDP checker task: the clean DUT must run to `$finish` with no `$error`, `$fatal`, failed assertions, or infinite waits.
- Every injected `BUG_i` DUT variant must trigger a checker failure and make xrun return nonzero. A good pattern is `error_count++` on each mismatch and final `if (error_count != 0) $fatal(1, ...)`.
- Do not downgrade or suppress `$error`/`$fatal`. Unlike coverage tasks, mutation detection requires real failing checkers.
- Keep the existing stimulus generator and add missing directed cases for each specified behavior, boundary condition, invalid input, reset/idle sequence, and protocol transition that could expose a mutant.
- Compare all architecturally visible outputs against a local expected/reference model. Do not compare a signal to itself or use the DUT output as the expected value.
- Align comparisons to the DUT latency, clocking, valid signals, and reset behavior. Avoid checking outputs while reset is active or before the DUT has produced valid outputs.
- Preserve the existing testbench module/file name, UUT instantiation, port names, and harness-visible generated file path. Do not modify immutable `src/` RTL.
- Keep Cadence Incisive/Xcelium compatibility: no classes, dynamic arrays, unsupported SVA, unclocked `assert property (@*)`, or declarations after procedural statements.
- Do not emit a `timescale directive; some immutable Verilog DUT files omit timescale and native xrun treats mixed timescale state as an elaboration fatal.
"""
        if is_coverage_task:
            task_instruction = (
                "Generate the complete corrected SystemVerilog coverage/stimulus testbench file. "
                "This coverage task is scored by clean simulation plus coverage TARGET, not by functional bug detection. "
                "Drive the immutable DUT through many legal deterministic scenarios and finish cleanly."
            )
            verification_runtime_rules = """
- CVDP coverage task: the clean DUT must run to `$finish` without `$error`, `$fatal`, `$stop`, failed assertions, or infinite waits.
- Do not add a reference-model checker unless the prompt explicitly requires one. For coverage-only work, focus on legal stimulus that exercises real DUT behavior; diagnostics may print observations, but the generated testbench must not fake, bypass, or overwrite coverage results.
- Do not call simulator coverage system tasks (`$coverage_control`, `$coverage_save`, `$coverage_get`, `$set_coverage_db_name`, etc.). The CVDP harness already runs xrun with coverage enabled and reports real coverage through imc.
- Do not use `force`, `release`, `bind`, hierarchical assignments to DUT internals, shell/system calls, or file writes. Coverage must come only from legal public DUT inputs and protocol-visible behavior.
- Maximize coverage by directed stimulus: reset asserted/deasserted more than once, idle/no-op cycles, all small enumerated values, boundary values, alternating/random-like patterns, valid/invalid handshakes, FSM transitions, back-to-back transactions, and timeout/flush paths where legal.
- If coverage is below TARGET, add new stimulus sequences for uncovered branches and state transitions; do not add assertions/checkers as a substitute for coverage.
- Instantiate only the DUT and preserve the existing testbench module/file name and DUT port names. Do not modify immutable `src/` RTL.
- Keep Cadence Incisive compatibility: no classes, dynamic arrays, unsupported SVA, unclocked `assert property (@*)`, or declarations after procedural statements.
- Avoid packed structs or packed unions that contain `string`, real, unpacked arrays, or other non-integral fields. For directed cases, prefer explicit task calls or simple parallel integral arrays.
- Initialize every driven input before reset release and avoid X/Z stimulus unless the prompt explicitly asks for it.
- Use bounded repeat loops and a watchdog that prints a message and `$finish`es cleanly; do not use a watchdog `$error` in coverage tasks.
"""
        stage_guidance = self._failure_stage_guidance(
            prompt_error_report,
            is_coverage_task,
            is_bug_detection_task,
        )
        if is_assertion_insertion_task:
            stage_guidance = "\n\n".join(
                part for part in [stage_guidance, self._assertion_stage_guidance(prompt_error_report)] if part
            )
        if is_lint_optimization_task:
            stage_guidance = "\n\n".join(
                part for part in [
                    stage_guidance,
                    self._rtl_optimization_stage_guidance(
                        prompt_error_report,
                        specification=specification,
                        code=buggy_code,
                    ),
                ] if part
            )
        if is_rtl_modification_task:
            stage_guidance = "\n\n".join(
                part for part in [
                    stage_guidance,
                    self._rtl_implementation_stage_guidance(
                        prompt_error_report,
                        specification=specification,
                        code=buggy_code,
                    ),
                ] if part
            )

        # Retrieve RTL coding rules for RTL tasks only. Verification/stimulus
        # tasks need harness-aligned finite testbench behavior instead.
        rtl_guidance = ""
        if not is_verification_task and not is_assertion_insertion_task and specification and not self.rtl_rules_retrieved:
            try:
                if self.rtl_rule_advisor.retrieve_rules_from_specification(specification):
                    rules = self.rtl_rule_advisor.get_retrieved_rules()
                    rtl_guidance = self.rtl_rule_advisor.format_advice_for_agent(rules, include_detailed=True)
                    self.rtl_rules_retrieved = True
                    self.logger.info(f"Retrieved {len(rules)} RTL coding rules for guidance")
            except Exception as e:
                self.logger.warning(f"RTL rule retrieval failed: {e}")
        elif self.rtl_rules_retrieved:
            # Use cached rules
            try:
                rules = self.rtl_rule_advisor.get_retrieved_rules()
                if rules:
                    rtl_guidance = self.rtl_rule_advisor.format_advice_for_agent(rules, include_detailed=True)
            except Exception as e:
                self.logger.debug(f"Could not get cached rules: {e}")

        # Check if stuck on current error for too long without progress
        force_new_error = False
        if self.current_focused_error and not self.error_solved and len(self.fix_history) >= 5:
            recent = self.fix_history[-5:]
            recent_on_same = [a for a in recent if a.focused_error == self.current_focused_error]
            if len(recent_on_same) >= 4:
                recent_counts = [a.mismatch_count for a in recent_on_same if a.mismatch_count >= 0]
                if recent_counts and min(recent_counts) >= max(recent_counts) * 0.85:
                    force_new_error = True
                    self.logger.info(f"Forcing new error focus - stuck on same error for 5+ iterations with <15% improvement")
        if self.current_focused_error and not self.error_solved and not force_new_error:
            focused_kind = self._error_focus_kind(self.current_focused_error)
            report_kind = self._error_focus_kind(prompt_error_report)
            if (
                focused_kind != "unknown"
                and report_kind != "unknown"
                and focused_kind != report_kind
            ):
                force_new_error = True
                self.logger.info(
                    "Switching error focus because current CVDP failure category changed "
                    f"from {focused_kind} to {report_kind}"
                )

        # Extract focused error - stick with current error if not solved yet
        if self.current_focused_error and not self.error_solved and not force_new_error:
            focused_error = self.current_focused_error
            self.logger.info(f"Continuing to focus on same error: {focused_error[:60]}...")

            error_reasoning, fix_guidance = self._generate_code_related_reasoning(
                focused_error, buggy_code, prompt_error_report, specification, testbench_code, rtl_guidance
            )
        else:
            if force_new_error:
                self.logger.info(f"Switching away from stalled error: {self.current_focused_error[:60] if self.current_focused_error else 'None'}...")
            # Extract new error AND generate reasoning/guidance in ONE LLM call
            focused_error, error_reasoning, fix_guidance = self._extract_and_analyze_error(
                prompt_error_report, buggy_code, specification, testbench_code, rtl_guidance
            )
            self.current_focused_error = focused_error
            self.error_solved = False
            self.logger.info(f"Focusing on new error: {focused_error[:60]}...")

        # Store guidance for later use (will be returned in metadata)
        self._current_fix_guidance = fix_guidance

        # Build history
        history = self._build_history_trajectory()

        # Check if we're using ACE-RTL for Stage 2 (code generation)
        # If yes, create shorter prompt to fit 32K token context limit
        using_ace_rtl_stage2 = 2 in self.ace_rtl_stages

        if using_ace_rtl_stage2:
            # SHORTER PROMPT FOR ACE-RTL (exclude error_report and testbench to fit 32K tokens)
            self.logger.info("Creating shortened prompt for ACE-RTL (32K context limit)")
            prompt = f"""# SystemVerilog Code Implementation - Iteration {iteration}

## 1. BUGGY {code_role.upper()} (to be fixed)
```systemverilog
{buggy_code}
```
"""
            # Step 3: Add context code section for multi-file datapoints
            section_num = 2
            if context_code:
                prompt += f"""
## {section_num}. CONTEXT CODE (helper modules - do NOT modify, shown for reference only)
```systemverilog
{context_code}
```

**NOTE**: The above context files are already correct. DO NOT generate them. Only fix the BUGGY CODE in section 1.
"""
                section_num += 1

            prompt += f"""
## {section_num}. SPECIFIC ERROR TO FIX
{focused_error}

{stage_guidance}

**Error Analysis:**
{error_reasoning}

## {section_num + 1}. FIX GUIDANCE (how to fix this error)
{fix_guidance}

## {section_num + 2}. SPECIFICATION
{specification}

## {section_num + 3}. HISTORY OF PREVIOUS ATTEMPTS
{history if history else "No previous attempts - this is your first try."}
"""
        else:
            # Full prompt for the external LLM (includes error_report and testbench)
            prompt = f"""# SystemVerilog Code Implementation - Iteration {iteration}

## 1. BUGGY {code_role.upper()} (to be fixed)
```systemverilog
{buggy_code}
```
"""
            # Step 3: Add context code section for multi-file datapoints
            section_num = 2
            if context_code:
                prompt += f"""
## {section_num}. CONTEXT CODE (helper modules - do NOT modify, shown for reference only)
```systemverilog
{context_code}
```

**NOTE**: The above context files are already correct. DO NOT generate them. Only fix the BUGGY CODE in section 1.
"""
                section_num += 1

            prompt += f"""
## {section_num}. ERROR REPORT
```
{prompt_error_report}
```

## {section_num + 1}. SPECIFIC ERROR TO FIX
{focused_error}

{stage_guidance}

**Error Analysis:**
{error_reasoning}

## {section_num + 2}. FIX GUIDANCE (how to fix this error)
{fix_guidance}

## {section_num + 3}. SPECIFICATION
{specification}

## {section_num + 4}. HISTORY OF PREVIOUS ATTEMPTS
{history if history else "No previous attempts - this is your first try."}
"""
            # Add testbench only if available (optional context) - only for LLM
            if testbench_code:
                section_num += 5
                # Mixed CVDP context may include Python, .env, and HDL files.
                tb_lang = (
                    "text"
                    if "# === CVDP HARNESS CONTEXT ===" in testbench_code
                    else ("systemverilog" if ("module " in testbench_code or "`timescale" in testbench_code) else "python")
                )
                prompt += f"""

## {section_num}. TESTBENCH / HARNESS CONTEXT (for understanding CVDP scoring, timing, and immutable source)
```{tb_lang}
{testbench_code}
```
"""

        # Add signal mismatch summary if available
        if per_signal_mismatches:
            failing_signals = [s for s in per_signal_mismatches if s["mismatches"] > 0]
            passing_signals = [s for s in per_signal_mismatches if s["mismatches"] == 0]
            if failing_signals:
                prompt += "\n## SIGNAL MISMATCH SUMMARY\n"
                prompt += "**ALL output signals and their status** (fix the target error WITHOUT breaking passing signals):\n"
                for s in sorted(failing_signals, key=lambda x: -x["mismatches"]):
                    prompt += f"  - `{s['signal']}`: {s['mismatches']} mismatches (first at time {s['first_mismatch_time']})\n"
                for s in passing_signals:
                    prompt += f"  - `{s['signal']}`: PASSING (0 mismatches)\n"
                prompt += "\n**CRITICAL**: Any fix MUST preserve all currently-passing signals. Do NOT introduce regressions.\n"

        # Public-spec reconciliation after stalled iterations
        if iteration > 5 and self._is_stalled():
            prompt += """
## SPEC RECONCILIATION WARNING

Mismatches have NOT improved significantly over the last several iterations.
The current repair strategy may be missing a timing, reset, mapping, or protocol
detail that is visible in the public specification or evaluator diagnostics.

**IMPORTANT**: Reconcile the candidate against the public specification and
observable evaluator messages. Do not infer hidden algorithms from private
checker/reference internals, and do not hardcode expected outputs.
"""

        response_example = (
            "// TARGET_FILE: rtl/example.sv\n"
            "// Full corrected SystemVerilog for that file\n\n"
            "// TARGET_FILE: rtl/another_file.sv\n"
            "// Full corrected SystemVerilog for that file"
            if requires_target_file_sections
            else "// Full corrected SystemVerilog file"
        )

        prompt += f"""

---

## YOUR TASK

Implement the FIX GUIDANCE above to generate corrected SystemVerilog code.

**Your job**:
- Follow the FIX GUIDANCE provided
- {'Use a focused rewrite if needed to satisfy the observable harness contract' if is_rtl_modification_task else 'Make the minimal changes needed to fix the specific error'}
- Ensure the fix aligns with the SPECIFICATION
- {"Preserve the interface and checked timing; internal structure may be simplified when it is causing repeated failures" if is_rtl_modification_task else "Don't rewrite working code - only fix what's broken"}
- {task_instruction}
{verification_runtime_rules}

## RESPONSE FORMAT

Provide ONLY the corrected code (no analysis or guidance needed - that's already done above):

<code>
{response_example}
</code>

**IMPORTANT**:
- Implement the guidance provided above
- Make MINIMAL changes
- Return the full replacement file, not a diff or partial snippet
"""

        return prompt, focused_error, fix_guidance
