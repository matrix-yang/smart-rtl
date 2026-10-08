# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused debugger implementation mixins."""

import os
import re
from typing import Dict, List, Optional, Tuple

from .focused_debugger_types import FixAttempt


class FocusedTaskDetectionMixin:
    def _is_verification_task(self, specification: str, code: str) -> bool:
        """Detect CVDP testbench/stimulus-generation tasks."""
        spec_low = specification.lower()
        code_low = code.lower()
        if "ace_task_kind=chipverilog_rtl_implementation" in spec_low:
            # Benchmark task identity is immutable.  A bad `tb_*` candidate must
            # be repaired back into the requested RTL, not allowed to redefine
            # the task as testbench generation.
            return False
        code_markers = [
            "$finish", "$display", "$error", "$fatal", "covergroup",
            "cover property", "always #", "module tb",
            "module testbench", "uut", "u_dut",
        ]
        explicit_spec_markers = [
            "testbench", "test bench", "stimulus", "coverage",
            "checker testbench", "checker functionality", "instantiate the",
            "uut",
        ]
        rtl_task_markers = [
            "modify the rtl", "rtl module", "synthesizable rtl",
            "target rtl", "generated rtl", "design specification",
        ]
        has_code_marker = any(marker in code_low for marker in code_markers)
        has_explicit_spec_marker = any(marker in spec_low for marker in explicit_spec_markers)

        # RTL implementation prompts often describe built-in testing control
        # units or "test vectors". That does not make the generated target a
        # testbench. Prefer code-level verification markers or an explicit
        # testbench/stimulus/checker request before switching roles.
        if any(marker in spec_low for marker in rtl_task_markers) and not has_code_marker:
            return has_explicit_spec_marker
        return has_code_marker or has_explicit_spec_marker

    def _is_cvdp_coverage_testbench_task(
        self, specification: str, code: str, testbench_code: Optional[str]
    ) -> bool:
        """Detect CVDP coverage-only testbench generation."""
        text = f"{specification}\n{code}\n{testbench_code or ''}"
        low = text.lower()
        has_cvdp_context = "# === cvdp harness context ===" in low
        has_target = re.search(r"\btarget\s*=", text, re.IGNORECASE) is not None
        has_coverage_flow = "coverage.cmd" in low or "test_coverage" in low or "test_report" in low
        has_bug_flow = re.search(r"\bnum_bugs\s*=", text, re.IGNORECASE) is not None or "bug_" in low
        return (
            has_cvdp_context
            and has_target
            and has_coverage_flow
            and not has_bug_flow
            and self._is_verification_task(specification, code)
        )

    def _is_cvdp_bug_detection_testbench_task(
        self, specification: str, code: str, testbench_code: Optional[str]
    ) -> bool:
        """Detect CVDP checker/bug-detection testbench generation."""
        text = f"{specification}\n{code}\n{testbench_code or ''}"
        low = text.lower()
        return (
            self._is_verification_task(specification, code)
            and (
                re.search(r"\bnum_bugs\s*=", text, re.IGNORECASE) is not None
                or "test_errors" in low
                or "bug_" in low
                or "checker_generator" in low
                or "checker functionality" in low
            )
        )

    def _is_cvdp_assertion_insertion_task(
        self, specification: str, code: str, testbench_code: Optional[str]
    ) -> bool:
        """Detect RTL assertion-insertion tasks."""
        text = f"{specification}\n{code}\n{testbench_code or ''}"
        low = text.lower()
        assertion_markers = [
            "create systemverilog assertions",
            "generate systemverilog assertions",
            "immediate assertion",
            "systemverilog assertions",
            "systemverilog assertion",
            "sva assertion",
            "assertion requirements",
            "assertion properties",
            "add assertions",
            "add an assertion",
            "add systemverilog assertions",
            "insert assertions",
            "assertions in the rtl",
            "assertions in rtl",
            "assertions must be created",
            "assertions have to be created",
            "assertions to validate",
            "assertions to verify",
            "assert property",
            "properties should",
            "sva",
        ]
        rtl_context_markers = [
            "rtl design",
            "rtl module",
            "generated rtl",
            "inside the rtl",
            "inside the design",
            "module named",
            "following should be the interface",
            "design specifications",
            "rtl specification",
            "interface:",
            "inputs:",
            "outputs:",
            "ports",
            "dut",
            "module",
            "during simulation",
            "rtl/",
            "verif/",
        ]
        bug_or_coverage_flow = (
            re.search(r"\bnum_bugs\s*=", text, re.IGNORECASE) is not None
            or "coverage.cmd" in low
            or "test_coverage" in low
            or "test_report" in low
            or "checker_generator" in low
        )
        return (
            any(marker in low for marker in assertion_markers)
            and any(marker in low for marker in rtl_context_markers)
            and not bug_or_coverage_flow
        )

    def _is_cvdp_lint_optimization_task(
        self, specification: str, code: str, testbench_code: Optional[str]
    ) -> bool:
        """Detect RTL lint/area/power-performance tasks."""
        text = f"{specification}\n{code}\n{testbench_code or ''}"
        low = text.lower()
        task_markers = [
            "lint code review",
            "area optimization",
            "latency optimization",
            "power-performance optimization",
            "power performance optimization",
            "reduce the utilization of cells",
            "reduce the utilization of wires",
            "reducing the utilization of cells",
            "reducing the utilization of wires",
            "optimize for power",
            "optimize for performance",
            "optimize for area",
        ]
        harness_markers = [
            "improvements",
            "percent_cells",
            "percent_wires",
            "number of cells",
            "number of wires",
            "synth.py",
            "synth.tcl",
            "yosys",
        ]
        return any(marker in low for marker in task_markers) or (
            "cvdp harness context" in low and any(marker in low for marker in harness_markers)
        )

    def _is_cvdp_rtl_modification_task(
        self, specification: str, code: str, testbench_code: Optional[str]
    ) -> bool:
        """Detect CVDP RTL modification/generation tasks."""
        text = f"{specification}\n{code}\n{testbench_code or ''}"
        low = text.lower()
        has_cvdp_context = "# === cvdp harness context ===" in low
        has_rtl_target_path = re.search(r"(?:^|[\s`'\"(])/code/rtl/[\w.$-]+\.(?:sv|v)\b|(?:^|[\s`'\"(])rtl/[\w.$-]+\.(?:sv|v)\b", low) is not None
        has_patch_style_rtl_request = "patch" in low and "single file" in low and "rtl/" in low
        if not (has_cvdp_context or has_rtl_target_path or has_patch_style_rtl_request):
            return False
        if any(marker in low for marker in ["coverage.cmd", "test_coverage", "num_bugs", "checker_generator"]):
            return False
        return any(
            marker in low
            for marker in [
                "create a verilog module",
                "create a systemverilog module",
                "implement a verilog module",
                "implement a systemverilog module",
                "top-level module",
                "modify the rtl module",
                "rtl module",
                "synthesizable rtl",
                "target rtl",
                "generated rtl",
            ]
        )

    def _rtl_implementation_stage_guidance(
        self,
        error_report: str,
        specification: str = "",
        code: str = "",
    ) -> str:
        """General guidance for RTL implementation failures."""
        guidance = [
            "CVDP RTL implementation triage:",
            "- This is an RTL implementation task. Return only synthesizable RTL for the requested target file(s); do not generate a testbench, checker, assertions, markdown, or explanatory text.",
            "- Preserve the exact module name, port list, parameters, reset polarity, clocking, and harness-visible output timing.",
            "- Legal parameterized module headers and instance parameter overrides are fine; avoid only real simulation delay controls such as `#5`, `#(...)` as a statement, or `<= #1`.",
            "- If the prompt or compiler requires multiple RTL modules/files, include each required module exactly once with its natural module name; do not hide missing helper modules inside a wrong top-level module.",
            "- Implement the behavior described by the prompt and provided context. If inherited microcode/FSM structure keeps failing, a simpler synthesizable implementation is acceptable only when it preserves the specified externally visible behavior.",
            "- Drive every output deterministically on reset and on all combinational/sequential paths; avoid X/Z values from uninitialized state, memories, or partial assignments.",
        ]
        if self._is_default_output_functional_mismatch(error_report):
            guidance.append(
                "- Functional simulation reached a checked valid/done cycle but the checked data output is still reset/default while the expected public output is nonzero. Do not treat this as a reset-only problem unless the log reports X/Z or uninitialized values. First inspect whether the result register updates one clock too late, whether the valid/done pipe is advanced ahead of the result pipe, whether output logic clears the bus on the checked cycle, and whether data/mode/key sideband registers are aligned with the accepted transaction."
            )
        if re.search(r"\b(?:Timeout for .+ expired|timeout expired|timed out)\b", error_report or "", re.IGNORECASE):
            guidance.append(
                "- The native CVDP run timed out. Treat this as a DUT progress/deadlock bug: make the FSM leave reset/idle, assert the expected valid/done/ready signal, bound internal loops/counters, and ensure the harness-observed transaction completes without changing the testbench."
            )
        missing_objects = sorted(set(
            re.findall(r"contains no object named\s+([A-Za-z_][A-Za-z0-9_$]*)", error_report)
        ))
        if missing_objects:
            guidance.append(
                "- Cocotb cannot access top-level object(s) "
                + ", ".join(f"`{name}`" for name in missing_objects[:8])
                + ". Restore those exact harness-visible port/signal or instance names in the TOPLEVEL module; "
                "do not rename them to aliases such as a different reset spelling/polarity."
            )
        missing_params = sorted(set(
            re.findall(r"parameter [`']([A-Za-z_][A-Za-z0-9_$]*)['`] not found", error_report, re.IGNORECASE)
        ))
        if missing_params:
            guidance.append(
                "- Elaboration reports missing parameter(s) "
                + ", ".join(f"`{name}`" for name in missing_params[:8])
                + ". Add them to the target module parameter list with compatible defaults and use/pass them without changing the public interface."
            )
        return "\n".join(guidance)

    def _rtl_optimization_stage_guidance(
        self,
        error_report: str,
        specification: str = "",
        code: str = "",
    ) -> str:
        """General guidance for RTL lint/optimization failures."""
        lower = error_report.lower()
        guidance = [
            "CVDP RTL lint/optimization triage:",
            "- This is an RTL rewrite task. Return the complete target RTL file only; do not generate testbench/checker code.",
            "- Preserve the exact module name, port list, parameters, reset polarity, valid/done timing, latency unless explicitly optimized by the prompt, and all externally visible behavior.",
            "- Legal parameterized module headers and instance parameter overrides are fine; avoid only real simulation delay controls such as `#5`, `#(...)` as a statement, or `<= #1`.",
            "- The objective has two services: `sanity` checks functional equivalence/behavior, and `synth` checks lint plus Yosys metrics such as cells, wires, wire bits, memories, or processes.",
        ]
        if re.search(r"\b(?:Timeout for .+ expired|timeout expired|timed out)\b", error_report or "", re.IGNORECASE):
            guidance.append(
                "- Sanity timed out. Fix forward progress before optimizing: drive reset/default state deterministically, make the expected valid/done/ready handshake complete, and avoid FSM states or counters that can stall forever."
            )
        if "sanity" in lower or "assertionerror" in lower or "test failed" in lower or "simulation" in lower:
            guidance.append(
                "- Sanity failed. First restore functional equivalence. Do not simplify away pipeline/register/FSM behavior unless the tests and prompt prove the latency change is allowed."
            )
        if any(token in lower for token in ["unable to accurately represent", "valueerror", "x/z", "uninitialized"]):
            guidance.append(
                "- The simulator cannot convert an X/Z value. Drive every output and internal state deterministically on reset and on every combinational path; never leave handshake, valid, done, PWM/bitstream, or data outputs unknown after reset."
            )
        if any(token in lower for token in ["one bit error", "exact match", "o_match", "tolerance", "pattern"]):
            guidance.append(
                "- Pattern matching failed. Implement the exact Hamming-distance rule: exact stored patterns must assert match, inputs with bit errors within the allowed tolerance must match, and one bit beyond tolerance must not match. Count differing bits across the active WIDTH only and reset/clear all temporary counts per comparison."
            )
        if any(token in lower for token in ["serial", "uart", "sigma", "delta", "audio", "pwm", "bitstream"]):
            guidance.append(
                "- Protocol/timing failed. Preserve cycle-level behavior: initialize idle levels, advance counters/FSMs exactly once per clock, hold outputs stable for the required bit/sample period, and avoid combinational glitches on serialized or PWM-like outputs."
            )
        if "synth" in lower or "yosys" in lower or "number of cells" in lower or "number of wires" in lower:
            guidance.append(
                "- Synthesis/optimization failed. Read the reported baseline/target metric and reduce only the requested metric(s): remove unused regs/wires/parameters, collapse redundant temporaries, simplify constant logic, share repeated expressions, reduce mux trees/case duplication, and avoid unnecessary arrays or wide intermediate signals."
            )
        if any(token in lower for token in ["latch", "undriven", "unused", "width", "multiple driven", "case", "verilator"]):
            guidance.append(
                "- Lint failed. Fix width casts/slices explicitly, assign every combinational output on every path, remove unused declarations, avoid inferred latches, and keep one driver per signal."
            )
        guidance.append(
            "- Keep the code synthesizable SystemVerilog only: no initial blocks, delays, $display/$error, assertions, classes, randomization, or testbench-only constructs."
        )
        guidance.append(
            "- Prefer small local changes over wholesale rewrites when the current design mostly works; the scoring flow punishes behavior regressions even if area improves."
        )
        return "\n".join(guidance)

    def _assertion_stage_guidance(self, error_report: str) -> str:
        """Deterministic guidance for RTL assertion-insertion failures."""
        lower = error_report.lower()
        guidance = [
            "CVDP assertion-insertion triage:",
            "- This is an assertion-generation task for the target `rtl/` or `verif/` file, not a coverage-stimulus or bug-checker testbench task.",
            "- If the target is an RTL file, preserve the original module name, port list, parameters, logic, latency, reset behavior, and output assignments. Assertions should observe behavior only.",
            "- If the prompt requests a separate assertion/verif module, preserve the exact existing checker module name and file role, instantiate/reference the DUT exactly as requested, and do not modify immutable RTL behavior.",
        ]
        if any(token in lower for token in ["syntax error", "*e,", "compilation", "elaboration", "xmvlog", "xmelab"]):
            guidance.append(
                "- Compilation/elaboration failed. Prefer simple Cadence-compatible immediate assertions inside existing procedural blocks; declare helper variables before statements and avoid unsupported SVA forms."
            )
            guidance.append(
                "- Old Cadence is strict: do not use `pure`, `virtual`, classes, dynamic arrays, unclocked `assert property`, property blocks without an explicit clock, `$past` outside a clocked block, or the boolean implication token `->` inside assertion/property expressions."
            )
            guidance.append(
                "- Do not use property-local variables inside sampled-value functions such as `$past(array[idx])`; Xcelium rejects this. For indexed history checks, capture the index/value in module-scope registers inside an `always_ff` block, or use immediate procedural assertions."
            )
        if any(token in lower for token in ["simulation ended with error", "$error", "assertion", "failed assertion"]):
            guidance.append(
                "- Assertion firing on the reference RTL means the assertion is too strict, mistimed, or not gated by reset/valid state. Gate functional checks during reset/invalid cycles and align checks with the design latency."
            )
            guidance.append(
                "- If all harness parameterizations fail on the clean/reference RTL, prioritize a clean run: remove or weaken over-specific internal algorithm/latency assertions, especially exact tree-shape/depth/iteration-count checks that depend on implementation details. Keep only assertions that are guaranteed by stable valid/done/reset state."
            )
            guidance.append(
                "- When prior functional/latency assertions still fire on the reference RTL, switch to local non-intrusive invariants derived from the existing implementation: reset/post-reset values, state/counter bounds, legal handshakes, done/valid stability, and known-output checks at stable states. Do not keep a sorting/algorithm/content assertion just because it sounds desirable if the clean RTL has already disproved its timing or capture assumptions."
            )
        if any(token in lower for token in ["missing", "did not", "without error", "bug", "mutation"]):
            guidance.append(
                "- If injected faulty RTL is not detected, add directed assertions for the stated invariants and boundary/protocol cases using only existing design signals."
            )
        guidance.append(
            "- Do not drive design outputs from assertion code. For RTL targets, do not replace the design with an assertion-only wrapper."
        )
        return "\n".join(guidance)

    def _failure_stage_guidance(
        self,
        error_report: str,
        is_coverage_task: bool,
        is_bug_detection_task: bool = False,
    ) -> str:
        """Return deterministic guidance based on the CVDP pytest stage that failed."""
        if not is_coverage_task:
            if not is_bug_detection_task:
                return ""

            lower = error_report.lower()
            bug_miss = re.search(
                r"BUG_([0-9]+)\s+simulation ended without error|injected bug\s+([0-9]+)",
                error_report,
                re.IGNORECASE,
            )
            missed_bug = None
            if bug_miss:
                missed_bug = next((g for g in bug_miss.groups() if g), None)

            guidance = [
                "CVDP checker-task triage:",
                "- The score is clean simulation plus mutation detection: the clean DUT must pass with no `$error`/`$fatal`, and every injected `BUG_i` DUT variant must make the generated checker fail the simulation.",
            ]
            if "test_sanity" in lower and "simulation ended with error" in lower:
                guidance.append(
                    "- `test_sanity` failed. The checker is too strict, mistimed, or using the wrong expected values for the correct DUT. Fix the reference model/timing first; do not silence checkers globally."
                )
            if "simulation ended without error" in lower or "did not detect injected bug" in lower:
                bug_text = f"BUG_{missed_bug}" if missed_bug is not None else "the reported BUG_i"
                guidance.append(
                    f"- `test_errors` missed {bug_text}. Add a directed stimulus case that exercises that behavior and add a reference-model comparison/assertion that fails only when the DUT deviates from the specification."
                )
            guidance.append(
                "- Use a local `error_count`; increment it on mismatches and call `$fatal(1, ...)` at the end only when `error_count != 0`, so bug simulations exit nonzero while the clean DUT exits 0."
            )
            guidance.append(
                "- Align checks to the DUT latency and valid/reset protocol. Sample outputs after the clock edge where the DUT has had time to update; skip checks during reset or invalid cycles."
            )
            return "\n".join(guidance)

        lower = error_report.lower()
        summary_lines = "\n".join(
            line.strip().lower()
            for line in error_report.splitlines()
            if re.match(r"(failed|error)\s+", line.strip(), re.IGNORECASE)
        )
        guidance = [
            "CVDP coverage-task triage:",
            "- The score is clean simulation plus coverage against TARGET; this is not a bug-detection task.",
        ]
        if "test_simulate" in summary_lines:
            guidance.append(
                "- `test_simulate` failed. First make the testbench compile and run to `$finish` on the clean DUT. "
                "If the failure is caused by testbench `$error`/assert/checker logic, remove or downgrade that checker."
            )
        if "$coverage_" in lower or "mssystf" in lower or "missyst" in lower:
            guidance.append(
                "- Cadence native xrun/imc already collects coverage through the harness command line. "
                "Remove all unsupported coverage system tasks such as `$coverage_control`, `$coverage_save`, "
                "`$coverage_get`, and `$set_coverage_db_name`; do not replace them with fabricated coverage output."
            )
        if "test_report" in summary_lines or "didn't achieve" in lower or "didn't achieved" in lower:
            guidance.append(
                "- Coverage failed. Add deterministic stimulus for reset, idle, branch, FSM, boundary, handshake, "
                "and toggle paths. Do not add assertions/checkers as a substitute for coverage."
            )
        if "test_coverage" in summary_lines or "coverage merge" in lower:
            guidance.append(
                "- Coverage merge failed. Keep the testbench simple and finite so xrun creates a fresh cov_work database."
            )
        return "\n".join(guidance)
