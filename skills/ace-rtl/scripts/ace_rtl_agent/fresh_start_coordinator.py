# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fresh Start Coordinator - Detects when iterative fixing is hopeless and restarts from scratch
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import re
from .utils.llm_client import LLMClient
from .utils.ace_model_client import AceRTLClient


@dataclass
class IterationSnapshot:
    """Snapshot of an iteration for trend analysis"""
    iteration: int
    test_passed: bool
    error_report: str
    rtl_code: str
    rtl_length: int
    error_summary: str  # Brief summary of main error
    mismatch_count: int = -1


class FreshStartCoordinator:
    """
    Detects when iterative RTL fixing is going in wrong direction and needs fresh start.

    This component analyzes iteration patterns and decides when the current approach
    is hopeless, then coordinates a complete restart with new RTL generation.
    """

    def __init__(self, model_name: str, logger, api_key: Optional[str] = None,
                 max_retries: int = 3, initial_backoff: float = 5.0,
                 ace_rtl_stages: set = None, ace_rtl_temperature: float = 0.2):
        """Initialize fresh start coordinator."""

        self.model_name = model_name
        self.logger = logger
        self.ace_rtl_stages = ace_rtl_stages if ace_rtl_stages is not None else set()
        self.ace_rtl_temperature = ace_rtl_temperature

        # LLM client for analysis/decisions.
        self.client = LLMClient(
            model=model_name,
            api_key=api_key,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            logger=logger
        )

        # Separate client for RTL generation: ACE-RTL when stage 0 is enabled
        if 0 in self.ace_rtl_stages:
            self.logger.info("FreshStartCoordinator: Using ACE-RTL for RTL generation (stage 0)")
            self.gen_client = AceRTLClient(
                logger=logger,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                temperature=ace_rtl_temperature
            )
        else:
            self.gen_client = self.client

        # Track iteration history for pattern analysis
        self.iteration_history: List[IterationSnapshot] = []
        self.fresh_start_count = 0
        self.last_fresh_start_iteration = 0

        # Reference to focused debugger for detailed history trajectory
        self.focused_debugger = None
        self.harness_dir: Optional[Path] = None

        self.logger.info("Initialized FreshStartCoordinator")

    def set_harness_directory(self, harness_dir: Path) -> None:
        """Set CVDP harness directory so fresh starts can see immutable context."""
        self.harness_dir = Path(harness_dir) if harness_dir else None
        if self.harness_dir:
            self.logger.info(f"FreshStartCoordinator harness context set: {self.harness_dir}")

    def mark_fresh_start(self, current_iteration: int) -> None:
        """Record a runner-managed fresh start."""
        self.fresh_start_count += 1
        self.last_fresh_start_iteration = current_iteration
        self.logger.info(
            "FreshStartCoordinator recorded runner-managed fresh start #%s at iteration %s",
            self.fresh_start_count,
            current_iteration,
        )

    def add_iteration_result(self, iteration: int, test_passed: bool, error_report: str,
                           rtl_code: str, mismatch_count: int = -1) -> None:
        """
        Add iteration result to history for trend analysis.

        Args:
            iteration: Iteration number
            test_passed: Whether test passed
            error_report: Error report from test
            rtl_code: Generated RTL code
        """
        # Extract brief error summary for pattern analysis
        error_summary = self._extract_error_summary(error_report)

        snapshot = IterationSnapshot(
            iteration=iteration,
            test_passed=test_passed,
            error_report=error_report,
            rtl_code=rtl_code,
            rtl_length=len(rtl_code),
            error_summary=error_summary,
            mismatch_count=mismatch_count,
        )

        self.iteration_history.append(snapshot)
        self.logger.debug(f"Added iteration {iteration} to fresh start analysis")

    def should_fresh_start(self, current_iteration: int) -> Tuple[bool, str]:
        """
        Let LLM analyze if current approach is hopeless and needs fresh start.

        Args:
            current_iteration: Current iteration number

        Returns:
            Tuple of (should_restart, reason)
        """
        # Let LLM make the decision with full context - no hardcoded constraints
        try:
            analysis_result = self._llm_analyze_hopeless_situation(current_iteration)
            return analysis_result

        except Exception as e:
            self.logger.error(f"Failed to analyze hopeless situation: {e}")
            return False, f"Analysis failed: {e}"

    def generate_fresh_rtl(self, specification: str, iteration_context: str = "",
                           current_iteration: Optional[int] = None) -> Tuple[str, str]:
        """
        Generate completely fresh RTL code from scratch using different approach.

        Args:
            specification: Original design specification
            iteration_context: Brief context about what went wrong in previous attempts

        Returns:
            Tuple of (fresh_rtl_code, generation_prompt)
        """
        try:
            # Create prompt for fresh RTL generation
            prompt = self._create_fresh_generation_prompt(specification, iteration_context)

            # Generate fresh RTL (uses ACE-RTL when stage 0 is enabled)
            gen_temp = self.ace_rtl_temperature if 0 in self.ace_rtl_stages else None
            if gen_temp is not None:
                response = self.gen_client.call_llm(prompt, temperature=gen_temp)
            else:
                response = self.gen_client.call_llm(prompt)
            if not response:
                raise ValueError("No response from LLM for fresh RTL generation")

            # Extract RTL code
            fresh_rtl = self._extract_rtl_from_response(response)

            # Update restart tracking
            self.fresh_start_count += 1
            self.last_fresh_start_iteration = (
                current_iteration if current_iteration is not None else len(self.iteration_history)
            )

            self.logger.info(f"Generated fresh RTL code (restart #{self.fresh_start_count})")
            return fresh_rtl, prompt

        except Exception as e:
            self.logger.error(f"Fresh RTL generation failed: {e}")
            raise

    def _llm_analyze_hopeless_situation(self, current_iteration: int) -> Tuple[bool, str]:
        """Use LLM to analyze if current debugging approach is hopeless - CONSERVATIVE approach."""

        # Get detailed history trajectory if available
        detailed_history = ""
        if self.focused_debugger and hasattr(self.focused_debugger, 'fix_history'):
            # Get the detailed history from focused debugger
            detailed_history = self.focused_debugger._build_history_trajectory()
            if not detailed_history:
                detailed_history = "No detailed history available yet"
        else:
            detailed_history = "Detailed history not available"

        coordinator_history = self._format_iteration_context(self.iteration_history)
        if coordinator_history and coordinator_history != "No iteration history available":
            detailed_history += f"\n\n**Coordinator Progress History**:\n{coordinator_history}"

        iterations_since_restart = current_iteration - self.last_fresh_start_iteration

        early_restart_reason = self._early_restart_reason(iterations_since_restart)
        if early_restart_reason:
            self.logger.info(f"Early fresh-start trigger: {early_restart_reason}")
            return True, early_restart_reason

        # CONSERVATIVE: Only consider restart if 12+ consecutive failures
        if iterations_since_restart < 12:
            reason = f"Only {iterations_since_restart} iterations since last restart - minimum 12 required before considering restart"
            self.logger.info(f"Auto-continue (conservative policy): {reason}")
            return False, reason

        # Build restart context
        restart_context = f"""
**Restart History**:
- Total restarts so far: {self.fresh_start_count}
- Last restart was at iteration: {self.last_fresh_start_iteration}
- Current iteration: {current_iteration}
- Iterations since last restart: {iterations_since_restart}
- **CONSERVATIVE POLICY**: Minimum 12 consecutive failures required before restart"""

        prompt = f"""Analyze if this RTL debugging approach is HOPELESS and needs a complete restart.

**DETAILED HISTORY TRAJECTORY**:
{detailed_history}

{restart_context}

**CONSERVATIVE RESTART POLICY**:
You are analyzing {iterations_since_restart} consecutive failed iterations. Be VERY CONSERVATIVE about restarting.

**ONLY restart if ALL of the following are true**:
1. At least 12 consecutive failures (we have {iterations_since_restart})
2. ZERO progress made (no tests improving, same errors repeating)
3. Multiple different fix approaches tried but all failed
4. Clear evidence of fundamental design flaw (not just implementation bugs)
5. Pretty sure continued debugging is HOPELESS

**Look for signs of progress (if ANY exist, CONTINUE)**:
- Different errors appearing (shows evolution)
- Test count improving (even 0->1 is progress)
- Different implementation approaches tried (shows exploration)
- Fix effectiveness improving (even partial fixes count)
- Any indication the approach might work with more iterations

**Critical Questions**:
- Is there ANY sign of progress in the history above?
- Have we tried genuinely different approaches, or same thing repeatedly?
- Is this a fundamental design flaw, or just needs more debugging?

**DECISION RULE**:
- If you see ANY progress or evolution -> CONTINUE
- If you're unsure -> CONTINUE (be conservative!)
- ONLY if you're VERY SURE it's hopeless -> RESTART

**CRITICAL: You MUST end your response with exactly one of these:**
- RESTART: [detailed reason why it's HOPELESS and needs fresh start]
- CONTINUE: [reason why current approach should continue]

**Your Analysis and Decision**:"""

        try:
            response = self.client.call_llm(prompt)
            if not response:
                return False, "No response from analysis"

            response = response.strip()

            # Look for RESTART or CONTINUE anywhere in the response
            if "RESTART:" in response:
                # Extract the reason after "RESTART:"
                restart_match = response.split("RESTART:", 1)
                if len(restart_match) > 1:
                    reason = restart_match[1].strip()
                    # Clean up common formatting
                    reason = reason.replace("**", "").strip()
                    self.logger.info(f"LLM recommends restart: {reason}")
                    return True, reason
                else:
                    self.logger.info("LLM recommends restart (no detailed reason)")
                    return True, "LLM analysis recommends fresh start"

            elif "CONTINUE:" in response:
                # Extract the reason after "CONTINUE:"
                continue_match = response.split("CONTINUE:", 1)
                if len(continue_match) > 1:
                    reason = continue_match[1].strip()
                    reason = reason.replace("**", "").strip()
                    self.logger.info(f"LLM recommends continue: {reason}")
                    return False, reason
                else:
                    self.logger.info("LLM recommends continue (no detailed reason)")
                    return False, "LLM analysis recommends continuing current approach"

            else:
                # Fallback: use heuristics to detect restart intent
                restart_keywords = ["fresh start", "restart", "start over", "abandon", "new approach", "fundamental", "hopeless"]
                continue_keywords = ["continue", "keep trying", "more iterations", "progress", "not ready"]

                response_lower = response.lower()
                restart_score = sum(1 for word in restart_keywords if word in response_lower)
                continue_score = sum(1 for word in continue_keywords if word in response_lower)

                if restart_score > continue_score and restart_score > 0:
                    self.logger.warning(f"No explicit RESTART/CONTINUE, but detected restart intent (score: {restart_score} vs {continue_score})")
                    return True, "Restart detected via heuristic analysis"
                else:
                    self.logger.warning(f"No explicit RESTART/CONTINUE, defaulting to continue (score: {restart_score} vs {continue_score})")
                    self.logger.debug(f"Response content: {response}")
                    return False, "LLM response unclear - defaulting to continue"

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return False, f"Analysis error: {e}"

    def _early_restart_reason(self, iterations_since_restart: int) -> Optional[str]:
        """Detect generic no-progress failure shapes that warrant a fresh attempt."""
        if iterations_since_restart < 3:
            return None

        recent = [
            snap for snap in self.iteration_history
            if snap.iteration > self.last_fresh_start_iteration
        ][-3:]
        if len(recent) < 3 or any(snap.test_passed for snap in recent):
            return None

        if all(self._is_default_output_functional_failure(snap.error_report) for snap in recent):
            return (
                "Three consecutive functional failures have checked outputs stuck at a "
                "default/zero value; start fresh to avoid a local patch loop around a "
                "dead or misaligned datapath."
            )

        return None

    @staticmethod
    def _is_default_output_functional_failure(error_report: str) -> bool:
        text = error_report.lower()
        if not any(marker in text for marker in ("does not match", "mismatch", "expected vs actual")):
            return False
        if any(marker in text for marker in ("syntax error", "compilation fails", "candidate_compile", "error: command")):
            return False

        default_markers = (
            "0x0 !=",
            "0 != ",
            "outputs 0x0",
            "output is all zeros",
            "output all zeros",
            "all-zero",
            "all zero",
            "default value",
        )
        return any(marker in text for marker in default_markers)

    def _format_iteration_context(self, iterations: List[IterationSnapshot]) -> str:
        """Format iteration history for LLM analysis."""

        if not iterations:
            return "No iteration history available"

        context_lines = []
        for iter_snap in iterations:
            status = "PASS" if iter_snap.test_passed else "FAIL"
            metric_text = (
                f", progress metric: {iter_snap.mismatch_count}"
                if iter_snap.mismatch_count >= 0 else ""
            )
            context_lines.append(
                f"Iteration {iter_snap.iteration}: {status} - {iter_snap.error_summary} "
                f"(Code: {iter_snap.rtl_length} chars{metric_text})"
            )

        return "\n".join(context_lines)

    def _create_fresh_generation_prompt(self, specification: str, context: str = "") -> str:
        """Create fresh generation prompt with high-level learnings and restart context."""

        # Extract lessons BEFORE incrementing counter so first restart also gets lessons
        lessons = self._llm_extract_high_level_lessons()

        lessons_section = ""
        if lessons:
            lessons_section = f"""

**High-Level Lessons from Previous Attempts**:
{lessons}
"""

        # Include restart reason/context if provided
        context_section = ""
        if context:
            context_section = f"""

**Why Previous Approach Failed**:
{context}
"""

        # Include per-signal mismatch context if available
        signal_section = ""
        if self.iteration_history:
            last_snapshot = self.iteration_history[-1]
            if hasattr(last_snapshot, 'error_report') and last_snapshot.error_report:
                import re
                signal_lines = []
                for line in last_snapshot.error_report.split('\n'):
                    if line.strip().startswith('Hint:') and 'Output' in line:
                        signal_lines.append(line.strip())
                if signal_lines:
                    signal_section = "\n**Signals That Were Failing** (pay special attention to these):\n"
                    for sl in signal_lines:
                        signal_section += f"  {sl}\n"

        spec_lower = specification.lower()
        is_chipverilog_rtl_task = (
            "ace_task_kind=chipverilog_rtl_implementation" in spec_lower
        )
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
        is_assertion_task = (
            not is_chipverilog_rtl_task
            and any(marker in spec_lower for marker in assertion_markers)
            and any(marker in spec_lower for marker in rtl_context_markers)
            and not any(
                marker in spec_lower
                for marker in ["coverage.cmd", "test_coverage", "test_report", "checker_generator", "num_bugs"]
            )
        )
        is_verification_task = (
            not is_chipverilog_rtl_task
            and any(
                marker in spec_lower
                for marker in ["testbench", "test bench", "stimulus", "test vectors", "tb_", "uut"]
            )
            and not is_assertion_task
        )
        is_coverage_task = (
            is_verification_task
            and "target" in spec_lower
            and any(marker in spec_lower for marker in ["coverage.cmd", "test_coverage", "test_report"])
            and not any(marker in spec_lower for marker in ["checker_generator", "num_bugs", "bug_"])
        )
        is_checker_task = is_verification_task and any(
            marker in spec_lower
            for marker in ["checker", "checker_generator", "bug_", "num_bugs", "mutation"]
        ) and not is_coverage_task
        is_lint_optimization_task = any(
            marker in spec_lower
            for marker in [
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
                "percent_cells",
                "percent_wires",
                "number of cells",
                "number of wires",
                "synth.py",
                "synth.tcl",
                "yosys",
            ]
        ) and not is_assertion_task and not is_verification_task
        is_plain_rtl_task = (
            is_chipverilog_rtl_task
            or (
                not is_assertion_task
                and not is_verification_task
                and not is_lint_optimization_task
                and (
                    "synthesizable rtl" in spec_lower
                    or "rtl module" in spec_lower
                    or "top-level module" in spec_lower
                    or "implement a verilog module" in spec_lower
                    or "implement a systemverilog module" in spec_lower
                    or re.search(r"(?:^|[\s`'\"(])/code/rtl/[\w.$-]+\.(?:sv|v)\b", spec_lower)
                    or re.search(r"(?:^|[\s`'\"(])rtl/[\w.$-]+\.(?:sv|v)\b", spec_lower)
                    or (
                        "patch" in spec_lower
                        and "single file" in spec_lower
                        and "rtl/" in spec_lower
                    )
                )
            )
        )
        if is_assertion_task:
            task_line = (
                "You are solving a CVDP RTL assertion-insertion problem. Respond with the complete "
                "target SystemVerilog `rtl/` or `verif/` file for the requested assertions. "
                "For RTL targets, preserve the original module name, ports, parameters, combinational/sequential logic, "
                "latency, reset behavior, and all output assignments. Assertions must observe behavior only and must not "
                "replace the design with a wrapper. For separate assertion/verif modules, preserve the requested checker "
                "module name/TOPLEVEL exactly and DUT instantiation/reference style. Do not rename the checker module. "
                "Prefer Cadence-compatible procedural immediate assertions inside clocked `always_ff`/`always` blocks; "
                "use clocked SVA only when it is simpler and clearly legal. Gate functional checks during reset/invalid "
                "cycles, and do not emit a `timescale directive. "
                "For conditional assertion checks, write procedural `if (cond) assert (expr) else $error(...)` or "
                "`assert (!cond || expr)`, never the boolean implication token `->` inside an assertion expression. "
                "Avoid old-Cadence hazards: no unclocked `assert property`, no property/endproperty block without an explicit "
                "clock, no `->` boolean implication in property/assert expressions, no `$past` outside a clocked procedural block, no property-local variables inside sampled-value "
                "functions like `$past(array[idx])`, no `pure`/`virtual`, no classes or dynamic arrays, and "
                "no declarations after procedural statements. Put helper `logic`/`integer` declarations at module scope or at "
                "the very start of a procedural block. For indexed temporal checks, capture indices/values in helper "
                "registers inside `always_ff` blocks rather than using SVA local variables in `$past` expressions. "
                "If the prior attempts failed on the clean/reference RTL across all parameterizations, simplify: remove "
                "or weaken over-specific internal algorithm/latency assertions and keep only reset/valid/done-gated "
                "invariants that are guaranteed not to fire during intermediate states. "
                "If repeated reference failures remain, stop asserting algorithm content/latency and use local "
                "implementation invariants with unambiguous timing: reset/post-reset values, state/counter bounds, "
                "legal handshakes, done/valid stability, and known-output checks at stable states. "
                "Return raw SystemVerilog only, with no markdown fences or prose."
            )
        elif is_coverage_task:
            task_line = (
                "You are solving a CVDP coverage/stimulus-generation problem. "
                "Respond with the complete SystemVerilog verification/testbench stimulus code for the generated `verif/` file. "
                "The clean DUT must run finite legal stimulus and finish cleanly; do not add reference-model checkers, assertions, "
                "`$error`, `$fatal`, or `$stop` unless the prompt explicitly requires them. "
                "Do not call `$coverage_control`, `$coverage_save`, `$coverage_get`, `$set_coverage_db_name`, or any coverage system task; "
                "the immutable CVDP harness already enables xrun coverage and reports real coverage through imc. "
                "Do not use `force`, `release`, `bind`, hierarchical assignments to DUT internals, shell/system calls, or file writes; "
                "coverage must come only from legal public DUT inputs and protocol-visible behavior. "
                "Avoid `typedef struct packed` with `string` or other non-integral fields; use explicit task calls or simple parallel integral arrays for directed cases. "
                "Increase coverage only by driving reset, idle, boundary, branch, FSM, handshake, and toggle stimulus. "
                "Do not modify immutable RTL, do not fake coverage reports, and do not emit a `timescale directive."
            )
        elif is_checker_task:
            task_line = (
                "You are solving a CVDP checker-generation problem. Respond with the complete "
                "SystemVerilog verification/testbench file for the generated `verif/` file. "
                "The clean DUT must finish without errors, while any real mismatch against the "
                "specification must call `$error`/`$fatal` so injected buggy DUT variants are detected. "
                "Preserve the UUT instantiation, use finite stimulus and bounded waits, do not modify immutable RTL, "
                "and do not emit a `timescale directive."
            )
        elif is_lint_optimization_task:
            task_line = (
                "You are solving a CVDP RTL lint/area/power-performance optimization problem. "
                "Respond with the complete target RTL file only. Preserve the exact module name, port list, "
                "parameters, reset polarity, clocking, valid/done behavior, output latency, and externally visible "
                "functionality while fixing lint issues or reducing the requested Yosys synthesis metric(s). "
                "Legal parameterized module headers and instance parameter overrides are allowed; avoid only real "
                "simulation delay controls such as `#5`, standalone `#(...)`, or `<= #1`. "
                "The generated RTL must pass both the functional `sanity` service and the `synth` metric service. "
                "For X/Z or `ValueError` simulator failures, drive all outputs and state deterministically after reset. "
                "For pattern/tolerance matchers, implement exact Hamming-distance counting over the active width and clear counts per candidate. "
                "For serial, PWM, sigma-delta, VGA, and cache controllers, preserve cycle-level protocol timing and output latency. "
                "Do not emit testbench code, assertions, markdown fences, or explanatory prose."
            )
        elif is_plain_rtl_task:
            benchmark_name = "ChipVerilog" if is_chipverilog_rtl_task else "CVDP"
            module_rule = (
                "Define the expected target module exactly once and do not define any "
                "testbench, wrapper, or additional module. "
                if is_chipverilog_rtl_task
                else "If the prompt requires helper modules, include each required "
                "synthesizable module exactly once. "
            )
            task_line = (
                f"You are solving a {benchmark_name} RTL implementation problem. Respond with only the complete "
                "synthesizable RTL for the requested target file. Preserve exact module names, ports, "
                "parameters, widths, reset polarity, clocking, latency, and externally visible behavior. "
                f"{module_rule}"
                "Do not emit testbench code, assertions, shell commands, markdown fences, patch prose, or explanatory text."
            )
        elif is_verification_task:
            task_line = (
                "You are solving a 'Test Plan to Testbench Stimulus Generation' problem. "
                "Respond with the complete SystemVerilog verification/testbench stimulus code for the generated file. "
                "The testbench must use finite stimulus, bounded waits/timeouts, and an explicit $finish after checks/coverage; "
                "do not modify the immutable UUT RTL."
            )
        else:
            task_line = (
                "You are solving a 'Specification to RTL Translation' problem. "
                "Respond with the RTL code translated from the specification."
            )

        harness_context = ""
        if is_verification_task or is_assertion_task or is_lint_optimization_task:
            harness_context = self._build_immutable_harness_context()

        harness_section = ""
        if harness_context:
            harness_section = f"""

**Immutable CVDP Harness Context**:
The following files are read-only context for matching ports, timing, scoring, and runner behavior. Do not regenerate these files.
```text
{harness_context}
```
"""

        prompt = f"""You are a helpful assistance.
Consider that you have a folder structure like the following:

    - rtl/*   : Contains files which are RTL code.
    - verif/* : Contains files which are used to verify the correctness of the RTL code.
    - docs/*  : Contains files used to document the project, like Block Guides, RTL Plans and Verification Plans.

When generating files, return the file name in the correct place at the folder structure.

{task_line}{context_section}{lessons_section}{signal_section}{harness_section}

----------------------------------------

{specification}"""

        return prompt

    def _build_immutable_harness_context(self, max_total_chars: int = 90000) -> str:
        """Read concise immutable CVDP context for verification fresh starts."""
        if not self.harness_dir:
            return ""
        harness_path = Path(self.harness_dir)
        src_dir = harness_path / "src"
        rtl_dir = harness_path / "rtl"
        parts: List[str] = []
        total = 0

        def add_file(path: Path, label: str, max_file_chars: int = 20000) -> None:
            nonlocal total
            if total >= max_total_chars:
                return
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return
            if len(content) > max_file_chars:
                keep = max_file_chars - len("\n... [FILE TRUNCATED] ...\n")
                head = max(0, int(keep * 0.65))
                tail = max(0, keep - head)
                content = content[:head] + "\n... [FILE TRUNCATED] ...\n" + content[-tail:]
            try:
                rel = path.relative_to(harness_path)
            except ValueError:
                rel = path
            block = f"# File: {rel} ({label})\n{content}\n"
            remaining = max_total_chars - total
            if len(block) > remaining:
                block = block[:remaining] + "\n... [HARNESS CONTEXT TRUNCATED] ...\n"
            parts.append(block)
            total += len(block)

        if src_dir.exists():
            for pattern, label in [
                (".env", "CVDP env/scoring knobs"),
                ("*.py", "CVDP pytest runner/checker"),
                ("*.sv", "immutable source/UUT HDL"),
                ("*.v", "immutable source/UUT HDL"),
                ("*.svh", "source include"),
                ("*.vh", "source include"),
            ]:
                for path in sorted(src_dir.glob(pattern)):
                    if path.is_file() and path.name not in {"__init__.py"}:
                        add_file(path, label)

        if rtl_dir.exists():
            for pattern in ("*.sv", "*.v", "*.svh", "*.vh"):
                for path in sorted(rtl_dir.glob(pattern)):
                    if path.is_file():
                        add_file(path, "immutable/helper RTL context")

        return "\n".join(parts)

    def _llm_extract_high_level_lessons(self) -> str:
        """Use LLM to extract high-level, general lessons from iteration history."""

        if not self.iteration_history or len(self.iteration_history) < 3:
            return ""  # Need some history to extract meaningful lessons

        # Format iteration history for LLM analysis
        history_summary = self._format_iteration_context(self.iteration_history)

        prompt = f"""Extract high-level, general lessons from these RTL generation failures.

**Iteration History**:
{history_summary}

**Your Task**:
Identify general patterns and extract 2-4 concise, high-level lessons that could help avoid similar mistakes in future attempts.

**Guidelines for Lessons**:
- Focus on methodology and approach, not specific technical details
- Keep lessons general and applicable to various RTL problems
- Avoid overfitting to specific error patterns or code structures
- Each lesson should be one concise bullet point
- Include all significant lessons learned - no artificial limits

**Examples of Good Lessons**:
- Follow specification interface requirements precisely
- Pay attention to timing behavior and synchronization requirements
- Prefer simple, clear implementations over complex architectures

**Your High-Level Lessons**:"""

        try:
            response = self.client.call_llm(prompt)
            if response and response.strip():
                # Clean up the response and extract lesson points
                lessons = self._clean_lesson_response(response)
                self.logger.info(f"LLM extracted {len(lessons.split('- '))-1} high-level lessons from iteration history")
                return lessons
            else:
                return ""

        except Exception as e:
            self.logger.error(f"Failed to extract lessons via LLM: {e}")
            return ""

    def _clean_lesson_response(self, response: str) -> str:
        """Clean and format LLM lesson response."""

        # Extract bullet points if they exist
        lines = response.strip().split('\n')
        lesson_lines = []

        for line in lines:
            line = line.strip()
            # Look for bullet points or numbered items
            if line.startswith('\u2022'):
                lesson_lines.append("- " + line[1:].lstrip())
            elif line.startswith('-') or line.startswith('*'):
                lesson_lines.append(line)
            elif line and len(lesson_lines) == 0:  # First substantial line if no bullets
                lesson_lines.append(f"- {line}")

        # Keep all lessons - no artificial truncation of important insights

        return '\n'.join(lesson_lines) if lesson_lines else ""

    def _extract_rtl_from_response(self, response: str) -> str:
        """Extract RTL code from LLM response."""
        import re

        # Look for code blocks
        patterns = [
            r'```systemverilog\s*\n(.*?)\n```',
            r'```verilog\s*\n(.*?)\n```',
            r'```sv\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
            if matches:
                code = matches[0].strip()
                if self._is_valid_rtl(code):
                    return code

        # Fallback: try to extract module...endmodule
        lines = response.split('\n')
        module_start = None
        module_end = None

        for i, line in enumerate(lines):
            if re.match(r'\s*module\s+\w+', line):
                module_start = i
            if module_start is not None and re.match(r'\s*endmodule', line):
                module_end = i
                break

        if module_start is not None and module_end is not None:
            code = '\n'.join(lines[module_start:module_end+1])
            if self._is_valid_rtl(code):
                return code

        self.logger.warning("Could not extract valid RTL from fresh generation response")
        return response.strip()

    def _is_valid_rtl(self, code: str) -> bool:
        """Basic RTL validation."""
        import re

        return (re.search(r'\bmodule\s+\w+', code) and
                'endmodule' in code and
                len(code.strip()) > 50)

    def _extract_error_summary(self, error_report: str) -> str:
        """Extract brief error summary for pattern analysis."""

        if not error_report:
            return "No errors"

        # Look for common error patterns
        lines = error_report.split('\n')

        # Find assertion failures, compilation errors, etc.
        for line in lines:
            line = line.strip()
            if 'assert' in line.lower() and ('fail' in line.lower() or 'error' in line.lower()):
                return f"Assertion failure: {line}"
            if 'error' in line.lower() and any(word in line.lower() for word in ['syntax', 'compile', 'parse']):
                return f"Compilation error: {line}"
            if 'expected' in line.lower() and 'got' in line.lower():
                return f"Value mismatch: {line}"

        # Fallback: scan entire report for error information
        for line in lines:  # Check all lines, not just first 10
            if line.strip() and any(word in line.lower() for word in ['error', 'fail', 'assert']):
                return f"Error: {line.strip()}"

        return "Test failure"

    def get_restart_summary(self) -> Dict[str, Any]:
        """Get summary of restart activity."""
        return {
            "total_restarts": self.fresh_start_count,
            "last_restart_iteration": self.last_fresh_start_iteration,
            "total_iterations_analyzed": len(self.iteration_history)
        }
