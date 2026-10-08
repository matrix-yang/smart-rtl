# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive Reasoning Coordinator - orchestrates structured RTL debugging.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from .context_builder import ContextBuilder, DebugContext
from .iteration_memory import IterationMemory, FailedAttempt
from .code_analyzer import CodeAnalyzer
from .utils.llm_client import LLMClient

class ComprehensiveReasoningCoordinator:
    """
    Coordinates RTL debugging using a comprehensive reasoning approach.

    This orchestration layer combines:
    1. Builds deep context understanding
    2. Learns from previous attempts
    3. Analyzes code structure deeply
    4. Makes targeted, reasoned fixes
    """

    def __init__(self, model_name: str, logger, api_key: Optional[str] = None,
                 max_retries: int = 3, initial_backoff: float = 5.0):
        """Initialize the comprehensive reasoning coordinator."""

        self.model_name = model_name
        self.logger = logger

        # Initialize all structured debugging components.
        self.context_builder = ContextBuilder(logger)
        self.iteration_memory = IterationMemory(logger)
        self.code_analyzer = CodeAnalyzer(logger)

        # Initialize LLM client for reasoning
        self.client = LLMClient(
            model=model_name,
            api_key=api_key,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            logger=logger
        )

        self.logger.info("Initialized comprehensive reasoning coordinator")

    def debug_rtl_iteratively(self, specification: str, current_code: str,
                             error_report: str, iteration: int, harness_dir: Optional[Path] = None) -> Tuple[str, str, Dict[str, Any]]:
        """
        Main entry point - debug RTL using a comprehensive reasoning approach.

        This follows a structured debugging flow:
        1. Build complete context understanding
        2. Check what we've learned from previous attempts
        3. Analyze current code structure deeply
        4. Reason about the fix using programming fundamentals
        5. Make targeted minimal change
        """

        self.logger.info(f"Starting comprehensive reasoning for iteration {iteration}")

        # Step 1: Build comprehensive debugging context (including testbench analysis)
        debug_context = self.context_builder.build_debug_context(
            specification=specification,
            current_code=current_code,
            error_report=error_report,
            previous_attempts=[attempt.__dict__ for attempt in self.iteration_memory.failed_attempts],
            harness_dir=harness_dir  # NEW: Pass harness dir for testbench analysis
        )

        # Step 2: Analyze current code structure deeply
        code_analysis = self.code_analyzer.analyze_rtl_code(current_code)

        # Step 2.5: Get fix suggestions based on code analysis + error context
        fix_suggestions = self.code_analyzer.get_fix_suggestions(code_analysis, debug_context.error_analysis)

        # Step 3: Get learning insights from previous attempts
        learning_insights = self.iteration_memory.get_learning_insights()

        # Step 4: Generate fix using comprehensive reasoning
        corrected_rtl, reasoning, success = self._generate_reasoned_fix(
            debug_context, code_analysis, learning_insights, fix_suggestions, error_report
        )

        # Step 5: Prepare debug metadata
        debug_metadata = {
            "strategy_used": "comprehensive_reasoning",
            "iteration": iteration,
            "context_analysis": {
                "test_intent": debug_context.test_intent,
                "error_type": debug_context.error_analysis["error_type"],
                "test_phase": debug_context.error_analysis["test_phase"]
            },
            "code_analysis_summary": {
                "module_name": code_analysis["module_interface"]["module_name"],
                "potential_issues": code_analysis["potential_issues"]
            },
            "learning_applied": learning_insights,
            "reasoning": reasoning
        }

        # Create the prompt for logging/debugging purposes
        prompt = self._create_comprehensive_prompt(debug_context, code_analysis, learning_insights, fix_suggestions, error_report)

        if success:
            self.logger.info("Comprehensive reasoning produced a fix")
            return corrected_rtl, prompt, "", debug_metadata
        else:
            self.logger.warning("Comprehensive reasoning failed to generate fix")
            raise RuntimeError("Failed to generate corrected RTL code")

    def _generate_reasoned_fix(self, debug_context: DebugContext, code_analysis: Dict[str, Any],
                              learning_insights: Dict[str, Any], fix_suggestions: List[str], error_report: str) -> Tuple[str, str, bool]:
        """Generate fix using comprehensive reasoning."""

        # Create comprehensive prompt
        prompt = self._create_comprehensive_prompt(debug_context, code_analysis, learning_insights, fix_suggestions, error_report)

        try:
            # Get LLM response
            response = self.client.call_llm(prompt)
            if not response:
                return "", "No response from LLM", False

            # Extract corrected RTL
            corrected_rtl = self._extract_rtl_code(response)

            # Extract reasoning from response
            reasoning = self._extract_reasoning(response)

            return corrected_rtl, reasoning, True

        except Exception as e:
            self.logger.error(f"Error generating reasoned fix: {e}")
            return "", f"Error: {str(e)}", False

    def _create_comprehensive_prompt(self, debug_context: DebugContext,
                                   code_analysis: Dict[str, Any],
                                   learning_insights: Dict[str, Any],
                                   fix_suggestions: List[str],
                                   error_report: str) -> str:
        """Create a comprehensive debugging prompt."""

        # Get memory context
        memory_context = self.iteration_memory.get_context_for_llm()

        # Get testbench context if available
        testbench_context = ""
        if debug_context.testbench_insight:
            testbench_context = self.context_builder.testbench_analyzer.get_test_context_for_error(
                debug_context.testbench_insight,
                error_report  # Pass the actual error_report parameter
            )

        prompt = f"""You are an expert RTL designer debugging code using pure reasoning and programming fundamentals.

## UNDERSTAND AND FIX THE PROBLEM

### INFORMATION AVAILABLE

**Error Report:**
```
{error_report}
```

**Current RTL Code:**
```systemverilog
{debug_context.current_code}
```

**Original Specification:**
```
{debug_context.specification}
```

{testbench_context if testbench_context else ""}

**Previous Attempts Context:**
{memory_context}

### DEBUG USING PURE REASONING

Read the error carefully and think through this systematically:

1. **What went wrong?** - Look at the assertion failure and understand the mismatch
2. **Why did this happen?** - Reason about what in the code could cause this
3. **How to fix it?** - Make the minimal change to address the root cause

Use general programming/debugging principles:
- Understand the data flow
- Check initialization and reset behavior
- Verify calculations and logic
- Consider timing and sequencing
- Look for edge cases

## YOUR SOLUTION

**Analysis:**
[Explain what you think went wrong and why]

**Fix:**
[Explain your solution approach]

```systemverilog
[Your corrected RTL code]
```
"""
        return prompt

    def _extract_rtl_code(self, response: str) -> str:
        """Extract SystemVerilog code from LLM response."""
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

        # Fallback: look for module...endmodule
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

        self.logger.warning("Could not extract valid RTL code from response")
        return response.strip()

    def _is_valid_rtl(self, code: str) -> bool:
        """Validate extracted RTL code."""
        if not re.search(r'\bmodule\s+\w+', code):
            return False
        if 'endmodule' not in code:
            return False
        if len(code.strip()) < 50:  # Too short to be meaningful
            return False
        return True

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning explanation from response."""
        lines = response.split('\n')
        reasoning_lines = []

        in_analysis_section = False
        for line in lines:
            if "My Analysis:" in line or "Analysis:" in line:
                in_analysis_section = True
                continue
            elif line.startswith("**") and in_analysis_section:
                break
            elif in_analysis_section:
                reasoning_lines.append(line.strip())

        if reasoning_lines:
            return ' '.join(reasoning_lines).strip()

        # Fallback: just return first few lines
        return ' '.join(lines).strip()

    def record_attempt_result(self, iteration: int, attempted_fix: str, reasoning: str,
                             code_before: str, code_after: str, success: bool, error_msg: str = ""):
        """Record the result of an attempt for learning."""

        if success:
            self.iteration_memory.record_successful_attempt(attempted_fix, reasoning)
            self.logger.info(f"Recorded successful attempt {iteration}")
        else:
            self.iteration_memory.record_failed_attempt(
                iteration=iteration,
                error_type="test_failure",
                attempted_fix=attempted_fix,
                reasoning=reasoning,
                code_before=code_before,
                code_after=code_after,
                result=error_msg or "test_failed"
            )
            self.logger.info(f"Recorded failed attempt {iteration}")

    def should_avoid_approach(self, potential_fix: str, reasoning: str) -> bool:
        """Check if this approach should be avoided based on previous failures."""
        return self.iteration_memory.should_avoid_approach(potential_fix, reasoning)
