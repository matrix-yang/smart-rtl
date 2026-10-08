# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal Change Coordinator - Orchestrates debugging with focus on minimal targeted changes
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .primary_failure_extractor import PrimaryFailureExtractor, PrimaryFailure
from .specification_example_parser import SpecificationExampleParser, ConcreteExample
from .progressive_debugging_strategy import ProgressiveDebuggingStrategy, DebuggingApproach
from .utils.llm_client import LLMClient

class MinimalChangeCoordinator:
    """
    Coordinates RTL debugging with focus on minimal, targeted changes.

    This implements key minimal-change debugging patterns:
    1. Focus on ONE assertion failure per iteration
    2. Use specification examples as ground truth
    3. Start simple, escalate complexity progressively
    4. Make minimal targeted changes
    """

    def __init__(self, model_name: str, logger, api_key: Optional[str] = None,
                 max_retries: int = 3, initial_backoff: float = 5.0, llm_context_limit: int = 300000):
        """Initialize minimal change coordinator."""

        self.model_name = model_name
        self.logger = logger

        # Initialize components
        self.failure_extractor = PrimaryFailureExtractor(logger, model_name, llm_context_limit)
        self.example_parser = SpecificationExampleParser(logger)
        self.strategy_selector = ProgressiveDebuggingStrategy(logger)

        # Initialize LLM client
        self.client = LLMClient(
            model=model_name,
            api_key=api_key,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            logger=logger
        )

        self.logger.info("Initialized MinimalChangeCoordinator")

    def debug_rtl_with_minimal_changes(self, specification: str, current_code: str,
                                     error_report: str, iteration: int,
                                     failure_analyzer=None) -> Tuple[str, str, Dict[str, Any]]:
        """
        Debug RTL using minimal change approach.

        This follows the successful debugging pattern:
        1. Extract primary failure
        2. Get specification examples
        3. Select appropriate debugging approach
        4. Generate targeted fix
        """

        self.logger.info(f"Starting minimal change debugging for iteration {iteration}")

        # Step 1: Extract primary assertion failure
        primary_failure = self.failure_extractor.extract_primary_failure(error_report)

        # Step 2: Extract specification examples for ground truth
        spec_examples = self.example_parser.extract_concrete_examples(specification)

        # Step 3: Select debugging approach based on iteration
        approach = self.strategy_selector.select_debugging_approach(iteration)

        # Step 4: Generate enhanced context for better understanding
        enhanced_context = ""
        if failure_analyzer and primary_failure:
            enhanced_context = failure_analyzer.create_enhanced_debugging_context(
                primary_failure, current_code, error_report, specification
            )

        # Step 5: Generate appropriate prompt with enhanced context
        if approach == DebuggingApproach.FOCUSED_ASSERTION and primary_failure:
            prompt = self.strategy_selector.create_approach_prompt(
                approach, specification, current_code, error_report,
                primary_failure=primary_failure, spec_examples=spec_examples,
                enhanced_context=enhanced_context
            )
        else:
            prompt = self.strategy_selector.create_approach_prompt(
                approach, specification, current_code, error_report,
                enhanced_context=enhanced_context
            )

        # Step 5: Get LLM response
        try:
            response = self.client.call_llm(prompt)
            if not response:
                raise ValueError("No response from LLM")

            # Step 6: Extract and validate LLM reasoning
            reasoning_analysis = self.strategy_selector.extract_llm_reasoning(response)
            self.logger.info(f"LLM reasoning quality: {reasoning_analysis['reasoning_quality']}")
            if reasoning_analysis['followed_framework']:
                self.logger.info(f"LLM completed {len(reasoning_analysis['completed_steps'])} reasoning steps")
            else:
                self.logger.warning("LLM did not follow structured reasoning framework")

            # Extract corrected RTL
            corrected_rtl = self._extract_rtl_code(response)

            # Prepare debug metadata with actual LLM reasoning
            debug_metadata = {
                "strategy_used": "minimal_change_coordination",
                "debugging_approach": approach.value,
                "iteration": iteration,
                "primary_failure": {
                    "expected": primary_failure.expected_value if primary_failure else "unknown",
                    "actual": primary_failure.actual_value if primary_failure else "unknown",
                    "context": primary_failure.context_lines if primary_failure else "unknown"
                } if primary_failure else None,
                "spec_examples_found": len(spec_examples),
                "reasoning": f"Applied {approach.value} debugging with minimal change focus",
                "llm_reasoning_analysis": reasoning_analysis,
                "reasoning_quality_check": {
                    "followed_framework": reasoning_analysis['followed_framework'],
                    "steps_completed": len(reasoning_analysis['completed_steps']),
                    "questions_answered": len(reasoning_analysis['answered_questions']),
                    "root_cause_identified": bool(reasoning_analysis['identified_root_cause']),
                    "fix_strategy_proposed": bool(reasoning_analysis['proposed_fix_strategy'])
                }
            }

            self.logger.info(f"Minimal change debugging completed: {approach.value}")
            return corrected_rtl, prompt, "", debug_metadata  # Empty response for minimal change coordinator

        except Exception as e:
            self.logger.error(f"Minimal change debugging failed: {e}")
            raise

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

        self.logger.warning("Could not extract valid RTL code, returning response")
        return response.strip()

    def _is_valid_rtl(self, code: str) -> bool:
        """Basic RTL validation."""
        return (re.search(r'\bmodule\s+\w+', code) and
                'endmodule' in code and
                len(code.strip()) > 50)
