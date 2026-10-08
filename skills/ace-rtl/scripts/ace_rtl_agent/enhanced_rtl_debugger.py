# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Enhanced RTL Debugger - Integrates all the new debugging components for systematic RTL debugging
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .regression_prevention import RegressionPreventionSystem
from .test_sequence_analyzer import TestSequenceAnalyzer
from .rtl_consistency_checker import RTLConsistencyChecker
from .progressive_fix_coordinator import ProgressiveFixCoordinator
from .utils.llm_client import LLMClient


class EnhancedRTLDebugger:
    """
    Enhanced RTL debugger that combines all the new debugging capabilities:
    - Regression prevention
    - Test sequence analysis
    - RTL consistency checking
    - Progressive fix coordination

    This replaces the ad-hoc debugging approach with a systematic one.
    """

    def __init__(self, model_name: str, logger, api_key: Optional[str] = None,
                 max_retries: int = 3, initial_backoff: float = 5.0):
        """Initialize the enhanced RTL debugger."""

        self.model_name = model_name
        self.logger = logger

        # Initialize all debugging components
        self.regression_prevention = RegressionPreventionSystem(logger)
        self.test_analyzer = TestSequenceAnalyzer(logger)
        self.consistency_checker = RTLConsistencyChecker(logger)
        self.fix_coordinator = ProgressiveFixCoordinator(logger)

        # Initialize LLM client
        self.client = LLMClient(
            model=model_name,
            api_key=api_key,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            logger=logger
        )

        self.logger.info("Initialized EnhancedRTLDebugger with all debugging components")

    def debug_rtl_systematically(self, specification: str, current_code: str,
                                error_report: str, iteration: int) -> Tuple[str, str, Dict[str, Any]]:
        """
        Main debugging method that uses all components systematically.

        This is the entry point that replaces the old debugging approach.
        """
        self.logger.info(f"Starting systematic RTL debugging for iteration {iteration}")

        # Step 1: Check for regressions
        regression_report = self.regression_prevention.check_for_regressions(error_report, iteration)

        if regression_report.newly_failing_assertions:
            self.logger.warning(f"REGRESSION DETECTED: {len(regression_report.newly_failing_assertions)} "
                              f"previously passing assertions now fail")

        # Step 2: Analyze test sequence and timing
        test_analysis = self.test_analyzer.analyze_test_sequence(error_report, specification)

        # Step 3: Check RTL consistency
        consistency_issues = self.consistency_checker.check_rtl_consistency(current_code)

        # Step 4: Analyze required fixes
        required_fixes = self.fix_coordinator.analyze_new_fix_requirements(
            error_report, current_code, iteration
        )

        # Step 5: Generate debugging context
        debugging_context = self._generate_comprehensive_debugging_context(
            regression_report, test_analysis, consistency_issues, required_fixes, iteration
        )

        # Step 6: Generate the enhanced debugging prompt
        prompt = self._create_enhanced_debugging_prompt(
            specification, current_code, error_report, debugging_context
        )

        # Step 7: Get LLM response
        try:
            response = self.client.chat_completion([
                {"role": "system", "content": "You are an expert RTL debugger using systematic debugging methods."},
                {"role": "user", "content": prompt}
            ])

            corrected_code = self._extract_rtl_code(response)

            # Step 8: Validate the corrected code
            validation_result = self._validate_corrected_code(
                corrected_code, consistency_issues, required_fixes
            )

            # Step 9: Update tracking systems
            self._update_tracking_systems(error_report, iteration, required_fixes)

            metadata = {
                'regression_report': regression_report,
                'test_analysis': test_analysis,
                'consistency_issues': consistency_issues,
                'required_fixes': required_fixes,
                'validation_result': validation_result
            }

            return corrected_code, response, metadata

        except Exception as e:
            self.logger.error(f"Enhanced RTL debugging failed: {str(e)}")
            # Fallback to basic approach
            return self._fallback_debugging(specification, current_code, error_report)

    def record_iteration_results(self, error_report: str, iteration: int,
                               test_passed: bool, applied_fixes: List[str] = None):
        """
        Record the results of an iteration for tracking and learning.
        """
        # Update regression prevention system
        newly_passing, still_failing = self.regression_prevention.analyze_test_results(
            error_report, iteration
        )

        # Record successful fixes
        if applied_fixes and test_passed:
            for fix_description in applied_fixes:
                # Try to categorize the fix
                category = self._categorize_fix(fix_description)
                self.fix_coordinator.record_successful_fix(
                    category, fix_description, "", f"Iteration {iteration} improvement", iteration
                )

        self.logger.info(f"Recorded iteration {iteration} results: "
                        f"{len(newly_passing)} newly passing, {len(still_failing)} still failing")

    def _generate_comprehensive_debugging_context(self, regression_report, test_analysis,
                                                consistency_issues, required_fixes, iteration) -> str:
        """Generate comprehensive debugging context from all analysis components."""

        context = ["# COMPREHENSIVE DEBUGGING CONTEXT", ""]

        # Regression prevention context
        cumulative_requirements = self.regression_prevention.generate_cumulative_requirements()
        context.append(cumulative_requirements)
        context.append("")

        # Regression warnings
        if regression_report.newly_failing_assertions:
            context.append("## REGRESSION ALERT")
            context.append(f"Severity: {regression_report.regression_severity}")
            context.append(f"Recommendation: {regression_report.recommendation}")
            context.append("")

        # Test sequence analysis
        timing_guidance = self.test_analyzer.generate_timing_guidance(test_analysis)
        context.append(timing_guidance)
        context.append("")

        # Consistency issues
        if consistency_issues:
            consistency_fixes = self.consistency_checker.suggest_fixes(consistency_issues)
            context.append(consistency_fixes)
            context.append("")

        # Progressive fix strategy
        if required_fixes:
            fix_strategy = self.fix_coordinator.generate_progressive_fix_strategy(
                required_fixes, ""  # current_code not needed for strategy
            )
            context.append(fix_strategy)
            context.append("")

        return "\n".join(context)

    def _create_enhanced_debugging_prompt(self, specification: str, current_code: str,
                                        error_report: str, debugging_context: str) -> str:
        """Create the enhanced debugging prompt with all context."""

        # Load the enhanced debugging template
        prompt_template = self._load_enhanced_prompt_template()

        # Fill in the template
        prompt = prompt_template.format(
            cumulative_requirements=self.regression_prevention.generate_cumulative_requirements(),
            test_sequence_analysis=debugging_context,
            progressive_fix_strategy=self.fix_coordinator.generate_progressive_fix_strategy([], current_code),
            specification=specification,
            current_code=current_code,
            error_report=error_report,
            debugging_context=debugging_context
        )

        return prompt

    def _load_enhanced_prompt_template(self) -> str:
        """Load the enhanced debugging prompt template."""
        try:
            # Try to load the new enhanced template
            template_path = Path(__file__).parent.parent / "prompts" / "enhanced_rtl_debugging.md"
            if template_path.exists():
                return template_path.read_text()
        except Exception as e:
            self.logger.warning(f"Could not load enhanced template: {e}")

        # Fallback template
        return """
# Enhanced RTL Debugging

## Current Requirements (Must Maintain):
{cumulative_requirements}

## Test Analysis:
{test_sequence_analysis}

## Fix Strategy:
{progressive_fix_strategy}

## Current Error:
{error_report}

## Current Code:
{current_code}

## Specification:
{specification}

Fix the RTL code while maintaining all previously working functionality.
"""

    def _extract_rtl_code(self, response: str) -> str:
        """Extract RTL code from the LLM response."""
        # Look for SystemVerilog code blocks
        import re

        # Try to find code in ```systemverilog or ```verilog blocks
        code_patterns = [
            r'```(?:system)?verilog\n(.*?)\n```',
            r'```\n(module.*?endmodule)\n```',
            r'```\n(.*?)\n```'
        ]

        for pattern in code_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no code blocks found, look for module...endmodule
        module_match = re.search(r'(module\s+.*?endmodule)', response, re.DOTALL | re.IGNORECASE)
        if module_match:
            return module_match.group(1).strip()

        # Fallback: return the response as-is
        return response.strip()

    def _validate_corrected_code(self, corrected_code: str, consistency_issues: List,
                               required_fixes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the corrected code against known issues."""
        validation_result = {
            'syntax_valid': True,
            'consistency_improved': False,
            'fixes_addressed': [],
            'new_issues': []
        }

        # Check if consistency issues were addressed
        new_consistency_issues = self.consistency_checker.check_rtl_consistency(corrected_code)
        high_priority_issues = [issue for issue in consistency_issues if issue.severity == "HIGH"]
        new_high_priority = [issue for issue in new_consistency_issues if issue.severity == "HIGH"]

        if len(new_high_priority) < len(high_priority_issues):
            validation_result['consistency_improved'] = True

        # Check basic syntax
        if 'module' in corrected_code and 'endmodule' in corrected_code:
            validation_result['syntax_valid'] = True
        else:
            validation_result['syntax_valid'] = False
            validation_result['new_issues'].append("Missing module structure")

        return validation_result

    def _update_tracking_systems(self, error_report: str, iteration: int, required_fixes: List):
        """Update all tracking systems with the current iteration results."""

        # This would be called after the test results are available
        # For now, just log the update
        self.logger.info(f"Updated tracking systems for iteration {iteration}")

    def _categorize_fix(self, fix_description: str) -> Any:
        """Categorize a fix description into a FixCategory."""
        from .progressive_fix_coordinator import FixCategory

        desc_lower = fix_description.lower()

        if 'reset' in desc_lower:
            return FixCategory.RESET_LOGIC
        elif 'handshak' in desc_lower or 'valid' in desc_lower or 'ready' in desc_lower:
            return FixCategory.HANDSHAKING
        elif 'data' in desc_lower or 'format' in desc_lower:
            return FixCategory.DATA_FORMATTING
        elif 'timing' in desc_lower or 'clock' in desc_lower:
            return FixCategory.TIMING_CONTROL
        elif 'syntax' in desc_lower or 'compil' in desc_lower:
            return FixCategory.COMPILATION_FIX
        else:
            return FixCategory.SIGNAL_DECLARATIONS

    def _fallback_debugging(self, specification: str, current_code: str,
                          error_report: str) -> Tuple[str, str, Dict[str, Any]]:
        """Fallback debugging approach if enhanced debugging fails."""
        self.logger.warning("Using fallback debugging approach")

        simple_prompt = f"""
Fix this RTL code:

Specification: {specification}

Current Code:
{current_code}

Error:
{error_report}

Provide the corrected RTL code.
"""

        try:
            response = self.client.chat_completion([
                {"role": "user", "content": simple_prompt}
            ])

            corrected_code = self._extract_rtl_code(response)
            return corrected_code, response, {"fallback": True}

        except Exception as e:
            self.logger.error(f"Fallback debugging also failed: {str(e)}")
            return current_code, f"Error: {str(e)}", {"error": True}
