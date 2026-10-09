# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Failure Context Analyzer - Helps LLM deeply understand test failures and timing issues
"""

from typing import Dict, List, Optional, Any, Tuple
from .utils.llm_client import LLMClient


class FailureContextAnalyzer:
    """
    Analyzes test failures to provide deeper context and insights to the LLM.

    This helps the LLM understand not just WHAT failed, but WHY it failed,
    including timing issues, test sequence context, and implementation patterns.
    """

    def __init__(self, model_name: str, logger, api_key: Optional[str] = None,
                 max_retries: int = 3, initial_backoff: float = 5.0):
        """Initialize failure context analyzer."""

        self.model_name = model_name
        self.logger = logger

        # Initialize LLM client for deep failure analysis
        self.client = LLMClient(
            model=model_name,
            api_key=api_key,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            logger=logger
        )

        self.logger.info("Initialized FailureContextAnalyzer")

    def analyze_failure_deeply(self, rtl_code: str, error_report: str,
                              specification: str) -> str:
        """
        Use LLM to deeply analyze why the test failed and provide insights.

        Args:
            rtl_code: The RTL code that failed
            error_report: The test error report
            specification: The original specification

        Returns:
            Deep analysis context for debugging
        """
        try:
            prompt = self._create_failure_analysis_prompt(rtl_code, error_report, specification)

            response = self.client.call_llm(prompt)
            if response and response.strip():
                self.logger.info("Generated deep failure analysis context")
                return response.strip()
            else:
                return ""

        except Exception as e:
            self.logger.error(f"Deep failure analysis failed: {e}")
            return ""

    def _create_failure_analysis_prompt(self, rtl_code: str, error_report: str,
                                      specification: str) -> str:
        """Create prompt for deep failure analysis."""

        prompt = f"""Analyze this RTL test failure deeply to understand the root cause.

**Specification**:
{specification}

**Failed RTL Code**:
```systemverilog
{rtl_code}
```

**Test Error Report**:
```
{error_report}
```

**Your Deep Analysis Task**:
1. **Understand the Test Sequence**: What is the test doing step by step?
2. **Identify the Failure Point**: At what exact moment does the failure occur?
3. **Analyze the Implementation**: How does your RTL code handle that scenario?
4. **Find the Mismatch**: Why does the test expect one thing but your code produces another?
5. **Understand the Timing**: Is this a combinational vs sequential logic issue?

**Provide Insight**:
- What is the fundamental misunderstanding between specification and implementation?
- What timing or behavioral assumption is wrong?
- What specific aspect of the design needs to be approached differently?

**Deep Analysis**:"""

        return prompt

    def create_enhanced_debugging_context(self, primary_failure: Any, rtl_code: str,
                                        error_report: str, specification: str) -> str:
        """
        Create enhanced debugging context with deep failure understanding.

        Args:
            primary_failure: Primary failure information
            rtl_code: Current RTL code
            error_report: Test error report
            specification: Original specification

        Returns:
            Enhanced context for debugging prompt
        """
        # Get deep analysis from LLM
        deep_analysis = self.analyze_failure_deeply(rtl_code, error_report, specification)

        if not deep_analysis:
            return ""

        enhanced_context = f"""
**DEEP FAILURE ANALYSIS**:
{deep_analysis}

**CRITICAL INSIGHT**: Use this analysis to understand the fundamental issue, not just the surface error."""

        return enhanced_context
