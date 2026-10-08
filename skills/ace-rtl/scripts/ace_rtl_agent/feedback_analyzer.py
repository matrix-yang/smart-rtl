# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Feedback Analyzer - Analyzes test failures and generates improvement feedback for RTL code using LLM
"""

import os
import re
from typing import Optional, Dict, List

from .utils.llm_client import LLMClient


class FeedbackAnalyzer:
    """
    Handles RTL error correction using LLM APIs.

    This class is simplified to work with cvdp_benchmark for generating corrected RTL code
    based on error reports from test failures.
    """

    def __init__(self, model_name: str, logger, api_key: Optional[str] = None,
                 max_retries: int = 3, initial_backoff: float = 5.0):
        """
        Initialize the Feedback Analyzer.

        Args:
            model_name: Name of the LLM model to use for analysis
            logger: Logger instance
            api_key: Optional LLM API key (will use environment variable if not provided)
            max_retries: Maximum number of retries for API calls (default: 3)
            initial_backoff: Initial backoff time in seconds (default: 5.0)
        """
        self.model_name = model_name
        self.logger = logger

        # Initialize robust LLM client
        self.client = LLMClient(
            model=model_name,
            api_key=api_key,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            logger=logger
        )

    def correct_code(self, specification: str, buggy_code: str, error_report: str) -> Optional[str]:
        """
        Generate corrected RTL code based on error report.

        Args:
            specification: Original problem specification
            buggy_code: The RTL code that failed
            error_report: Complete error report from test failure

        Returns:
            Corrected RTL code or None if correction failed
        """
        self.logger.info("Generating corrected RTL code based on error report")

        try:
            response = self.client.correct_rtl_code(specification, buggy_code, error_report)
            return response

        except Exception as e:
            self.logger.error(f"Error during code correction: {str(e)}")
            return None
