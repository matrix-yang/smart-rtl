# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Focused RTL Debugger - Minimal, clear prompts with actionable guidance.
"""

from typing import List, Optional

from .focused_debugger_types import FixAttempt
from .focused_task_detection import FocusedTaskDetectionMixin
from .focused_prompting import FocusedPromptingMixin
from .focused_history import FocusedHistoryMixin
from .focused_error_analysis import FocusedErrorAnalysisMixin
from .focused_guidance import FocusedGuidanceMixin
from .utils.llm_client import LLMClient
from .utils.ace_model_client import AceRTLClient


class FocusedDebugger(
    FocusedTaskDetectionMixin,
    FocusedPromptingMixin,
    FocusedHistoryMixin,
    FocusedErrorAnalysisMixin,
    FocusedGuidanceMixin,
):
    """Minimal debugger that tracks history and provides focused prompts."""

    def __init__(self, logger, model_name: str = "nvidia/nemotron-3-ultra-550b-a55b",
                 ace_rtl_stages: set = None, ace_rtl_temperature: float = 0.2,
                 api_key: Optional[str] = None, max_retries: int = 3,
                 initial_backoff: float = 5.0):
        self.logger = logger
        self.model_name = model_name
        self.ace_rtl_stages = ace_rtl_stages if ace_rtl_stages is not None else set()
        self.ace_rtl_temperature = ace_rtl_temperature
        self.fix_history: List[FixAttempt] = []

        # LLM client for stage 1 (error analysis & fix guidance)
        # Use ACE-RTL if stage 1 is in ace_rtl_stages, otherwise use LLM
        if 1 in self.ace_rtl_stages:
            self.logger.info("Stage 1 (error analysis & fix guidance): Using ACE-RTL API client")
            self.llm_client = AceRTLClient(
                logger=logger,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                temperature=self.ace_rtl_temperature
            )
        else:
            self.logger.debug("Stage 1 (error analysis & fix guidance): Using LLM model")
            self.llm_client = LLMClient(
                model=model_name,
                api_key=api_key,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                logger=logger
            )

        self.current_focused_error: Optional[str] = None  # Track which error we're working on
        self.error_solved: bool = False  # Track if current error is solved

        # Initialize RTL rule advisor for architectural guidance
        from .rtl_rule_advisor import RTLRuleAdvisor
        self.rtl_rule_advisor = RTLRuleAdvisor(model_name=model_name)
        self.rtl_rules_retrieved = False

    def clear_history(self):
        """Clear history - used when doing a fresh start/restart."""
        self.fix_history = []
        self.current_focused_error = None
        self.error_solved = False
        self.rtl_rule_advisor.reset_session()
        self.rtl_rules_retrieved = False
        self.logger.info("Cleared focused debugger history for fresh start")
