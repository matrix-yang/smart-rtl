# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main ACE-RTL Agent - Orchestrates iterative RTL generation with feedback loops
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from .agent_types import AgentConfig, IterationResult
from .agent_processing import AgentProcessingMixin
from .agent_loop import AgentLoopMixin
from .agent_reporting import AgentReportingMixin
from .rtl_generator import RTLGenerator
from .feedback_analyzer import FeedbackAnalyzer
from .datapoint_processor import DatapointProcessor
from .utils.logging_utils import setup_logger
from .utils.cvdp_integration import CVDPIntegration
from .utils.verilogeval_integration import VerilogEvalIntegration
from .utils.chipbench_integration import ChipBenchIntegration
from .utils.chipverilog_integration import ChipVerilogIntegration


class AceRTLAgent(AgentProcessingMixin, AgentLoopMixin, AgentReportingMixin):
    """
    Main agent class that orchestrates iterative RTL generation with feedback loops.

    This agent takes a benchmark problem (CVDP or VerilogEval) and iteratively generates
    and improves RTL code until it passes the testbench or reaches max iterations.
    """

    def __init__(self, config: AgentConfig, cvdp_benchmark_path: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the ACE-RTL Agent.

        Args:
            config: Agent configuration
            cvdp_benchmark_path: Path to the CVDP benchmark directory (required for CVDP mode)
            api_key: Optional LLM API key (will use environment variable if not provided)
        """
        self.config = config
        self.benchmark_mode = config.benchmark_mode
        self.cvdp_benchmark_path = Path(cvdp_benchmark_path) if cvdp_benchmark_path else None
        self.api_key = api_key  # Store for creating thread-local instances
        self.logger = setup_logger("ace_rtl_agent", debug=config.enable_debug)

        # Set retry configuration early
        os.environ['ACE_RTL_LLM_MAX_RETRIES'] = str(config.max_retries)
        os.environ['ACE_RTL_LLM_RETRY_BACKOFF'] = str(config.initial_backoff)
        os.environ['LLM_RETRY_COUNT'] = str(config.max_retries)

        # Initialize core components with LLM API key
        self.rtl_generator = RTLGenerator(
            config.model_name,
            self.logger,
            api_key,
            max_retries=config.max_retries,
            initial_backoff=config.initial_backoff,
            use_systematic_debugging=config.use_systematic_debugging,
            llm_context_limit=config.llm_context_limit,
            ace_rtl_stages=config.ace_rtl_stages,
            ace_rtl_temperature=config.ace_rtl_temperature,
            disable_testbench_access=config.disable_testbench_access
        )
        self.feedback_analyzer = FeedbackAnalyzer(
            config.model_name,
            self.logger,
            api_key,
            max_retries=config.max_retries,
            initial_backoff=config.initial_backoff
        )
        self.datapoint_processor = DatapointProcessor(self.logger)

        # Initialize benchmark integration based on mode
        if self.benchmark_mode == "verilogeval":
            self.logger.info(f"Initializing VerilogEval mode")
            if not config.verilogeval_path:
                raise ValueError("verilogeval_path is required for VerilogEval mode")
            self.benchmark_integration = VerilogEvalIntegration(
                config.verilogeval_path,
                self.logger,
                max_retries=config.max_retries,
                initial_backoff=config.initial_backoff
            )
            self.cvdp_integration = None
        elif self.benchmark_mode == "chipbench":
            self.logger.info(f"Initializing ChipBench mode")
            if not config.chipbench_path:
                raise ValueError("chipbench_path is required for ChipBench mode")
            self.benchmark_integration = ChipBenchIntegration(
                config.chipbench_path,
                self.logger,
                max_retries=config.max_retries,
                initial_backoff=config.initial_backoff
            )
            self.cvdp_integration = None
        elif self.benchmark_mode == "chipverilog":
            self.logger.info("Initializing ChipVerilog mode")
            if not config.chipverilog_path:
                raise ValueError("chipverilog_path is required for ChipVerilog mode")
            self.benchmark_integration = ChipVerilogIntegration(
                config.chipverilog_path,
                self.logger,
                max_retries=config.max_retries,
                initial_backoff=config.initial_backoff,
                evaluator_depth=config.chipverilog_eval_depth,
                evaluator_timeout=config.chipverilog_eval_timeout,
            )
            self.cvdp_integration = None
        else:  # cvdp mode
            self.logger.info(f"Initializing CVDP mode")
            if not cvdp_benchmark_path:
                raise ValueError("cvdp_benchmark_path is required for CVDP mode")
            self.benchmark_integration = CVDPIntegration(
                cvdp_benchmark_path,
                self.logger,
                max_retries=config.max_retries,
                initial_backoff=config.initial_backoff,
                ace_rtl_stages=config.ace_rtl_stages,
                ace_rtl_temperature=config.ace_rtl_temperature
            )
            self.cvdp_integration = self.benchmark_integration  # For backward compatibility

        # Results storage
        self.iteration_results: List[IterationResult] = []
        self.current_datapoint: Optional[Dict] = None

        self.logger.info(f"Initialized ACE-RTL Agent with max_iterations={config.max_iterations} using model: {config.model_name}")
        self.logger.info(f"Benchmark mode: {self.benchmark_mode}")
        if config.ace_rtl_stages:
            stages_str = ', '.join(f"Stage {s}" for s in sorted(config.ace_rtl_stages))
            self.logger.info(f"ACE-RTL model enabled for: {stages_str} (0=initial gen, 1=error analysis, 2=code correction)")
        else:
            self.logger.info(f"ACE-RTL model: disabled (using LLM for all stages)")

    def _create_process_components(self):
        """Create process-local instances of components for parallel execution."""
        # Create fresh instances for this process
        rtl_generator = RTLGenerator(
            self.config.model_name,
            self.logger,
            self.api_key,
            max_retries=self.config.max_retries,
            initial_backoff=self.config.initial_backoff,
            use_systematic_debugging=self.config.use_systematic_debugging,
            llm_context_limit=self.config.llm_context_limit,
            ace_rtl_stages=self.config.ace_rtl_stages,
            ace_rtl_temperature=self.config.ace_rtl_temperature,
            disable_testbench_access=self.config.disable_testbench_access
        )
        feedback_analyzer = FeedbackAnalyzer(
            self.config.model_name,
            self.logger,
            self.api_key,
            max_retries=self.config.max_retries,
            initial_backoff=self.config.initial_backoff
        )
        datapoint_processor = DatapointProcessor(self.logger)

        # Create benchmark integration based on mode
        if self.benchmark_mode == "verilogeval":
            benchmark_integration = VerilogEvalIntegration(
                self.config.verilogeval_path,
                self.logger,
                max_retries=self.config.max_retries,
                initial_backoff=self.config.initial_backoff
            )
        elif self.benchmark_mode == "chipbench":
            benchmark_integration = ChipBenchIntegration(
                self.config.chipbench_path,
                self.logger,
                max_retries=self.config.max_retries,
                initial_backoff=self.config.initial_backoff
            )
        elif self.benchmark_mode == "chipverilog":
            benchmark_integration = ChipVerilogIntegration(
                self.config.chipverilog_path,
                self.logger,
                max_retries=self.config.max_retries,
                initial_backoff=self.config.initial_backoff,
                evaluator_depth=self.config.chipverilog_eval_depth,
                evaluator_timeout=self.config.chipverilog_eval_timeout,
            )
        else:  # cvdp mode
            benchmark_integration = CVDPIntegration(
                str(self.cvdp_benchmark_path),
                self.logger,
                max_retries=self.config.max_retries,
                initial_backoff=self.config.initial_backoff,
                ace_rtl_stages=self.config.ace_rtl_stages,
                ace_rtl_temperature=self.config.ace_rtl_temperature
            )

        return rtl_generator, feedback_analyzer, datapoint_processor, benchmark_integration
