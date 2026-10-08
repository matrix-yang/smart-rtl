# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACE-RTL agent shared data types."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class IterationResult:
    """Result from a single iteration of RTL generation and testing"""
    iteration: int
    rtl_code: str
    test_passed: bool
    error_message: Optional[str] = None
    feedback: Optional[str] = None
    execution_time: float = 0.0
    compilation_success: bool = False
    test_results: Optional[Dict] = None
    prompt_content: Optional[str] = None  # Added to store LLM prompt
    response_content: Optional[str] = None  # Added to store LLM response


@dataclass
class AgentConfig:
    """Configuration for the ACE-RTL Agent"""
    max_iterations: int = 10
    model_name: str = "nvidia/nemotron-3-ultra-550b-a55b"
    output_dir: str = "output_results"
    enable_debug: bool = False
    save_intermediate: bool = True
    timeout_per_iteration: int = 300  # 5 minutes per iteration
    use_systematic_debugging: bool = True  # Enable systematic debugging by default
    max_retries: int = 3  # Maximum number of retries for LLM API calls
    initial_backoff: float = 5.0  # Initial backoff time in seconds for API retries
    iteration_retries: int = 2  # Number of times to retry a failed iteration
    llm_context_limit: int = 300000  # Context limit for LLM error analysis in characters
    ace_rtl_stages: set = None  # Stages to use ACE-RTL model: {0: initial gen, 1: error analysis, 2: code correction}
    num_parallel_experiments: int = 1  # Number of parallel experiments to run (default: 1, no parallelism)
    ace_rtl_temperature: float = 0.2  # Temperature for ACE-RTL model API calls (default: 0.2)
    # When True (DEFAULT), the agent never feeds the testbench/test file to the LLM;
    # it only sees the simulation error report. Prevents the model from hardcoding
    # outputs by reading expected values out of the testbench. Re-enable testbench
    # access explicitly with --with-testbench if needed.
    disable_testbench_access: bool = True
    # Benchmark mode settings
    benchmark_mode: str = "cvdp"  # Mode: cvdp, verilogeval, chipbench, or chipverilog
    verilogeval_path: Optional[str] = None  # Path to verilog-eval benchmark
    chipbench_path: Optional[str] = None  # Path to ChipBench-full benchmark
    chipverilog_path: Optional[str] = None  # Path to ChipVerilog benchmark
    chipverilog_eval_depth: int = 16
    chipverilog_eval_timeout: int = 120

    def __post_init__(self):
        """Initialize ace_rtl_stages as empty set if None."""
        if self.ace_rtl_stages is None:
            self.ace_rtl_stages = set()
