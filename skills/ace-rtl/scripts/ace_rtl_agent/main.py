# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main CLI interface for ACE-RTL Agent
"""

import argparse
import json
import sys
import multiprocessing
from pathlib import Path
from typing import Dict, Any

from .agent import AceRTLAgent, AgentConfig
from .utils.logging_utils import setup_logger


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="ACE-RTL Agent - Iterative RTL Generation with Feedback Loops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single datapoint with systematic debugging (default)
  ./ace_rtl_agent.py --datapoint cvdp_example_fibonacci_series_0001

  # Run 3 parallel experiments for a datapoint (first to succeed wins)
  ./ace_rtl_agent.py --datapoint cvdp_example_fibonacci_series_0001 --parallel-experiments 3

  # Process with traditional debugging only
  ./ace_rtl_agent.py --datapoint cvdp_example_fibonacci_series_0001 --no-systematic-debugging

  # Process all datapoints from a JSONL file with debug output
  python -m ace_rtl_agent --jsonl /path/to/benchmark.jsonl --debug

  # Use a NVIDIA Inference model with systematic debugging
  ./ace_rtl_agent.py --datapoint cvdp_example_16qam_mapper_0001 --model nvidia/nemotron-3-ultra-550b-a55b --max-iterations 8

  # Enable debug mode to see systematic debugging reasoning
  ./ace_rtl_agent.py --datapoint cvdp_example_fibonacci_series_0001 --debug --systematic-debugging

  # Customize API retry behavior for both CVDP initial generation AND systematic debugging
  ./ace_rtl_agent.py --datapoint cvdp_example_fibonacci_series_0001 --max-retries 5 --initial-backoff 10.0

  # High-reliability mode with extended retries for unreliable networks
  ./ace_rtl_agent.py --datapoint cvdp_example_fibonacci_series_0001 --max-retries 10 --initial-backoff 2.0 --debug

  # Run 5 parallel experiments with increased iterations
  ./ace_rtl_agent.py --datapoint cvdp_example_fibonacci_series_0001 --parallel-experiments 5 --max-iterations 15
        """
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--datapoint",
        type=str,
        help="Process a single CVDP datapoint by name"
    )
    input_group.add_argument(
        "--jsonl",
        type=str,
        help="Process all datapoints from a CVDP JSONL file"
    )
    input_group.add_argument(
        "--verilogeval-problem",
        type=str,
        help="Process a single VerilogEval problem (e.g., Prob001_zero)"
    )
    input_group.add_argument(
        "--chipbench-problem",
        type=str,
        help="Process a single ChipBench Verilog Gen problem (e.g., Prob000_Four-to-one_multiplexer)"
    )
    input_group.add_argument(
        "--chipverilog-problem",
        type=str,
        help="Process a single ChipVerilog module (e.g., mips_alu or or1200_alu)"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/nemotron-3-ultra-550b-a55b",
        help="LLM model to use for generation/debugging (default: nvidia/nemotron-3-ultra-550b-a55b via NVIDIA Inference API)"
    )

    # Agent configuration
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=10,
        help="Maximum number of iterations per datapoint (default: 10)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output_results",
        help="Output directory for results (default: output_results)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per iteration in seconds (default: 300)"
    )

    # Benchmark integration
    parser.add_argument(
        "--cvdp-benchmark-path",
        type=str,
        default=None,
        help="Path to the workspace-local CVDP benchmark directory (for CVDP mode)"
    )

    parser.add_argument(
        "--verilogeval-path",
        type=str,
        default=None,
        help="Path to the workspace-local VerilogEval benchmark directory (for VerilogEval mode)"
    )

    parser.add_argument(
        "--chipbench-path",
        type=str,
        default=None,
        help="Path to the workspace-local ChipBench benchmark directory (for ChipBench mode)"
    )
    parser.add_argument(
        "--chipverilog-path",
        type=str,
        default=None,
        help="Path to the ChipVerilog repository (for ChipVerilog mode)"
    )
    parser.add_argument(
        "--chipverilog-eval-depth",
        type=int,
        default=16,
        help="ChipVerilog bounded/induction proof depth (default: 16)"
    )
    parser.add_argument(
        "--chipverilog-eval-timeout",
        type=int,
        default=120,
        help="Per-command ChipVerilog evaluator timeout in seconds (default: 120)"
    )

    # API configuration
    parser.add_argument(
        "--api-key",
        type=str,
        help="LLM API key (optional, will use NVIDIA_API_KEY unless ACE_RTL_LLM_SCRIPT is set)"
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for ALL LLM API calls including CVDP initial generation and systematic debugging (default: 3)"
    )

    parser.add_argument(
        "--initial-backoff",
        type=float,
        default=5.0,
        help="Initial backoff time in seconds for API retries in systematic debugging (default: 5.0)"
    )

    parser.add_argument(
        "--llm-context-limit",
        type=int,
        default=300000,
        help="Context limit for LLM error analysis in characters. For long error reports, top and bottom sections will be analyzed (default: 300000)"
    )

    parser.add_argument(
        "--use-ace_rtl-model",
        nargs='*',
        type=int,
        choices=[0, 1, 2],
        metavar='STAGE',
        help="Stages to use ACE-RTL model (0=initial gen, 1=error analysis, 2=code correction). "
             "Examples: --use-ace_rtl-model 0 1 2 (all stages), --use-ace_rtl-model 0 2 (skip stage 1)"
    )

    parser.add_argument(
        "--ace_rtl-temperature",
        type=float,
        default=0.2,
        help="Temperature for ACE-RTL model API calls (default: 0.2, range: 0.0-2.0)"
    )

    parser.add_argument(
        "--parallel-experiments",
        type=int,
        default=1,
        help="Number of parallel experiments to run for each datapoint (default: 1, no parallelism). First experiment to succeed wins."
    )

    # Logging and debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (optional)"
    )

    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        default=True,
        help="Save intermediate iteration results (default: True)"
    )

    parser.add_argument(
        "--no-save-intermediate",
        action="store_false",
        dest="save_intermediate",
        help="Don't save intermediate iteration results"
    )

    # Systematic debugging options
    parser.add_argument(
        "--systematic-debugging",
        action="store_true",
        default=True,
        help="Enable systematic debugging with triage and forensic analysis (default: True)"
    )

    parser.add_argument(
        "--no-systematic-debugging",
        action="store_false",
        dest="systematic_debugging",
        help="Disable systematic debugging, use traditional approach only"
    )

    # Testbench access is DISABLED by default: the LLM only sees the simulation
    # error report, never the testbench (prevents hardcoding to expected values).
    parser.add_argument(
        "--no-testbench",
        action="store_true",
        dest="disable_testbench_access",
        default=True,
        help="(DEFAULT) Do NOT show the testbench/test file to the LLM; it only "
             "sees the simulation error report. Prevents hardcoding to the testbench."
    )
    parser.add_argument(
        "--with-testbench",
        action="store_false",
        dest="disable_testbench_access",
        help="Allow the LLM to see the testbench/test file (re-enables testbench access)."
    )

    # Output format
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results in JSON format"
    )

    return parser


def validate_args(args) -> bool:
    """Validate command line arguments."""
    errors = []

    # Validate max iterations
    if args.max_iterations < 1 or args.max_iterations > 100:
        errors.append("max-iterations must be between 1 and 100")

    # Validate timeout
    if args.timeout < 10 or args.timeout > 3600:
        errors.append("timeout must be between 10 and 3600 seconds")

    # Validate retry configuration
    if args.max_retries < 1 or args.max_retries > 20:
        errors.append("max-retries must be between 1 and 20")

    if args.initial_backoff < 0.1 or args.initial_backoff > 60.0:
        errors.append("initial-backoff must be between 0.1 and 60.0 seconds")

    # Validate LLM context limit
    if args.llm_context_limit < 1000 or args.llm_context_limit > 1000000:
        errors.append("llm-context-limit must be between 1000 and 1000000 characters")

    # Validate parallel experiments
    if args.parallel_experiments < 1 or args.parallel_experiments > 10:
        errors.append("parallel-experiments must be between 1 and 10")

    # Validate ACE-RTL temperature
    if args.ace_rtl_temperature < 0.0 or args.ace_rtl_temperature > 2.0:
        errors.append("ace_rtl-temperature must be between 0.0 and 2.0")

    # Validate paths based on mode
    if args.verilogeval_problem:
        if not args.verilogeval_path or not Path(args.verilogeval_path).exists():
            errors.append(f"VerilogEval path not found: {args.verilogeval_path}")
    elif args.chipbench_problem:
        if not args.chipbench_path or not Path(args.chipbench_path).exists():
            errors.append(f"ChipBench path not found: {args.chipbench_path}")
    elif args.chipverilog_problem:
        if not args.chipverilog_path or not Path(args.chipverilog_path).exists():
            errors.append(f"ChipVerilog path not found: {args.chipverilog_path}")
        if args.chipverilog_eval_depth < 1:
            errors.append("chipverilog-eval-depth must be at least 1")
        if args.chipverilog_eval_timeout < 1:
            errors.append("chipverilog-eval-timeout must be at least 1 second")
    else:
        # CVDP mode
        if args.jsonl and not Path(args.jsonl).exists():
            errors.append(f"JSONL file not found: {args.jsonl}")

        if not Path(args.cvdp_benchmark_path).exists():
            errors.append(f"CVDP benchmark path not found: {args.cvdp_benchmark_path}")

    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


def print_results(results: Dict[str, Any], json_output: bool = False):
    """Print results in the specified format."""
    if json_output:
        print(json.dumps(results, indent=2))
    else:
        if isinstance(results, dict) and "datapoint_name" in results:
            # Single datapoint result
            print_single_result(results)
        else:
            # Multiple datapoint results
            print_multiple_results(results)


def print_single_result(result: Dict[str, Any]):
    """Print results for a single datapoint."""
    print(f"\n{'='*60}")
    print(f"Results for: {result.get('datapoint_name', 'Unknown')}")
    print(f"{'='*60}")

    success = result.get('success', False)
    status = "SUCCESS" if success else "FAILED"
    print(f"Status: {status}")

    # Find and print the iteration count where tests first passed
    iteration_count = None
    if success:
        iteration_history = result.get('iteration_history', [])
        for iteration in iteration_history:
            if iteration.get('test_passed', False):
                iteration_count = iteration.get('iteration', None)
                break

    if iteration_count is not None:
        print(f"Iteration Count to Pass: {iteration_count}")
    else:
        max_iterations = result.get('total_iterations', result.get('max_iterations', 30))
        print(f"Iteration Count (Max Reached): {max_iterations}")

    print(f"Total Iterations: {result.get('total_iterations', 0)}")
    print(f"Successful Iterations: {result.get('successful_iterations', 0)}")
    print(f"Total Time: {result.get('total_time', 0):.2f}s")
    print(f"Average Iteration Time: {result.get('average_iteration_time', 0):.2f}s")

    if not success and result.get('final_error'):
        print(f"Final Error: {result['final_error']}")

    # Show systematic debugging information if available
    debug_info = result.get('systematic_debugging', {})
    if debug_info.get('enabled'):
        print(f"\nSystematic Debugging: Enabled")

        strategies = debug_info.get('debug_strategies_used', [])
        if strategies:
            print(f"Debug Strategies Used:")
            for strategy in strategies:
                status = "PASSED" if strategy['passed'] else "FAILED"
                print(f"  Iteration {strategy['iteration']}: {strategy['strategy']} - {status}")

        summary = debug_info.get('debug_summary', {})
        if summary:
            successful_strategies = len(summary.get('successful_iterations', []))
            total_strategies = summary.get('total_iterations', 0)
            if total_strategies > 0:
                print(f"Debug Success Rate: {successful_strategies}/{total_strategies} strategies worked")

    elif debug_info.get('enabled') is False:
        print(f"\nSystematic Debugging: Disabled (traditional mode)")

    # Show iteration breakdown
    iteration_history = result.get('iteration_history', [])
    if iteration_history:
        print(f"\nIteration History:")
        for iteration in iteration_history:
            iter_num = iteration.get('iteration', 0)
            passed = iteration.get('test_passed', False)
            time_taken = iteration.get('execution_time', 0)
            status = "PASS" if passed else "FAIL"

            # Try to show debug strategy if available
            debug_strategy = ""
            if (iteration.get('test_results') and
                isinstance(iteration['test_results'], dict) and
                'debug_metadata' in iteration['test_results']):
                metadata = iteration['test_results']['debug_metadata']
                if metadata and 'strategy_used' in metadata:
                    debug_strategy = f" [{metadata['strategy_used']}]"

            print(f"  {iter_num:2d}: {status} ({time_taken:.1f}s){debug_strategy}")

    print(f"{'='*60}\n")


def print_multiple_results(results: Dict[str, Any]):
    """Print results for multiple datapoints."""
    total_datapoints = len(results)
    successful_datapoints = sum(1 for r in results.values() if r.get('success', False))

    print(f"\n{'='*60}")
    print(f"Overall Results")
    print(f"{'='*60}")
    print(f"Total Datapoints: {total_datapoints}")
    print(f"Successful: {successful_datapoints}")
    print(f"Failed: {total_datapoints - successful_datapoints}")
    print(f"Success Rate: {(successful_datapoints/total_datapoints)*100:.1f}%")
    print(f"{'='*60}")

    # Show individual results
    for datapoint_name, result in results.items():
        success = result.get('success', False)
        iterations = result.get('total_iterations', 0)
        time_taken = result.get('total_time', 0)
        status = "SUCCESS" if success else "FAILED"

        print(f"{datapoint_name}: {status} ({iterations} iterations, {time_taken:.1f}s)")

        if not success and isinstance(result, dict) and result.get('error'):
            print(f"  Error: {result['error']}")

    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Make output_dir absolute relative to ace_rtl-agent directory
    if not Path(args.output_dir).is_absolute():
        # Get the ace_rtl-agent directory (3 levels up from this file: main.py -> ace_rtl_agent -> src -> ace_rtl-agent)
        ace_rtl_agent_dir = Path(__file__).parent.parent.parent
        args.output_dir = str(ace_rtl_agent_dir / args.output_dir)

    # Validate arguments
    if not validate_args(args):
        sys.exit(1)

    # Set up logging
    logger = setup_logger(
        "ace_rtl_agent",
        log_file=args.log_file,
        debug=args.debug
    )

    logger.info("Starting ACE-RTL Agent")

    # Determine benchmark mode
    if args.verilogeval_problem:
        benchmark_mode = "verilogeval"
    elif args.chipbench_problem:
        benchmark_mode = "chipbench"
    elif args.chipverilog_problem:
        benchmark_mode = "chipverilog"
    else:
        benchmark_mode = "cvdp"

    logger.info(f"Benchmark mode: {benchmark_mode}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max iterations: {args.max_iterations}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Systematic debugging: {'Enabled' if args.systematic_debugging else 'Disabled'}")
    logger.info(f"Testbench access: {'DISABLED (error report only)' if args.disable_testbench_access else 'Enabled'}")
    logger.info(f"API retry config: max_retries={args.max_retries}, initial_backoff={args.initial_backoff}s")
    logger.info(f"LLM context limit: {args.llm_context_limit} characters")
    logger.info(f"Parallel experiments: {args.parallel_experiments} {'(parallel mode enabled)' if args.parallel_experiments > 1 else '(single experiment)'}")

    # Convert use_ace_rtl_model list to set
    ace_rtl_stages = set(args.use_ace_rtl_model) if args.use_ace_rtl_model else set()

    # Log ACE-RTL configuration
    if ace_rtl_stages:
        stage_names = {0: "initial gen", 1: "error analysis", 2: "code correction"}
        stages_str = ", ".join([f"{s} ({stage_names[s]})" for s in sorted(ace_rtl_stages)])
        logger.info(f"ACE-RTL model enabled for stages: {stages_str}")
        logger.info(f"ACE-RTL model temperature: {args.ace_rtl_temperature}")
    else:
        logger.info("ACE-RTL model disabled - using NV Inference API/frontier LLMs for all stages")

    try:
        # Create agent configuration
        config = AgentConfig(
            max_iterations=args.max_iterations,
            model_name=args.model,
            output_dir=args.output_dir,
            enable_debug=args.debug,
            save_intermediate=args.save_intermediate,
            timeout_per_iteration=args.timeout,
            use_systematic_debugging=args.systematic_debugging,
            disable_testbench_access=args.disable_testbench_access,
            max_retries=args.max_retries,
            initial_backoff=args.initial_backoff,
            llm_context_limit=args.llm_context_limit,
            ace_rtl_stages=ace_rtl_stages,
            num_parallel_experiments=args.parallel_experiments,
            ace_rtl_temperature=args.ace_rtl_temperature,
            benchmark_mode=benchmark_mode,
            verilogeval_path=args.verilogeval_path if benchmark_mode == "verilogeval" else None,
            chipbench_path=args.chipbench_path if benchmark_mode == "chipbench" else None,
            chipverilog_path=args.chipverilog_path if benchmark_mode == "chipverilog" else None,
            chipverilog_eval_depth=args.chipverilog_eval_depth,
            chipverilog_eval_timeout=args.chipverilog_eval_timeout,
        )

        # Initialize agent
        cvdp_path = args.cvdp_benchmark_path if benchmark_mode == "cvdp" else None  # Not needed for verilogeval/chipbench
        agent = AceRTLAgent(config, cvdp_path, args.api_key)

        try:
            # Process based on mode
            if args.verilogeval_problem:
                logger.info(f"Processing VerilogEval problem: {args.verilogeval_problem}")
                results = agent.process_datapoint(args.verilogeval_problem)
            elif args.chipbench_problem:
                logger.info(f"Processing ChipBench problem: {args.chipbench_problem}")
                results = agent.process_datapoint(args.chipbench_problem)
            elif args.chipverilog_problem:
                logger.info(f"Processing ChipVerilog problem: {args.chipverilog_problem}")
                results = agent.process_datapoint(args.chipverilog_problem)
            elif args.datapoint:
                logger.info(f"Processing CVDP datapoint: {args.datapoint}")
                results = agent.process_datapoint(args.datapoint)
            else:
                logger.info(f"Processing CVDP JSONL file: {args.jsonl}")
                results = agent.process_jsonl_file(args.jsonl)

            # Print results
            print_results(results, args.json_output)

            # Determine exit code
            if isinstance(results, dict):
                if "success" in results:
                    # Single datapoint
                    exit_code = 0 if results["success"] else 1
                else:
                    # Multiple datapoints
                    exit_code = 0 if any(r.get("success", False) for r in results.values()) else 1
            else:
                exit_code = 1

            logger.info(f"ACE-RTL Agent completed with exit code {exit_code}")

        finally:
            # Always cleanup agent resources
            try:
                agent.cleanup()
            except Exception as e:
                logger.warning(f"Agent cleanup failed: {str(e)}")

        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        try:
            if 'agent' in locals():
                agent.cleanup()
        except:
            pass
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        try:
            if 'agent' in locals():
                agent.cleanup()
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    # Required for multiprocessing on Windows and ensures clean process spawning
    multiprocessing.freeze_support()
    main()
