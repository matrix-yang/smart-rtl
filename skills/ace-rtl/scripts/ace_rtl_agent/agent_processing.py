# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACE-RTL agent implementation mixins."""

import json
import multiprocessing
import os
import re
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .agent_types import AgentConfig, IterationResult


class AgentProcessingMixin:
    def process_datapoint(self, datapoint_name: str, experiment_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a single datapoint through iterative RTL generation.

        Args:
            datapoint_name: Name of the CVDP datapoint to process
            experiment_id: Optional experiment ID for parallel execution (0, 1, 2, etc.)

        Returns:
            Dictionary containing final results and all iteration history
        """
        # If parallel experiments are configured and this is not a sub-experiment, run in parallel
        if self.config.num_parallel_experiments > 1 and experiment_id is None:
            result = self._process_datapoint_parallel(datapoint_name)
        else:
            # Run single experiment (or this is a parallel sub-experiment)
            result = self._process_datapoint_single(datapoint_name, experiment_id)

        # Append to summary log only from main entry point (not sub-experiments)
        if experiment_id is None:
            self._append_to_summary_log(result)

        return result

    def _process_datapoint_parallel(self, datapoint_name: str) -> Dict[str, Any]:
        """
        Run multiple parallel experiments for a datapoint and return first successful result.
        Uses multiprocessing for true parallelism and complete isolation.

        Args:
            datapoint_name: Name of the CVDP datapoint to process

        Returns:
            Dictionary containing results from the first successful experiment
        """
        num_experiments = self.config.num_parallel_experiments
        self.logger.info(f"Starting {num_experiments} parallel experiments for {datapoint_name}")
        print(f"\n{'='*60}")
        print(f"PARALLEL EXECUTION: Running {num_experiments} experiments (multiprocessing)")
        print(f"   Datapoint: {datapoint_name}")
        print(f"   Strategy: First successful experiment wins")
        print(f"   Isolation: Complete (separate processes)")
        print(f"{'='*60}\n")

        # Create multiprocessing synchronization primitives
        manager = multiprocessing.Manager()
        success_event = manager.Event()  # Signals when an experiment succeeds
        result_queue = manager.Queue()   # Collects results from all experiments

        def run_experiment_process(exp_id: int, config_dict: Dict, cvdp_path: str, api_key: Optional[str],
                                   datapoint_name: str, success_event, result_queue):
            """Run a single experiment in a separate process."""
            try:
                print(f"Experiment {exp_id}: Starting in process {os.getpid()}...")

                # Reconstruct config from dict (can't pickle dataclass directly across processes easily)
                from .agent import AceRTLAgent
                from .agent_types import AgentConfig
                config = AgentConfig(**config_dict)

                # Create a standalone agent instance for this process
                agent = AceRTLAgent(config, cvdp_path, api_key)

                # Run the experiment
                result = agent._process_datapoint_single(datapoint_name, experiment_id=exp_id)

                # If this experiment succeeded, signal and send result
                if result.get('success', False):
                    if not success_event.is_set():
                        print(f"\n{'='*60}")
                        print(f"Experiment {exp_id}: SUCCESS - Signaling other experiments to stop")
                        print(f"{'='*60}\n")
                        success_event.set()  # Signal other processes to stop
                        result_queue.put(('success', exp_id, result))
                    else:
                        print(f"Experiment {exp_id}: Also succeeded (but another won)")
                        result_queue.put(('late_success', exp_id, result))
                else:
                    print(f"Experiment {exp_id}: Failed after {result.get('total_iterations', 0)} iterations")
                    result_queue.put(('failure', exp_id, result))

            except Exception as e:
                print(f"Experiment {exp_id}: Exception - {str(e)}")
                result_queue.put(('error', exp_id, {
                    "success": False,
                    "error": str(e),
                    "experiment_id": exp_id
                }))

        # Convert config to dict for pickling across processes
        config_dict = asdict(self.config)

        # Start all experiment processes
        processes = []
        for exp_id in range(num_experiments):
            process = multiprocessing.Process(
                target=run_experiment_process,
                args=(exp_id, config_dict, str(self.cvdp_benchmark_path), self.api_key,
                      datapoint_name, success_event, result_queue),
                name=f"Experiment-{exp_id}"
            )
            process.start()
            processes.append(process)
            print(f"Started experiment {exp_id} in process {process.pid}")

        # Monitor processes and terminate others when one succeeds
        all_results = []
        successful_result = None

        # Poll for results while processes are running
        while any(p.is_alive() for p in processes):
            # Check for any results
            try:
                status, exp_id, result = result_queue.get(timeout=1.0)
                all_results.append((status, exp_id, result))

                # If we got a success and it's the first one, terminate others
                if status == 'success' and successful_result is None:
                    successful_result = result
                    print(f"\nTerminating other experiments after experiment {exp_id} succeeded...")

                    # Terminate all other running processes
                    for i, p in enumerate(processes):
                        if i != exp_id and p.is_alive():
                            p.terminate()
                            print(f"   Terminated experiment {i} (PID: {p.pid})")

                    # Wait briefly for clean shutdown
                    for p in processes:
                        p.join(timeout=2.0)
                    break

            except:
                # Queue timeout, continue monitoring
                continue

        # Collect any remaining results
        while not result_queue.empty():
            status, exp_id, result = result_queue.get()
            all_results.append((status, exp_id, result))
            if status == 'success' and successful_result is None:
                successful_result = result

        # Return the first successful result, or aggregate failure
        if successful_result:
            self.logger.info(f"Parallel execution completed successfully")
            return successful_result
        else:
            self.logger.warning(f"All {num_experiments} experiments failed")
            # Return results from all experiments for debugging
            return {
                "success": False,
                "datapoint_name": datapoint_name,
                "num_experiments": num_experiments,
                "all_experiments_failed": True,
                "experiment_results": [result for status, exp_id, result in all_results]
            }

    def _should_stop_experiment(self, success_event=None) -> bool:
        """Check if experiment should stop due to another experiment succeeding."""
        # For multiprocessing, the event is passed as parameter
        # For backward compatibility, also check old threading-based attribute
        if success_event and success_event.is_set():
            return True
        return False

    def _process_datapoint_single(self, datapoint_name: str, experiment_id: Optional[int] = None,
                                  rtl_generator=None, feedback_analyzer=None,
                                  datapoint_processor=None, cvdp_integration=None) -> Dict[str, Any]:
        """
        Process a single datapoint through iterative RTL generation.

        Args:
            datapoint_name: Name of the CVDP datapoint to process
            experiment_id: Optional experiment ID for parallel execution subdirectory
            rtl_generator: Optional thread-local RTL generator instance
            feedback_analyzer: Optional thread-local feedback analyzer instance
            datapoint_processor: Optional thread-local datapoint processor instance
            cvdp_integration: Optional thread-local CVDP integration instance

        Returns:
            Dictionary containing final results and all iteration history
        """
        # Use thread-local instances if provided, otherwise use shared instances
        rtl_gen = rtl_generator or self.rtl_generator
        feedback_an = feedback_analyzer or self.feedback_analyzer
        dp_proc = datapoint_processor or self.datapoint_processor
        benchmark_int = cvdp_integration or self.benchmark_integration

        exp_suffix = f" (Experiment {experiment_id})" if experiment_id is not None else ""
        self.logger.info(f"Starting processing of datapoint: {datapoint_name}{exp_suffix}")
        start_time = time.time()

        # Create local iteration results list (thread-safe for parallel execution)
        iteration_results = []

        if hasattr(rtl_gen, 'reset_for_new_datapoint'):
            rtl_gen.reset_for_new_datapoint()

        # Clear previous results and reset RTL rule advisor session (only for shared instance in single mode)
        if experiment_id is None:
            self.iteration_results = []
        if hasattr(feedback_an, 'comprehensive_error_analyzer') and \
           hasattr(feedback_an.comprehensive_error_analyzer, 'rtl_rule_advisor'):
            feedback_an.comprehensive_error_analyzer.rtl_rule_advisor.reset_session()
        # Also reset the analyzer used by the LLM-centric debugger to ensure retrieval on restarts
        if hasattr(rtl_gen, 'llm_centric_debugger') and \
           hasattr(rtl_gen.llm_centric_debugger, 'error_analyzer') and \
           hasattr(rtl_gen.llm_centric_debugger.error_analyzer, 'rtl_rule_advisor'):
            rtl_gen.llm_centric_debugger.error_analyzer.rtl_rule_advisor.reset_session()

        try:
            # CVDP owns JSONL datapoints. Direct benchmark integrations expose
            # their public prompt themselves, so do not scan unrelated CVDP
            # datasets before starting those runs.
            if self.benchmark_mode == "cvdp":
                current_datapoint = dp_proc.load_datapoint(datapoint_name)
            else:
                current_datapoint = {
                    "id": datapoint_name,
                    "categories": [self.benchmark_mode],
                    "input": {
                        "prompt": benchmark_int.get_problem_prompt(datapoint_name),
                        "context": {},
                    },
                    "output": {"response": "", "context": {}},
                    "harness": {"files": {}},
                }

            # Sync to shared state for single execution mode (backward compatibility)
            if experiment_id is None:
                self.current_datapoint = current_datapoint

            # Setup CVDP environment (it will create the proper directory structure)
            # If experiment_id is provided, append it to output_dir for separate experiment directories
            output_dir = self.config.output_dir
            if experiment_id is not None:
                output_dir = f"{output_dir}/{experiment_id}"

            benchmark_int.setup_datapoint_environment(
                datapoint_name,
                output_dir,  # Pass base output directory (with experiment subdirectory if parallel)
                self.config.model_name
            )

            # Iterative improvement loop (iteration log will be initialized after structure discovery)
            final_result = self._iterative_improvement_loop(datapoint_name, rtl_gen, feedback_an, dp_proc, benchmark_int, iteration_results, current_datapoint)

            # Generate final report
            total_time = time.time() - start_time
            final_report = self._generate_final_report(datapoint_name, total_time, iteration_results)

            # Add experiment_id to final report for tracking
            final_report['experiment_id'] = experiment_id if experiment_id is not None else 0

            # Extract and add CID from datapoint categories
            if current_datapoint and 'categories' in current_datapoint:
                categories = current_datapoint['categories']
                for cat in categories:
                    if isinstance(cat, str) and cat.startswith('cid'):
                        final_report['cid'] = cat
                        break

            # Save results (use the base path if available)
            output_dir = (
                getattr(benchmark_int, 'base_path', None)
                or getattr(benchmark_int, 'problem_output_dir', None)
                or (Path(self.config.output_dir) / datapoint_name)
            )
            self._save_results(output_dir, final_report)
            if hasattr(benchmark_int, "write_summary"):
                benchmark_int.write_summary(final_report)

            # Sync local results to shared state for single execution mode (backward compatibility)
            if experiment_id is None:
                self.iteration_results = iteration_results

            self.logger.info(f"Completed processing {datapoint_name} in {total_time:.2f}s")
            return final_report

        except Exception as e:
            self.logger.error(f"Error processing datapoint {datapoint_name}: {str(e)}")
            raise
        finally:
            # Clean up test environment and any remaining files
            try:
                benchmark_int.cleanup_test_environment(datapoint_name)
            except Exception as e:
                self.logger.warning(f"Cleanup failed for {datapoint_name}: {str(e)}")

            # Final cleanup of any files that might have been created (CVDP-specific)
            if hasattr(benchmark_int, '_cleanup_unwanted_cvdp_files'):
                try:
                    benchmark_int._cleanup_unwanted_cvdp_files()
                except Exception as e:
                    self.logger.warning(f"CVDP file cleanup failed for {datapoint_name}: {str(e)}")

    def process_jsonl_file(self, jsonl_file: str) -> Dict[str, Any]:
        """
        Process all datapoints from a JSONL file.

        Args:
            jsonl_file: Path to the JSONL file containing datapoints

        Returns:
            Dictionary with results for each datapoint
        """
        self.logger.info(f"Processing JSONL file: {jsonl_file}")

        datapoints = self.datapoint_processor.load_jsonl_file(jsonl_file)
        if hasattr(self.benchmark_integration, "preferred_jsonl_file"):
            self.benchmark_integration.preferred_jsonl_file = jsonl_file
        else:
            setattr(self.benchmark_integration, "preferred_jsonl_file", jsonl_file)
        setattr(self.benchmark_integration, "preserve_existing_component_dirs", True)
        os.environ["ACE_RTL_CVDP_JSONL"] = str(Path(jsonl_file).resolve())
        results = {}

        for datapoint in datapoints:
            datapoint_name = datapoint.get("id", "unknown")
            try:
                result = self.process_datapoint(datapoint_name)
                results[datapoint_name] = result

                # Clean up files after each datapoint (CVDP-specific)
                if hasattr(self.benchmark_integration, '_cleanup_unwanted_cvdp_files'):
                    try:
                        self.benchmark_integration._cleanup_unwanted_cvdp_files()
                    except Exception as cleanup_error:
                        self.logger.warning(f"CVDP file cleanup failed after {datapoint_name}: {cleanup_error}")

            except Exception as e:
                self.logger.error(f"Failed to process {datapoint_name}: {str(e)}")
                results[datapoint_name] = {
                    "success": False,
                    "error": str(e),
                    "iterations": 0
                }

                # Clean up even on error (CVDP-specific)
                if hasattr(self.benchmark_integration, '_cleanup_unwanted_cvdp_files'):
                    try:
                        self.benchmark_integration._cleanup_unwanted_cvdp_files()
                    except Exception as cleanup_error:
                        self.logger.warning(f"CVDP file cleanup failed after error on {datapoint_name}: {cleanup_error}")

        # Clean up global resources after processing all datapoints
        try:
            self.benchmark_integration.cleanup_global_resources()
        except Exception as e:
            self.logger.warning(f"Global cleanup failed: {str(e)}")

        return results
