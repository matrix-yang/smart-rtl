# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VerilogEval Integration - Interfaces with the VerilogEval benchmark
"""

import os
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .vcd_parser import VCDParser


class VerilogEvalIntegration:
    """
    Integrates with the VerilogEval benchmark to run RTL testing using iverilog.
    Much simpler than CVDP as it doesn't require Docker or complex infrastructure.
    """

    def __init__(self, verilogeval_path: str, logger, max_retries: int = 3, initial_backoff: float = 5.0):
        """
        Initialize VerilogEval integration.

        Args:
            verilogeval_path: Path to the VerilogEval benchmark directory
            logger: Logger instance
            max_retries: Maximum number of retries for LLM API calls (default: 3)
            initial_backoff: Initial backoff time in seconds for retries (default: 5.0)
        """
        self.verilogeval_path = Path(verilogeval_path)
        self.dataset_path = self.verilogeval_path / "dataset_spec-to-rtl"
        self.logger = logger
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

        # Storage for current problem context
        self.current_problem = None
        self.current_prompt = None
        self.current_testbench = None
        self.current_ref_solution = None  # Only for compilation, NEVER shown to LLM

        # Validate VerilogEval environment
        self._validate_environment()

    def _validate_environment(self):
        """Validate that the VerilogEval environment is available."""
        if not self.verilogeval_path.exists():
            raise FileNotFoundError(f"VerilogEval path not found: {self.verilogeval_path}")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"VerilogEval dataset path not found: {self.dataset_path}")

        # Check for iverilog
        try:
            result = subprocess.run(
                ["iverilog", "-V"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.logger.info(f"Found iverilog: {result.stdout.split()[0:3]}")
            else:
                self.logger.warning("iverilog command failed")
        except FileNotFoundError:
            self.logger.error("iverilog not found in PATH. Please install iverilog v12.")
            raise RuntimeError("iverilog not installed or not in PATH")
        except Exception as e:
            self.logger.warning(f"Could not verify iverilog: {str(e)}")

        self.logger.info("VerilogEval environment validation completed")

    def get_problem_list(self) -> List[str]:
        """
        Get list of all problems in the dataset.

        Returns:
            List of problem names (e.g., ['Prob001_zero', 'Prob002_m2014_q4i', ...])
        """
        problems = set()
        for file in self.dataset_path.glob("Prob*_prompt.txt"):
            # Extract problem name without _prompt.txt suffix
            problem_name = file.stem.replace("_prompt", "")
            problems.add(problem_name)

        return sorted(list(problems))

    def load_problem(self, problem_name: str) -> Dict[str, Any]:
        """
        Load problem specification, testbench, and reference solution.

        Note: The reference solution is ONLY used for compilation (testbench requires RefModule).
              It is NEVER shown to the LLM or used in prompts.

        Args:
            problem_name: Name of the problem (e.g., 'Prob001_zero')

        Returns:
            Dictionary containing prompt and testbench content
        """
        prompt_file = self.dataset_path / f"{problem_name}_prompt.txt"
        test_file = self.dataset_path / f"{problem_name}_test.sv"
        ref_file = self.dataset_path / f"{problem_name}_ref.sv"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        if not ref_file.exists():
            raise FileNotFoundError(f"Reference file not found: {ref_file}")

        with open(prompt_file, 'r') as f:
            prompt = f.read()

        with open(test_file, 'r') as f:
            testbench = f.read()

        with open(ref_file, 'r') as f:
            ref_solution = f.read()

        self.current_problem = problem_name
        self.current_prompt = prompt
        self.current_testbench = testbench
        self.current_ref_solution = ref_solution

        self.logger.info(f"Loaded problem: {problem_name}")
        self.logger.info(f"  Prompt: {len(prompt)} chars")
        self.logger.info(f"  Testbench: {len(testbench)} chars")
        self.logger.info(f"  Reference (for compilation only): {len(ref_solution)} chars")

        return {
            "problem_name": problem_name,
            "prompt": prompt,
            "testbench": testbench
        }

    def setup_datapoint_environment(self, problem_name: str, base_output_dir: str, model_name: str):
        """
        Set up the problem environment (create output directories).

        Args:
            problem_name: Name of the problem
            base_output_dir: Base output directory
            model_name: Model name for organizing results
        """
        self.logger.info(f"Setting up VerilogEval environment for problem: {problem_name}")

        # Store configuration
        self.current_problem = problem_name
        self.base_output_dir = Path(base_output_dir)
        self.model_name = model_name

        # Create output directory structure
        self.problem_output_dir = self.base_output_dir / model_name / problem_name
        self.problem_output_dir.mkdir(parents=True, exist_ok=True)

        # Set base_path to problem_output_dir for iteration logging compatibility
        self.base_path = self.problem_output_dir

        # Load problem data
        self.load_problem(problem_name)

        self.logger.info(f"Problem output directory: {self.problem_output_dir}")
        self.logger.info(f"Iterations will be saved under: {self.base_path / 'iterations'}")

    def get_problem_prompt(self, problem_name: str) -> str:
        """
        Get the prompt for a specific problem.

        Args:
            problem_name: Name of the problem

        Returns:
            Problem prompt text
        """
        if self.current_problem == problem_name and self.current_prompt:
            return self.current_prompt

        problem_data = self.load_problem(problem_name)
        return problem_data["prompt"]

    def get_sanitized_testbench(self) -> Optional[str]:
        """
        Return a sanitized version of the current testbench suitable for LLM context.

        Strips out RefModule instantiation and comparison logic while keeping:
        - stimulus_gen module (input generation patterns)
        - Clock/reset generation
        - TopModule port connections
        - Signal declarations
        - Timeout value

        This gives the LLM insight into stimulus patterns without leaking the golden solution.
        """
        if not self.current_testbench:
            return None

        lines = self.current_testbench.split('\n')
        sanitized = []
        skip_ref_block = False

        for line in lines:
            stripped = line.strip()

            # Skip RefModule instantiation block
            if 'RefModule' in line and ('good1' in line or 'ref' in line.lower()):
                skip_ref_block = True
                continue
            if skip_ref_block:
                if ');' in line:
                    skip_ref_block = False
                continue

            # Skip _ref signal declarations
            if re.search(r'\b\w+_ref\b', stripped) and ('logic' in stripped or 'wire' in stripped or 'reg' in stripped):
                continue

            # Skip $dumpvars lines (contain _ref signals)
            if '$dumpvars' in stripped:
                continue

            # Skip tb_match / verification XOR lines that reference _ref
            if '_ref' in line and ('tb_match' in line or 'assign' in stripped or '===' in line or '!==' in line):
                continue

            # Skip mismatch counting logic that references _ref
            if '_ref' in line and ('errors' in line or 'errortime' in line):
                continue

            sanitized.append(line)

        result = '\n'.join(sanitized)

        # Add a header explaining what this is
        header = (
            "// TESTBENCH STIMULUS CONTEXT (sanitized - golden solution removed)\n"
            "// Shows: input generation, clock/reset, TopModule port connections, timeout\n"
            "// Purpose: Understand stimulus patterns, timing, and which signals are tested\n\n"
        )

        return header + result

    def test_rtl_code(self, problem_name: str, rtl_code: str) -> Dict[str, Any]:
        """
        Test RTL code using iverilog and the problem's testbench.

        Args:
            problem_name: Name of the problem
            rtl_code: RTL code to test

        Returns:
            Dictionary with test results including success/failure and error messages
        """
        self.logger.info(f"Testing RTL code for problem: {problem_name}")

        # Load problem if not already loaded
        if self.current_problem != problem_name or not self.current_testbench:
            self.load_problem(problem_name)

        # Create temporary directory for compilation and testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write RTL code to file (generated by LLM)
            rtl_file = temp_path / "TopModule.sv"
            with open(rtl_file, 'w') as f:
                f.write(rtl_code)

            # Write testbench to file
            tb_file = temp_path / "testbench.sv"
            with open(tb_file, 'w') as f:
                f.write(self.current_testbench)

            # Write reference solution to file (needed for testbench RefModule)
            # NOTE: This is ONLY for compilation - NEVER shown to LLM
            ref_file = temp_path / "RefModule.sv"
            with open(ref_file, 'w') as f:
                f.write(self.current_ref_solution)

            # Compile with iverilog (includes ref file for RefModule)
            compile_result = self._compile_with_iverilog(temp_path, rtl_file, tb_file, ref_file)

            if not compile_result["success"]:
                # Clean up temp dir paths in error messages for readability
                clean_error = self._clean_temp_paths(compile_result["error"], temp_path)
                return {
                    "passed": False,
                    "compilation_success": False,
                    "error_message": "Compilation failed",
                    "error_report": clean_error,
                    "details": compile_result
                }

            # Run simulation
            sim_result = self._run_simulation(temp_path)

            if not sim_result["success"]:
                return {
                    "passed": False,
                    "compilation_success": True,
                    "error_message": "Simulation failed",
                    "error_report": sim_result["error"],
                    "details": sim_result
                }

            # Check if tests passed (mismatches == 0)
            passed = sim_result["mismatches"] == 0

            if passed:
                return {
                    "passed": True,
                    "compilation_success": True,
                    "error_message": None,
                    "error_report": None,
                    "details": sim_result
                }
            else:
                # Use enhanced error report if available, otherwise fall back to raw output
                error_report = sim_result.get("enhanced_error_report") or sim_result["output"]

                return {
                    "passed": False,
                    "compilation_success": True,
                    "error_message": f"Test failed with {sim_result['mismatches']} mismatches",
                    "error_report": error_report,
                    "details": sim_result
                }

    def _compile_with_iverilog(self, work_dir: Path, rtl_file: Path, tb_file: Path, ref_file: Path) -> Dict[str, Any]:
        """
        Compile RTL and testbench with iverilog.

        Note: ref_file contains RefModule needed by testbench but is NEVER shown to LLM.

        Args:
            work_dir: Working directory
            rtl_file: Path to generated RTL file (TopModule)
            tb_file: Path to testbench file
            ref_file: Path to reference solution file (RefModule - for compilation only)

        Returns:
            Dictionary with compilation results
        """
        output_file = work_dir / "sim"

        try:
            result = subprocess.run(
                ["iverilog", "-g2012", "-o", str(output_file), str(tb_file), str(rtl_file), str(ref_file)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                self.logger.info("Compilation successful")
                return {
                    "success": True,
                    "error": None,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                self.logger.warning(f"Compilation failed with return code {result.returncode}")
                error_msg = result.stderr if result.stderr else result.stdout
                return {
                    "success": False,
                    "error": error_msg,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        except subprocess.TimeoutExpired:
            self.logger.error("Compilation timeout")
            return {
                "success": False,
                "error": "Compilation timeout after 30 seconds"
            }
        except Exception as e:
            self.logger.error(f"Compilation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _run_simulation(self, work_dir: Path) -> Dict[str, Any]:
        """
        Run iverilog simulation and capture detailed results including VCD waveform.

        Args:
            work_dir: Working directory containing compiled simulation

        Returns:
            Dictionary with simulation results including detailed error information
        """
        sim_file = work_dir / "sim"
        vcd_file = work_dir / "wave.vcd"

        try:
            result = subprocess.run(
                ["vvp", str(sim_file)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=60
            )

            output = result.stdout + result.stderr

            # Parse output for mismatches
            mismatches = self._parse_mismatches(output)

            # Extract detailed mismatch information
            mismatch_details = self._parse_mismatch_details(output)

            # Parse VCD file if it exists and there are mismatches
            vcd_data = None
            waveform_excerpt = None
            if vcd_file.exists() and mismatches > 0:
                try:
                    vcd_data = self._extract_vcd_data(vcd_file, mismatch_details)
                    waveform_excerpt = vcd_data.get('waveform_text', None)
                except Exception as e:
                    self.logger.warning(f"Could not parse VCD file: {str(e)}")

            # Parse per-signal mismatch breakdown from Hint: lines
            per_signal_mismatches = self._parse_per_signal_mismatches(output)

            # Generate enhanced error report
            enhanced_error_report = None
            if mismatches > 0:
                enhanced_error_report = self._generate_enhanced_error_report(
                    output, mismatch_details, vcd_data
                )

            base_result = {
                "output": output,
                "mismatches": mismatches,
                "mismatch_details": mismatch_details,
                "vcd_data": vcd_data,
                "waveform_excerpt": waveform_excerpt,
                "enhanced_error_report": enhanced_error_report,
                "per_signal_mismatches": per_signal_mismatches,
                "returncode": result.returncode
            }

            if result.returncode == 0:
                self.logger.info(f"Simulation completed: {mismatches} mismatches")
                base_result.update({"success": True, "error": None})
                return base_result
            else:
                self.logger.warning(f"Simulation failed with return code {result.returncode}")
                base_result.update({"success": False, "error": output})
                return base_result
        except subprocess.TimeoutExpired:
            self.logger.error("Simulation timeout")
            return {
                "success": False,
                "error": "Simulation timeout after 60 seconds",
                "output": "",
                "mismatches": -1
            }
        except Exception as e:
            self.logger.error(f"Simulation error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "mismatches": -1
            }

    def _clean_temp_paths(self, error_msg: str, temp_path: Path) -> str:
        """Replace /tmp/tmpXXXX/ paths in error messages with readable filenames."""
        if not error_msg:
            return error_msg

        temp_str = str(temp_path)
        # Replace the full temp path with descriptive prefix
        cleaned = error_msg.replace(temp_str + "/", "")
        # Also handle any other /tmp/tmpXXX/ patterns that might appear
        cleaned = re.sub(r'/tmp/tmp[a-zA-Z0-9_]+/', '', cleaned)
        return cleaned

    def _parse_per_signal_mismatches(self, output: str) -> List[Dict[str, Any]]:
        """
        Parse per-signal mismatch info from simulation Hint: lines.

        Returns list of dicts with signal name, mismatch count, and first mismatch time.
        """
        signals = []
        for line in output.split('\n'):
            line = line.strip()
            if not line.startswith('Hint:'):
                continue

            # "Hint: Output 'X' has N mismatches. First mismatch occurred at time T."
            match = re.search(
                r"Hint:\s*Output\s+'(\w+)'\s+has\s+(\d+)\s+mismatches?\.\s*First mismatch occurred at time\s+(\d+)",
                line
            )
            if match:
                signals.append({
                    "signal": match.group(1),
                    "mismatches": int(match.group(2)),
                    "first_mismatch_time": int(match.group(3))
                })
                continue

            # "Hint: Output 'X' has no mismatches."
            match = re.search(r"Hint:\s*Output\s+'(\w+)'\s+has\s+no\s+mismatches", line)
            if match:
                signals.append({
                    "signal": match.group(1),
                    "mismatches": 0,
                    "first_mismatch_time": None
                })

        return signals

    def _parse_mismatches(self, output: str) -> int:
        """
        Parse simulation output to extract mismatch count.

        Args:
            output: Simulation output text

        Returns:
            Number of mismatches (or -1 if couldn't parse)
        """
        # Format 1 (VerilogEval): "Mismatches: X in Y samples"
        match = re.search(r"Mismatches:\s*(\d+)\s+in\s+\d+\s+samples", output)
        if match:
            return int(match.group(1))

        # Format 2 (ChipBench): "Total mismatched samples is X out of Y samples"
        match = re.search(r"Total mismatched samples is (\d+) out of (\d+)", output)
        if match:
            return int(match.group(1))

        # Format 3 (ChipBench zero-mismatch variant): "No mismatched samples"
        if re.search(r"No mismatched samples", output):
            return 0

        if "ERROR" in output.upper() or "FATAL" in output.upper():
            return -1

        return -1

    def cleanup_test_environment(self, problem_name: str):
        """Clean up any temporary test environments."""
        self.logger.info(f"Cleaning up test environment for {problem_name}")
        # For VerilogEval, we use temporary directories so no explicit cleanup needed

    def cleanup_global_resources(self):
        """Clean up global resources."""
        self.logger.info("VerilogEval cleanup complete")
        # No global resources to clean up for VerilogEval

    def _parse_mismatch_details(self, output: str) -> List[Dict[str, Any]]:
        """
        Parse testbench output to extract detailed mismatch information.

        Args:
            output: Simulation output text

        Returns:
            List of dictionaries containing mismatch details
        """
        details = []

        # Look for "Hint: Output 'signal' has X mismatches. First mismatch occurred at time Y."
        output_pattern = r"Hint: Output '(\w+)' has (\d+) mismatches?\. First mismatch occurred at time (\d+)\."
        for match in re.finditer(output_pattern, output):
            signal_name = match.group(1)
            mismatch_count = int(match.group(2))
            first_time = int(match.group(3))

            details.append({
                'signal': signal_name,
                'mismatch_count': mismatch_count,
                'first_mismatch_time': first_time
            })

        # Look for total sample information
        total_pattern = r"Hint: Total mismatched samples is (\d+) out of (\d+) samples"
        match = re.search(total_pattern, output)
        if match:
            total_mismatches = int(match.group(1))
            total_samples = int(match.group(2))
            for detail in details:
                detail['total_mismatches'] = total_mismatches
                detail['total_samples'] = total_samples
                detail['mismatch_percentage'] = (total_mismatches / total_samples * 100) if total_samples > 0 else 0

        return details

    def _extract_vcd_data(self, vcd_file: Path, mismatch_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract relevant signal values from VCD file around mismatch times.

        Args:
            vcd_file: Path to VCD file
            mismatch_details: List of mismatch details

        Returns:
            Dictionary containing extracted VCD data and formatted waveforms
        """
        try:
            parser = VCDParser(vcd_file)
            parser.parse()

            vcd_info = {
                'signals_at_mismatches': [],
                'waveform_text': None
            }

            # Extract signal values at each mismatch time
            for detail in mismatch_details:
                mismatch_time = detail['first_mismatch_time']
                signal_name = detail['signal']

                # Get all signals at the mismatch time
                all_signals = parser.get_all_signals_at_time(mismatch_time)

                # Get signals around the mismatch time for context
                signals_around = parser.get_signals_around_time(mismatch_time, before=20, after=20)

                vcd_info['signals_at_mismatches'].append({
                    'time': mismatch_time,
                    'signal': signal_name,
                    'all_signals': all_signals,
                    'context': signals_around
                })

            # Generate a text-based waveform for the first mismatch
            if mismatch_details:
                first_mismatch = mismatch_details[0]
                mismatch_time = first_mismatch['first_mismatch_time']
                signal_name = first_mismatch['signal']

                # Get all relevant signal names
                sample_signals = parser.get_all_signals_at_time(mismatch_time)

                # Prioritize showing: inputs, expected output, actual output
                priority_signals = []

                # Add the main signal's ref and dut versions
                ref_signal = f"{signal_name}_ref"
                dut_signal = f"{signal_name}_dut"
                if ref_signal in sample_signals:
                    priority_signals.append(ref_signal)
                if dut_signal in sample_signals:
                    priority_signals.append(dut_signal)

                # Add input signals (exclude ref/dut/internal signals)
                for sig in sorted(sample_signals.keys()):
                    if (not sig.endswith('_ref') and
                        not sig.endswith('_dut') and
                        not sig.startswith('tb_') and
                        sig not in priority_signals):
                        priority_signals.append(sig)
                        if len(priority_signals) >= 8:  # Limit to 8 signals total
                            break

                # Generate waveform text
                start_time = max(0, mismatch_time - 40)
                end_time = mismatch_time + 20

                if priority_signals:
                    vcd_info['waveform_text'] = parser.format_waveform_text(
                        priority_signals,
                        start_time,
                        end_time,
                        mark_time=mismatch_time
                    )

            return vcd_info

        except Exception as e:
            self.logger.warning(f"Error extracting VCD data: {str(e)}")
            return {}

    def _generate_enhanced_error_report(self, output: str, mismatch_details: List[Dict[str, Any]],
                                       vcd_data: Optional[Dict[str, Any]]) -> str:
        """
        Generate an enhanced error report with detailed debugging information.

        Args:
            output: Original simulation output
            mismatch_details: Parsed mismatch details
            vcd_data: Extracted VCD waveform data

        Returns:
            Formatted enhanced error report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("TEST FAILED - DETAILED ERROR REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary section
        if mismatch_details:
            total_mismatches = mismatch_details[0].get('total_mismatches', 'unknown')
            total_samples = mismatch_details[0].get('total_samples', 'unknown')
            mismatch_pct = mismatch_details[0].get('mismatch_percentage', 0)

            lines.append("SUMMARY:")
            lines.append(f"  Total Mismatches: {total_mismatches} out of {total_samples} samples ({mismatch_pct:.1f}%)")
            lines.append(f"  Signals with errors: {len(mismatch_details)}")
            lines.append("")

        # Detailed mismatch information for each signal
        lines.append("DETAILED MISMATCH ANALYSIS:")
        lines.append("-" * 80)

        for detail in mismatch_details:
            signal = detail['signal']
            count = detail['mismatch_count']
            first_time = detail['first_mismatch_time']

            lines.append(f"\nOutput Signal: '{signal}'")
            lines.append(f"  Number of mismatches: {count}")
            lines.append(f"  First mismatch occurred at: {first_time}ps")
            lines.append("")

            # For very early mismatches, show initialization context
            if first_time <= 10 and vcd_data and 'signals_at_mismatches' in vcd_data:
                lines.append("  NOTE: This is a very early mismatch (initialization issue)")
                lines.append("  The expected output likely needs to start at a specific initial value.")
                lines.append("")

            # Add VCD data if available with expected vs actual
            if vcd_data and 'signals_at_mismatches' in vcd_data:
                for mismatch_info in vcd_data['signals_at_mismatches']:
                    if mismatch_info['signal'] == signal and mismatch_info['time'] == first_time:
                        all_sigs = mismatch_info['all_signals']

                        # Extract expected (ref) and actual (dut) values
                        expected_key = f"{signal}_ref"
                        actual_key = f"{signal}_dut"

                        lines.append(f"  At time {first_time}ps:")

                        # Show input signals
                        lines.append("    Input signals:")
                        for sig_name, sig_val in sorted(all_sigs.items()):
                            if not sig_name.endswith('_ref') and not sig_name.endswith('_dut') and sig_name != signal:
                                # Format binary values more nicely
                                if len(sig_val) > 1:
                                    lines.append(f"      {sig_name:15s} = 8'b{sig_val}")
                                else:
                                    lines.append(f"      {sig_name:15s} = {sig_val}")

                        # Show expected vs actual
                        lines.append("")
                        lines.append("    Output comparison:")
                        expected_val = all_sigs.get(expected_key, 'unknown')
                        actual_val = all_sigs.get(actual_key, 'unknown')

                        if expected_val != 'unknown' or actual_val != 'unknown':
                            if len(expected_val) > 1 if expected_val != 'unknown' else False:
                                lines.append(f"      Expected ({signal}): 8'b{expected_val}")
                                lines.append(f"      Your code produced: 8'b{actual_val}")
                            else:
                                lines.append(f"      Expected ({signal}): {expected_val}")
                                lines.append(f"      Your code produced: {actual_val}")

                            # Detect misleading identical values from VCD timing artifacts
                            if expected_val != 'unknown' and actual_val != 'unknown' and expected_val == actual_val:
                                lines.append("")
                                lines.append("      WARNING: Expected and actual values appear IDENTICAL at this VCD sample point.")
                                lines.append("      The mismatch likely occurred at a clock edge transition that the VCD did not capture.")
                                lines.append("      This means your signal had the WRONG value briefly at the clock edge, then settled")
                                lines.append("      to the correct value. Focus on combinational logic timing or registered output behavior.")
                                lines.append(f"      Check signal transitions around time {first_time}ps (+/- one clock cycle).")

                            # Show bit-by-bit comparison for multi-bit values
                            if expected_val != 'unknown' and actual_val != 'unknown' and len(expected_val) > 1:
                                lines.append("")
                                lines.append("      Bit-by-bit comparison:")
                                lines.append(f"        Bit positions: {''.join([str(7-i) for i in range(min(8, len(expected_val)))])}")
                                lines.append(f"        Expected:      {expected_val[:8] if len(expected_val) >= 8 else expected_val}")
                                lines.append(f"        Actual:        {actual_val[:8] if len(actual_val) >= 8 else actual_val}")
                                # Show which bits differ
                                diff_bits = []
                                for i in range(min(len(expected_val), len(actual_val))):
                                    if expected_val[i] != actual_val[i]:
                                        bit_pos = len(expected_val) - 1 - i
                                        diff_bits.append(str(bit_pos))
                                if diff_bits:
                                    lines.append(f"        Bits that differ: {', '.join(diff_bits)}")
                        else:
                            # Fall back to showing all signals
                            for sig_name, sig_val in sorted(all_sigs.items()):
                                if len(sig_val) > 1:
                                    lines.append(f"      {sig_name:20s} = 8'b{sig_val}")
                                else:
                                    lines.append(f"      {sig_name:20s} = {sig_val}")

                        break
            else:
                lines.append("  (VCD waveform data not available)")

        # Add waveform excerpt if available
        if vcd_data and vcd_data.get('waveform_text'):
            lines.append("")
            lines.append("=" * 80)
            lines.append(vcd_data['waveform_text'])
            lines.append("")
        else:
            lines.append("")
            lines.append("(Waveform data not available - VCD parsing may have failed)")
            lines.append("")

        # Pattern analysis
        lines.append("=" * 80)
        lines.append("FAILURE PATTERN ANALYSIS:")
        lines.append("-" * 80)

        patterns = self._analyze_failure_patterns(mismatch_details, vcd_data)
        if patterns:
            for pattern in patterns:
                lines.append(f"  - {pattern}")
        else:
            lines.append("  No obvious patterns detected.")

        # Suggestions
        lines.append("")
        lines.append("=" * 80)
        lines.append("DEBUGGING SUGGESTIONS:")
        lines.append("-" * 80)
        suggestions = self._generate_debugging_suggestions(mismatch_details, vcd_data)
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                lines.append(f"  {i}. {suggestion}")
        else:
            lines.append("  Check signal values and timing carefully.")

        lines.append("")

        # Append raw Hint: lines from simulation output as fallback context
        hint_lines = [line.strip() for line in output.split('\n') if line.strip().startswith('Hint:')]
        if hint_lines:
            lines.append("=" * 80)
            lines.append("RAW SIMULATION HINTS (per-signal mismatch counts and first-mismatch times):")
            lines.append("-" * 80)
            for hint in hint_lines:
                lines.append(f"  {hint}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def _analyze_failure_patterns(self, mismatch_details: List[Dict[str, Any]],
                                  vcd_data: Optional[Dict[str, Any]]) -> List[str]:
        """
        Analyze failure patterns to provide insights.

        Args:
            mismatch_details: Parsed mismatch details
            vcd_data: Extracted VCD data

        Returns:
            List of pattern descriptions
        """
        patterns = []

        if not mismatch_details:
            return patterns

        # Check if only a small percentage of tests fail
        if mismatch_details and 'mismatch_percentage' in mismatch_details[0]:
            pct = mismatch_details[0]['mismatch_percentage']
            if pct < 5:
                patterns.append(f"Very low failure rate ({pct:.1f}%) suggests initialization or edge case issue")
            elif pct > 70:
                patterns.append(f"High failure rate ({pct:.1f}%) suggests fundamental logic error")

        # Check if failure occurs very early
        first_times = [d['first_mismatch_time'] for d in mismatch_details]
        min_time = min(first_times) if first_times else 0
        if min_time < 10:
            patterns.append(f"First mismatch at time {min_time}ps (very early) - check initialization/reset logic")

        # Check if multiple signals fail
        if len(mismatch_details) > 1:
            patterns.append(f"Multiple outputs affected ({len(mismatch_details)} signals) - likely affects shared logic")

        # Check if all mismatches are at same time
        if len(set(first_times)) == 1 and len(first_times) > 1:
            patterns.append(f"All signals fail at same time ({first_times[0]}ps) - likely timing/control issue")

        return patterns

    def _generate_debugging_suggestions(self, mismatch_details: List[Dict[str, Any]],
                                       vcd_data: Optional[Dict[str, Any]]) -> List[str]:
        """
        Generate specific debugging suggestions based on failure patterns.

        Args:
            mismatch_details: Parsed mismatch details
            vcd_data: Extracted VCD data

        Returns:
            List of debugging suggestions
        """
        suggestions = []

        if not mismatch_details:
            return suggestions

        # Suggestions based on timing
        first_times = [d['first_mismatch_time'] for d in mismatch_details]
        min_time = min(first_times) if first_times else 0

        if min_time < 10:
            suggestions.append("Check if outputs need explicit initial values (e.g., initial q = 0;)")
            suggestions.append("Verify reset behavior and initialization logic")

        # Suggestions based on failure rate
        if mismatch_details and 'mismatch_percentage' in mismatch_details[0]:
            pct = mismatch_details[0]['mismatch_percentage']
            if pct < 5:
                suggestions.append("Focus on edge cases and boundary conditions")
                suggestions.append("Check behavior at initialization and reset")
            elif pct > 70:
                suggestions.append("Review core logic - may have fundamental error in implementation")
                suggestions.append("Double-check understanding of specification")

        # Suggestions based on signal names
        signal_names = [d['signal'] for d in mismatch_details]
        if any('disc' in s or 'flag' in s or 'err' in s for s in signal_names):
            suggestions.append("FSM output issue detected - verify state transitions and output logic")

        # Add VCD-based suggestions
        if vcd_data and vcd_data.get('signals_at_mismatches'):
            suggestions.append("Check the waveform excerpt above to understand signal relationships")
            suggestions.append("Compare actual signal values at mismatch time with expected behavior")

        return suggestions
