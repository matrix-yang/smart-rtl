# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ChipBench Integration - Interfaces with the ChipBench Verilog Gen benchmark.

Extends VerilogEvalIntegration to handle ChipBench's multi-dataset directory
structure (dataset_self_contain, dataset_not_self_contain, dataset_cpu_ip)
while reusing the identical compile/simulate/parse pipeline.
"""

from pathlib import Path
from typing import Dict, Any, List

from .verilogeval_integration import VerilogEvalIntegration


VERILOG_GEN_DATASETS = [
    "dataset_self_contain",
    "dataset_not_self_contain",
    "dataset_cpu_ip",
]


class ChipBenchIntegration(VerilogEvalIntegration):
    """
    Integrates with ChipBench Verilog Gen benchmarks.

    ChipBench uses the same file triple as VerilogEval
    ({stem}_prompt.txt, {stem}_test.sv, {stem}_ref.sv) and the same
    TopModule/RefModule testbench pattern, but spreads problems across
    three subdirectories under 'Verilog Gen/'.
    """

    def __init__(self, chipbench_path: str, logger, max_retries: int = 3, initial_backoff: float = 5.0):
        self.chipbench_path = Path(chipbench_path)
        self.verilog_gen_path = self.chipbench_path / "Verilog Gen"
        self.logger = logger
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

        self.current_problem = None
        self.current_prompt = None
        self.current_testbench = None
        self.current_ref_solution = None

        # Build mapping: problem_name -> dataset directory path
        self._problem_dir_map: Dict[str, Path] = {}
        self._scan_datasets()

        self._validate_environment()

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _validate_environment(self):
        """Validate ChipBench paths and iverilog availability."""
        if not self.chipbench_path.exists():
            raise FileNotFoundError(f"ChipBench path not found: {self.chipbench_path}")
        if not self.verilog_gen_path.exists():
            raise FileNotFoundError(f"ChipBench 'Verilog Gen' directory not found: {self.verilog_gen_path}")
        if not self._problem_dir_map:
            raise FileNotFoundError(
                f"No ChipBench problems found under {self.verilog_gen_path}. "
                f"Expected subdirectories: {VERILOG_GEN_DATASETS}"
            )

        # Reuse parent's iverilog check (we just need the subprocess part)
        import subprocess
        try:
            result = subprocess.run(
                ["iverilog", "-V"],
                capture_output=True,
                text=True,
                timeout=5,
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

        self.logger.info(
            f"ChipBench environment validation completed - "
            f"{len(self._problem_dir_map)} problems across {len(VERILOG_GEN_DATASETS)} datasets"
        )

    def get_problem_list(self) -> List[str]:
        """Return sorted list of all Verilog Gen problem stems."""
        return sorted(self._problem_dir_map.keys())

    def load_problem(self, problem_name: str) -> Dict[str, Any]:
        """Load prompt, testbench, and reference from the correct dataset directory."""
        dataset_dir = self._problem_dir_map.get(problem_name)
        if dataset_dir is None:
            available = ", ".join(sorted(self._problem_dir_map.keys())[:10])
            raise FileNotFoundError(
                f"ChipBench problem '{problem_name}' not found. "
                f"Available problems (first 10): {available} ..."
            )

        prompt_file = dataset_dir / f"{problem_name}_prompt.txt"
        test_file = dataset_dir / f"{problem_name}_test.sv"
        ref_file = dataset_dir / f"{problem_name}_ref.sv"

        for path, label in [(prompt_file, "Prompt"), (test_file, "Test"), (ref_file, "Reference")]:
            if not path.exists():
                raise FileNotFoundError(f"{label} file not found: {path}")

        with open(prompt_file, "r") as f:
            prompt = f.read()
        with open(test_file, "r") as f:
            testbench = f.read()
        with open(ref_file, "r") as f:
            ref_solution = f.read()

        self.current_problem = problem_name
        self.current_prompt = prompt
        self.current_testbench = testbench
        self.current_ref_solution = ref_solution

        self.logger.info(f"Loaded ChipBench problem: {problem_name} (from {dataset_dir.name})")
        self.logger.info(f"  Prompt: {len(prompt)} chars")
        self.logger.info(f"  Testbench: {len(testbench)} chars")
        self.logger.info(f"  Reference (for compilation only): {len(ref_solution)} chars")

        return {
            "problem_name": problem_name,
            "prompt": prompt,
            "testbench": testbench,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_datasets(self):
        """Discover all problems across the three Verilog Gen dataset directories."""
        for dataset_name in VERILOG_GEN_DATASETS:
            dataset_dir = self.verilog_gen_path / dataset_name
            if not dataset_dir.exists():
                self.logger.warning(f"ChipBench dataset directory not found: {dataset_dir}")
                continue

            for prompt_file in sorted(dataset_dir.glob("Prob*_prompt.txt")):
                stem = prompt_file.stem.replace("_prompt", "")
                if stem in self._problem_dir_map:
                    self.logger.warning(
                        f"Duplicate problem stem '{stem}' found in {dataset_name} "
                        f"(already registered from {self._problem_dir_map[stem].name}); skipping"
                    )
                    continue
                self._problem_dir_map[stem] = dataset_dir

        self.logger.info(
            f"Scanned ChipBench datasets - found {len(self._problem_dir_map)} problems"
        )
