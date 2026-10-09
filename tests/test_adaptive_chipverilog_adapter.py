from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "skills" / "ace-rtl" / "scripts"
import sys

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ace_cvdp_native.rtl_search import chipverilog_adapter as ADAPTER
from ace_cvdp_native.rtl_search.contracts import CandidateArtifact
from ace_cvdp_native.rtl_search.node import EDAStage
from ace_cvdp_native.sv import normalize_generated_sv
from ace_rtl_agent.utils.chipverilog_integration import discover_chipverilog_tasks
from extract_solution import extract_code


class ChipVerilogSourcePreservationTest(unittest.TestCase):
    CORDIC_STYLE_SOURCE = """Model preamble that is not HDL.
`timescale 1ns/1ps
`ifndef XY_BITS
`define XY_BITS 16
`endif

function [16:0] tanangle;
    input integer i;
    tanangle = i;
endfunction

module cordic (
    input wire signed [`XY_BITS:0] x_i,
    output wire signed [`XY_BITS:0] x_o
);
    assign x_o = x_i;
endmodule
`default_nettype wire
"""

    def test_extract_code_preserves_compilation_unit_preamble_and_tail(self):
        extracted = extract_code(self.CORDIC_STYLE_SOURCE)
        self.assertTrue(extracted.startswith("`timescale 1ns/1ps"))
        self.assertIn("`define XY_BITS 16", extracted)
        self.assertIn("function [16:0] tanangle", extracted)
        self.assertIn("module cordic", extracted)
        self.assertTrue(extracted.endswith("`default_nettype wire"))
        self.assertNotIn("Model preamble", extracted)

    def test_normalizer_preserves_compilation_unit_preamble_and_tail(self):
        normalized, _actions = normalize_generated_sv(self.CORDIC_STYLE_SOURCE)
        self.assertTrue(normalized.startswith("`timescale 1ns/1ps"))
        self.assertIn("`define XY_BITS 16", normalized)
        self.assertIn("function [16:0] tanangle", normalized)
        self.assertIn("module cordic", normalized)
        self.assertTrue(normalized.rstrip().endswith("`default_nettype wire"))
        self.assertNotIn("Model preamble", normalized)


class ChipVerilogEvidenceMappingTest(unittest.TestCase):
    @staticmethod
    def _result(status, verdict, formal_status, *, passed=False, valid=True, **extra):
        feedback = {
            "status": status,
            "verdict": verdict,
            "failure_stage": status.upper(),
            "reason": extra.pop("reason", status),
            "formal_status": formal_status,
            **extra,
        }
        return {
            "passed": passed,
            "details": {"candidate_contract_valid": valid},
            "sanitized_feedback": feedback,
        }

    def test_complete_proven_pass_is_the_only_pass(self):
        evidence = ADAPTER.evidence_from_chipverilog_result(
            self._result("pass", "pass", "equivalent", passed=True)
        )
        self.assertEqual(evidence.stage, EDAStage.PASS)
        self.assertTrue(evidence.passed)
        self.assertEqual((evidence.passed_subtests, evidence.total_subtests), (1, 1))

    def test_nominal_pass_with_unknown_formal_status_fails_closed(self):
        evidence = ADAPTER.evidence_from_chipverilog_result(
            self._result("pass", "pass", "unknown", passed=True)
        )
        self.assertEqual(evidence.stage, EDAStage.INFRASTRUCTURE)
        self.assertFalse(evidence.passed)
        self.assertEqual(evidence.failure_class, "evaluator_unknown")

    def test_explicit_self_checking_simulation_pass_is_a_pass_path(self):
        evidence = ADAPTER.evidence_from_chipverilog_result(
            self._result(
                "pass",
                "pass",
                "skip",
                passed=True,
                proof_type="simulation_tb",
                flow="simulation_tb",
            )
        )
        self.assertEqual(evidence.stage, EDAStage.PASS)
        self.assertTrue(evidence.passed)
        self.assertIn("self-checking simulation", evidence.failure_signature)

    def test_evaluator_unknown_is_not_candidate_failure_or_pass(self):
        evidence = ADAPTER.evidence_from_chipverilog_result(
            self._result(
                "equivalence_timeout",
                "unknown",
                "unknown",
                reason="solver timeout",
            )
        )
        self.assertEqual(evidence.stage, EDAStage.INFRASTRUCTURE)
        self.assertFalse(evidence.passed)
        self.assertEqual(evidence.total_subtests, 0)

    def test_known_failures_map_to_hard_eda_stages(self):
        cases = {
            "invalid_candidate": EDAStage.INCOMPLETE,
            "compile_fail": EDAStage.COMPILE,
            "interface_fail": EDAStage.ELABORATE,
            "function_fail": EDAStage.SIMULATION,
        }
        for status, expected in cases.items():
            with self.subTest(status=status):
                evidence = ADAPTER.evidence_from_chipverilog_result(
                    self._result(status, "fail", "skip", valid=status != "invalid_candidate")
                )
                self.assertEqual(evidence.stage, expected)
                self.assertFalse(evidence.passed)

    def test_private_fields_are_dropped_from_public_evidence(self):
        result = self._result("function_fail", "fail", "not_equivalent")
        result["sanitized_feedback"].update(
            {
                "reference_file": "SECRET_GOLDEN_RTL",
                "proof_script": "SECRET_PROOF_SCRIPT",
                "tb_file": "SECRET_HIDDEN_TEST",
            }
        )
        rendered = ADAPTER.evidence_from_chipverilog_result(result).public_feedback
        self.assertNotIn("SECRET_GOLDEN_RTL", rendered)
        self.assertNotIn("SECRET_PROOF_SCRIPT", rendered)
        self.assertNotIn("SECRET_HIDDEN_TEST", rendered)

    def test_private_benchmark_paths_are_redacted_from_diagnostics(self):
        result = self._result(
            "interface_fail",
            "fail",
            "skip",
            reason=(
                "/workspace/chipverilog/Des/mips_16/alu/alu_tb_0.v:38: "
                "error: port `a` is missing"
            ),
        )
        rendered = ADAPTER.evidence_from_chipverilog_result(result).public_feedback
        self.assertNotIn("/workspace/chipverilog", rendered)
        self.assertNotIn("alu_tb_0.v", rendered)
        self.assertIn("<benchmark-private-artifact>:38", rendered)

    def test_relative_testbench_and_hierarchical_references_are_redacted(self):
        result = self._result(
            "function_fail",
            "fail",
            "skip",
            reason=(
                "ID_stage_tb_0_v observed uut.ex_alu_cmd while gold.rom[3] "
                "differed; register_file_tb_0.v also failed"
            ),
            counterexample_summary="dut.secret_state[2] differs from reference.value",
        )
        rendered = ADAPTER.evidence_from_chipverilog_result(result).public_feedback
        for private_text in (
            "ID_stage_tb_0_v",
            "uut.ex_alu_cmd",
            "gold.rom",
            "register_file_tb_0.v",
            "dut.secret_state",
            "reference.value",
        ):
            self.assertNotIn(private_text, rendered)
        self.assertIn("<benchmark-private-testbench>", rendered)
        self.assertIn("<benchmark-private-hierarchical-reference>", rendered)


class ChipVerilogAdaptiveContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(
            os.environ.get(
                "CHIPVERILOG_ROOT",
                "/workspace/chipverilog",
            )
        )
        if not (root / "Src").is_dir() or not (root / "Des").is_dir():
            raise unittest.SkipTest("ChipVerilog checkout unavailable")
        cls.root = root
        cls.task = {task.task_id: task for task in discover_chipverilog_tasks(root)}[
            "signed_shifter"
        ]

    def test_public_task_contract_has_one_artifact_and_no_reference_text(self):
        contract = ADAPTER.task_contract_from_chipverilog_task(self.task)
        self.assertEqual(contract.target_files, ("rtl.sv",))
        self.assertIn("ACE_EXPECTED_TARGET_TOP=signed_shifter", contract.specification)
        self.assertIn("target reference RTL and benchmark testbench are hidden", contract.specification)
        target_dir = str((self.root / "Des" / self.task.family / self.task.task_id).resolve())
        self.assertNotIn(target_dir, contract.specification)

    def test_wrong_or_duplicate_module_is_rejected_without_evaluator_call(self):
        calls = {"test": 0, "cleanup": 0}

        class FakeIntegration:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def setup_datapoint_environment(self, *args, **kwargs):
                del args, kwargs

            def test_rtl_code(self, *args, **kwargs):
                del args, kwargs
                calls["test"] += 1
                raise AssertionError("golden-backed evaluator must not run")

            def cleanup_global_resources(self):
                calls["cleanup"] += 1

        contract = ADAPTER.task_contract_from_chipverilog_task(self.task)
        candidates = (
            CandidateArtifact.from_mapping({"rtl.sv": "module wrong; endmodule"}),
            CandidateArtifact.from_mapping(
                {
                    "rtl.sv": (
                        "module signed_shifter; endmodule\n"
                        "module signed_shifter; endmodule"
                    )
                }
            ),
            CandidateArtifact.from_mapping(
                {
                    "rtl.sv": (
                        "module signed_shifter; endmodule\n"
                        "module helper; endmodule"
                    )
                }
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            ADAPTER, "ChipVerilogIntegration", FakeIntegration
        ):
            verifier = ADAPTER.ChipVerilogVerifier(
                chipverilog_root=self.root,
                benchmark_task=self.task,
                run_dir=Path(temp_dir),
                validate_tools=False,
            )
            try:
                for candidate in candidates:
                    evidence = verifier.verify(contract, candidate)
                    self.assertEqual(evidence.stage, EDAStage.INCOMPLETE)
                    self.assertFalse(evidence.complete_bundle)
            finally:
                verifier.close()
        self.assertEqual(calls["test"], 0)
        self.assertEqual(calls["cleanup"], 1)


if __name__ == "__main__":
    unittest.main()
