from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "skills"
    / "ace-rtl"
    / "scripts"
    / "ace_rtl_agent"
    / "utils"
    / "chipverilog_integration.py"
)
SPEC = importlib.util.spec_from_file_location("ace_chipverilog_integration_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

SCRIPTS_ROOT = REPO_ROOT / "skills" / "ace-rtl" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ace_rtl_agent.focused_task_detection import FocusedTaskDetectionMixin
from ace_rtl_agent.fresh_start_coordinator import FreshStartCoordinator
from ace_rtl_agent.utils import llm_client as LLM_MODULE


class ChipVerilogFeedbackTest(unittest.TestCase):
    def setUp(self):
        self.integration = object.__new__(MODULE.ChipVerilogIntegration)

    def test_private_report_fields_are_not_forwarded(self):
        row = {
            "status": "function_fail",
            "flow": "equivalence",
            "reason": "proof failed",
            "counterexample_summary": "first mismatch",
            "reference_file": "SECRET_REFERENCE_RTL",
            "tb_file": "SECRET_TESTBENCH",
            "proof_script": "read_verilog SECRET_REFERENCE_RTL",
        }
        feedback = self.integration._sanitized_feedback(row)
        rendered = str(feedback)
        self.assertNotIn("SECRET_REFERENCE_RTL", rendered)
        self.assertNotIn("SECRET_TESTBENCH", rendered)
        self.assertEqual(feedback["verdict"], "fail")

    def test_unknown_status_is_not_ranked_as_candidate_failure(self):
        result = self.integration._result_from_row(
            {
                "status": "equivalence_timeout",
                "flow": "equivalence",
                "reason": "solver timeout",
                "formal_status": "unknown",
            },
            elapsed=2.5,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["details"]["verdict"], "unknown")
        self.assertEqual(result["details"]["mismatches"], -1)
        self.assertIn("inconclusive", result["error_report"])

    def test_hard_gate_progress_order(self):
        self.assertLess(
            MODULE.STATUS_PROGRESS["function_fail"],
            MODULE.STATUS_PROGRESS["interface_fail"],
        )
        self.assertLess(
            MODULE.STATUS_PROGRESS["interface_fail"],
            MODULE.STATUS_PROGRESS["compile_fail"],
        )

    def test_function_feedback_withholds_repeated_private_mismatch_excerpts(self):
        feedback = self.integration._sanitized_feedback(
            {
                "status": "function_fail",
                "reason": "first expected/actual | second expected/actual | third expected/actual",
                "generated_stub_modules": "[]",
            }
        )
        self.assertIn("first expected/actual", feedback["reason"])
        self.assertNotIn("second expected/actual", feedback["reason"])
        self.assertEqual(feedback["generated_stub_modules"], [])


class ChipVerilogCandidateContractTest(unittest.TestCase):
    def test_exact_target_module_is_valid(self):
        contract = MODULE.inspect_candidate_contract(
            "module alu(input logic a, output logic y); assign y = a; endmodule",
            "alu",
        )
        self.assertTrue(contract["valid"])
        self.assertEqual(contract["candidate_top"], "alu")
        self.assertEqual(contract["target_definition_count"], 1)

    def test_testbench_only_candidate_is_rejected(self):
        contract = MODULE.inspect_candidate_contract(
            "module tb_alu; alu dut(); initial $finish; endmodule",
            "alu",
        )
        self.assertFalse(contract["valid"])
        self.assertIn("tb_alu", contract["unexpected_module_names"])
        self.assertTrue(
            any("does not define expected target" in reason for reason in contract["reasons"])
        )

    def test_wrong_top_and_extra_module_are_rejected(self):
        wrong_top = MODULE.inspect_candidate_contract("module adder; endmodule", "alu")
        extra_top = MODULE.inspect_candidate_contract(
            "module alu; endmodule\nmodule tb_alu; endmodule",
            "alu",
        )
        self.assertFalse(wrong_top["valid"])
        self.assertFalse(extra_top["valid"])
        self.assertIn("tb_alu", extra_top["unexpected_module_names"])

    def test_automatic_target_module_is_accepted(self):
        contract = MODULE.inspect_candidate_contract(
            "module automatic alu; endmodule",
            "alu",
        )
        self.assertTrue(contract["valid"])

    def test_commented_target_declaration_does_not_satisfy_contract(self):
        for candidate in (
            "// module alu; endmodule\nmodule tb_alu; endmodule",
            "/* module alu; endmodule */\nmodule tb_alu; endmodule",
        ):
            with self.subTest(candidate=candidate):
                contract = MODULE.inspect_candidate_contract(candidate, "alu")
                self.assertFalse(contract["valid"])
                self.assertEqual(contract["target_definition_count"], 0)


class ChipVerilogTaskRoutingTest(unittest.TestCase):
    def test_focused_debugger_does_not_let_tb_candidate_change_task_identity(self):
        detector = object.__new__(FocusedTaskDetectionMixin)
        specification = (
            f"{MODULE.CHIPVERILOG_TASK_MARKER}\n"
            "The testbench is hidden. Implement synthesizable RTL module alu."
        )
        candidate = "module tb_alu; initial $finish; endmodule"
        self.assertFalse(detector._is_verification_task(specification, candidate))

    def test_fresh_restart_keeps_chipverilog_in_rtl_generation_route(self):
        coordinator = object.__new__(FreshStartCoordinator)
        coordinator.iteration_history = []
        coordinator._llm_extract_high_level_lessons = lambda: ""
        specification = (
            f"{MODULE.CHIPVERILOG_TASK_MARKER}\n"
            "The testbench is hidden and will evaluate module alu."
        )
        prompt = coordinator._create_fresh_generation_prompt(specification)
        self.assertIn("ChipVerilog RTL implementation problem", prompt)
        self.assertNotIn("Test Plan to Testbench Stimulus Generation", prompt)


class ChipVerilogTelemetryTest(unittest.TestCase):
    def test_adapter_wrapper_preserves_content_and_usage(self):
        payload = json.dumps(
            {
                "content": "module alu; endmodule",
                "raw": {
                    "id": "response-1",
                    "model": "deepseek-v4-pro",
                    "choices": [{"finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 101,
                        "completion_tokens": 23,
                        "total_tokens": 124,
                        "completion_tokens_details": {"reasoning_tokens": 7},
                    },
                },
            }
        )
        content, metadata = LLM_MODULE._extract_adapter_response(payload)
        usage = LLM_MODULE._normalize_usage(metadata["usage"])
        self.assertEqual(content, "module alu; endmodule")
        self.assertEqual(metadata["response_id"], "response-1")
        self.assertEqual(metadata["finish_reason"], "stop")
        self.assertEqual(usage["input_tokens"], 101)
        self.assertEqual(usage["output_tokens"], 23)
        self.assertEqual(usage["reasoning_tokens"], 7)
        self.assertEqual(usage["total_tokens"], 124)

    def test_successful_llm_call_writes_normalized_usage_event(self):
        class FakeClient:
            provider = "adapter"
            last_response_metadata = {
                "response_id": "response-2",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 5,
                    "total_tokens": 17,
                },
            }

            @staticmethod
            def prompt(prompt, temperature=None, max_tokens=None):
                del prompt, temperature, max_tokens
                return "module alu; endmodule"

        client = object.__new__(LLM_MODULE.LLMClient)
        client.model = "deepseek-v4-pro"
        client.max_retries = 0
        client.initial_backoff = 0.0
        client.logger = logging.getLogger("llm-telemetry-test")
        client.telemetry_role = "rtl_generator"
        client.client = FakeClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            telemetry_path = Path(temp_dir) / "telemetry.jsonl"
            with mock.patch.dict(
                os.environ,
                {
                    "ACE_RTL_LLM_TELEMETRY_FILE": str(telemetry_path),
                    "ACE_RTL_TASK_ID": "mips_alu",
                },
            ):
                response = client.call_llm("generate alu", temperature=0.2, max_tokens=32)
            event = json.loads(telemetry_path.read_text(encoding="utf-8"))
        self.assertEqual(response, "module alu; endmodule")
        self.assertEqual(event["event"], "llm_call")
        self.assertEqual(event["task_id"], "mips_alu")
        self.assertEqual(event["role"], "rtl_generator")
        self.assertTrue(event["usage_available"])
        self.assertEqual(event["usage"]["total_tokens"], 17)
        self.assertNotIn("generate alu", json.dumps(event))

    def test_task_usage_summary_aggregates_roles_and_failed_calls(self):
        events = [
            {
                "event": "llm_call",
                "role": "rtl_generator",
                "success": True,
                "usage_available": True,
                "latency_seconds": 1.5,
                "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            },
            {
                "event": "llm_call",
                "role": "rtl_generator",
                "success": False,
                "usage_available": False,
                "latency_seconds": 0.5,
            },
            {
                "event": "llm_call",
                "role": "focused_debugger",
                "success": True,
                "usage_available": True,
                "latency_seconds": 2.0,
                "usage": {"input_tokens": 20, "output_tokens": 6, "total_tokens": 26},
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "telemetry.jsonl"
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            summary = MODULE.summarize_llm_telemetry(path)
        self.assertEqual(summary["calls"], 3)
        self.assertEqual(summary["successful_calls"], 2)
        self.assertEqual(summary["failed_calls"], 1)
        self.assertEqual(summary["usage_available_calls"], 2)
        self.assertEqual(summary["input_tokens"], 30)
        self.assertEqual(summary["output_tokens"], 10)
        self.assertEqual(summary["total_tokens"], 40)
        self.assertEqual(summary["by_role"]["rtl_generator"]["calls"], 2)


class ChipVerilogDiscoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("CHIPVERILOG_ROOT")
        fallback = Path("/workspace/chipverilog")
        cls.root = Path(configured).resolve() if configured else fallback
        if not (cls.root / "Src").is_dir() or not (cls.root / "Des").is_dir():
            raise unittest.SkipTest("ChipVerilog checkout is not available")
        cls.tasks = MODULE.discover_chipverilog_tasks(cls.root)

    def test_discovers_expected_suite_and_module_names(self):
        by_id = {task.task_id: task for task in self.tasks}
        self.assertEqual(len(self.tasks), 64)
        self.assertEqual(by_id["mips_alu"].target_top, "alu")
        self.assertEqual(by_id["fpu"].target_top, "fpu")
        self.assertEqual(by_id["fpu_addsub_pipeline"].target_top, "fpu_addsub")

    def test_target_reference_directory_is_never_a_dependency(self):
        for task in self.tasks:
            reference_leaf = (
                task.task_id[len("mips_") :]
                if task.family == "mips_16" and task.task_id.startswith("mips_")
                else task.task_id
            )
            target_dir = (self.root / "Des" / task.family / reference_leaf).resolve()
            self.assertTrue(
                all(path.resolve().parent != target_dir for path in task.dependency_files),
                msg=f"target reference leaked into dependency pack for {task.task_id}",
            )

    def test_prompt_declares_immutable_rtl_task_contract(self):
        by_id = {task.task_id: task for task in self.tasks}
        prompt = MODULE.render_chipverilog_prompt(by_id["mips_alu"])
        self.assertIn(MODULE.CHIPVERILOG_TASK_MARKER, prompt)
        self.assertIn("ACE_EXPECTED_TARGET_TOP=alu", prompt)
        self.assertIn("never a testbench", prompt)

    def test_tb_only_candidate_is_rejected_before_evaluator_subprocess(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            integration = MODULE.ChipVerilogIntegration(
                str(self.root),
                logging.getLogger("chipverilog-contract-test"),
                validate_tools=False,
            )
            integration.setup_datapoint_environment(
                "mips_alu",
                temp_dir,
                "contract-test-model",
            )
            try:
                with mock.patch.object(MODULE.subprocess, "run") as run:
                    result = integration.test_rtl_code(
                        "mips_alu",
                        "module tb_alu; alu dut(); initial $finish; endmodule",
                    )
                run.assert_not_called()
                self.assertEqual(result["details"]["status"], "invalid_candidate")
                self.assertFalse(result["details"]["candidate_contract_valid"])
                invocation_path = Path(result["details"]["evaluator_run_dir"]) / "evaluator_invocation.json"
                invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
                self.assertTrue(invocation["skipped"])
            finally:
                integration.cleanup_global_resources()


if __name__ == "__main__":
    unittest.main()
