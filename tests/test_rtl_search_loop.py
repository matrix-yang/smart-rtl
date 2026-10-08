from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "skills" / "ace-rtl" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ace_cvdp_native.rtl_search import (  # noqa: E402
    CandidateArtifact,
    GeneralACEInstructions,
    RTLSearchEngine,
    TaskContract,
)
from ace_cvdp_native.rtl_search.scripted import make_scripted_roles  # noqa: E402
from ace_cvdp_native.rtl_search.smoke import run_smoke  # noqa: E402


class GeneralRoleContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.task = TaskContract.from_parts(
            task_id="generic_task",
            specification="Implement a synchronous synthesizable public RTL interface.",
            target_files=("rtl/top.sv",),
            immutable_context={"docs/spec.md": "public context"},
        )

    def test_general_instructions_have_ace_authority_boundaries_without_sample_fixes(self):
        provider = GeneralACEInstructions()
        text = "\n".join(
            [
                provider.planner(self.task, (), 2),
                provider.director(self.task),
            ]
        ).lower()
        for forbidden in ("dynamic_equalizer", "phase_rotation", "cid003", "chipverilog"):
            self.assertNotIn(forbidden, text)
        self.assertIn("do not emit rtl", text)
        self.assertIn("do not write rtl", text)
        self.assertIn("private tests", text)

    def test_task_contract_separates_targets_from_immutable_context(self):
        with self.assertRaisesRegex(ValueError, "cannot also be immutable context"):
            TaskContract.from_parts(
                task_id="bad",
                specification="public spec",
                target_files=("rtl/top.sv",),
                immutable_context={"rtl/top.sv": "reference content"},
            )

    def test_candidate_contract_rejects_missing_extra_and_empty_targets(self):
        missing = CandidateArtifact.from_mapping({})
        extra = CandidateArtifact.from_mapping(
            {"rtl/top.sv": "module top; endmodule", "tb/top_tb.sv": "module top_tb; endmodule"}
        )
        empty = CandidateArtifact.from_mapping({"rtl/top.sv": ""})
        self.assertIn("missing declared target file", missing.contract_issues(self.task)[0])
        self.assertTrue(any("undeclared generated file" in issue for issue in extra.contract_issues(self.task)))
        self.assertIn("empty generated target file", empty.contract_issues(self.task)[0])


class FullMechanismLoopTest(unittest.TestCase):
    def test_scripted_loop_exercises_full_control_flow(self):
        payload = run_smoke()
        self.assertTrue(payload["acceptance"]["passed"])
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["plans_created"], 2)
        self.assertEqual(payload["llm_api_calls"], 0)
        self.assertEqual(
            set(payload["action_counts"]),
            {"refine", "backtrack", "expand", "stop"},
        )
        self.assertGreaterEqual(payload["contract_rejected_candidates"], 1)

    def test_backtrack_changes_actual_parent_of_next_candidate(self):
        payload = run_smoke()
        structural = next(
            step
            for step in payload["steps"]
            if step["decision_reason"] == "structural_regression"
        )
        following = next(
            step for step in payload["steps"] if step["step"] == structural["step"] + 1
        )
        self.assertEqual(following["parent_id"], structural["destination_id"])
        self.assertEqual(following["producing_action"], "expand")

    def test_general_instruction_provider_is_wired_into_roles(self):
        roles = make_scripted_roles()
        task = TaskContract.from_parts(
            task_id="instruction_wiring",
            specification="Implement the public synchronous RTL contract.",
            target_files=("rtl/top.sv",),
        )
        engine = RTLSearchEngine(
            roles=roles,
            instructions=GeneralACEInstructions(),
            branch_width=2,
            max_steps=8,
        )
        result = engine.run(task)
        self.assertEqual(result.status, "passed")
        self.assertTrue(roles.planner.instructions)
        self.assertTrue(roles.builder.instructions)
        self.assertTrue(roles.analyst.instructions)
        self.assertTrue(all("ROLE: RTL Builder" in text for text in roles.builder.instructions))
        self.assertTrue(all("ROLE: RTL Failure Analyst" in text for text in roles.analyst.instructions))


if __name__ == "__main__":
    unittest.main()
