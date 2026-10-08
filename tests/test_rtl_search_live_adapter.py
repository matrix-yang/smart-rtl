from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "skills" / "ace-rtl" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ace_cvdp_native.rtl_search.ace_adapter import (  # noqa: E402
    AceLLMAnalyst,
    AceLLMBuilder,
    AceLLMPlanner,
    CallLedger,
)
from ace_cvdp_native.rtl_search.contracts import CandidateArtifact, TaskContract  # noqa: E402
from ace_cvdp_native.rtl_search.contracts import PlanHypothesis  # noqa: E402
from ace_cvdp_native.rtl_search.cvdp_adapter import (  # noqa: E402
    PrebuiltCandidateModel,
    evidence_from_result,
    sanitize_live_row,
    task_contract_from_row,
)
from ace_cvdp_native.rtl_search.engine import RTLSearchEngine  # noqa: E402
from ace_cvdp_native.rtl_search.instructions import GeneralACEInstructions  # noqa: E402
from ace_cvdp_native.rtl_search.live import parse_args as parse_live_args  # noqa: E402
from ace_cvdp_native.rtl_search.node import ActionKind, EDAEvidence, EDAStage  # noqa: E402
from ace_cvdp_native.rtl_search.roles import RoleSet  # noqa: E402
import deepseek_adapter  # noqa: E402


class SequencedRoleCall:
    def __init__(self) -> None:
        self.generator_calls = 0

    def __call__(self, **kwargs):
        component = kwargs["component"]
        if component == "coordinator":
            content = json.dumps(
                {
                    "plans": [
                        {
                            "plan_id": "direct",
                            "summary": "Implement one direct registered datapath.",
                            "distinguishing_assumption": "One state element is sufficient.",
                            "invariants": ["preserve interface"],
                        },
                        {
                            "plan_id": "staged",
                            "summary": "Separate public control and datapath state.",
                            "distinguishing_assumption": "Explicit staging avoids control ambiguity.",
                            "invariants": ["preserve interface"],
                        },
                    ]
                }
            )
        elif component == "generator":
            self.generator_calls += 1
            marker = "MOCK_FAIL" if self.generator_calls == 1 else "MOCK_PASS"
            content = f"module top; // {marker}\nendmodule\n"
        elif component == "reflector":
            content = json.dumps(
                {
                    "failure_class": "functional_unlocalized",
                    "rationale": "The public mismatch does not isolate one local edit.",
                    "repair_obligation": "try the remaining distinct public plan",
                    "localized_failure": False,
                    "plan_conflict": False,
                    "distinct_hypotheses_available": True,
                }
            )
        else:
            raise AssertionError(component)
        out_json = Path(kwargs["out_json"])
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(
                {
                    "provider": "fake",
                    "content": content,
                    "raw": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                }
            ),
            encoding="utf-8",
        )
        return content


class MarkerVerifier:
    def verify(self, task, candidate):
        content = "\n".join(candidate.as_dict().values())
        if "MOCK_PASS" in content:
            return EDAEvidence(
                stage=EDAStage.PASS,
                original_stage="PASS",
                failure_class="none",
                failure_signature="no failure detected",
                complete_bundle=True,
                passed_subtests=1,
                total_subtests=1,
            )
        return EDAEvidence(
            stage=EDAStage.SIMULATION,
            original_stage="CANDIDATE_FUNCTIONAL",
            failure_class="candidate_functional",
            failure_signature="public functional failure",
            complete_bundle=True,
            passed_subtests=0,
            total_subtests=1,
            public_feedback="bounded public mismatch report",
        )


def _analyst_inputs():
    task = TaskContract.from_parts(
        task_id="analyst_contract",
        specification="Implement the public synchronous top module.",
        target_files=("rtl/top.sv",),
    )
    plan = PlanHypothesis("p1", "direct datapath", "one registered stage")
    candidate = CandidateArtifact.from_mapping({"rtl/top.sv": "module top; endmodule"})
    evidence = EDAEvidence(
        stage=EDAStage.SIMULATION,
        original_stage="CANDIDATE_FUNCTIONAL",
        failure_class="candidate_functional",
        failure_signature="public output mismatch",
        complete_bundle=True,
        public_feedback="one bounded mismatch",
    )
    return task, plan, candidate, evidence


class TruncatedThenValidAnalystCall:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        first = len(self.calls) == 1
        if first:
            content = "unfinished private reasoning"
            payload = {
                "content": content,
                "finish_reason": "length",
                "truncated": True,
                "raw": {"usage": {"completion_tokens": 8192}},
            }
        else:
            content = json.dumps(
                {
                    "failure_class": "functional_timing",
                    "rationale": "The public mismatch appears at the visible output cycle.",
                    "repair_obligation": "align data and valid by one cycle",
                    "localized_failure": True,
                    "plan_conflict": False,
                    "distinct_hypotheses_available": False,
                }
            )
            payload = {"content": content, "finish_reason": "stop"}
        Path(kwargs["out_json"]).write_text(json.dumps(payload), encoding="utf-8")
        return content


class AlwaysInvalidAnalystCall:
    def __call__(self, **kwargs):
        content = "not a json diagnosis"
        Path(kwargs["out_json"]).write_text(
            json.dumps({"content": content, "finish_reason": "stop"}), encoding="utf-8"
        )
        return content


class LiveRoleAdapterIntegrationTest(unittest.TestCase):
    def test_live_role_effort_defaults_are_low(self):
        args = parse_live_args(
            ["--dataset", "dataset.jsonl", "--cid", "cid003", "--id", "sample"]
        )
        self.assertEqual(args.planner_reasoning_effort, "low")
        self.assertEqual(args.builder_reasoning_effort, "low")
        self.assertEqual(args.analyst_reasoning_effort, "low")
        self.assertEqual(args.analyst_max_tokens, 8192)

    def test_deepseek_adapter_passes_effort_json_mode_and_keeps_final_content_separate(self):
        request = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "Return JSON."}],
            "max_tokens": 8192,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
        }
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"ok": true}',
                        "reasoning_content": "internal reasoning",
                    },
                }
            ],
            "usage": {"completion_tokens": 12},
        }
        output = io.StringIO()
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False), mock.patch(
            "deepseek_adapter.requests.post", return_value=response
        ) as post, mock.patch("sys.stdin", io.StringIO(json.dumps(request))), contextlib.redirect_stdout(output):
            self.assertEqual(deepseek_adapter.main(), 0)

        body = post.call_args.kwargs["json"]
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["max_tokens"], 8192)
        adapter_output = json.loads(output.getvalue())
        self.assertEqual(adapter_output["content"], '{"ok": true}')
        self.assertEqual(adapter_output["reasoning_content"], "internal reasoning")
        self.assertEqual(adapter_output["finish_reason"], "stop")

    def test_deepseek_adapter_does_not_promote_truncated_reasoning_to_content(self):
        request = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "Return JSON."}],
            "max_tokens": 8192,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "low",
        }
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "", "reasoning_content": "unfinished reasoning"},
                }
            ]
        }
        output = io.StringIO()
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False), mock.patch(
            "deepseek_adapter.requests.post", return_value=response
        ), mock.patch("sys.stdin", io.StringIO(json.dumps(request))), contextlib.redirect_stdout(output):
            self.assertEqual(deepseek_adapter.main(), 0)
        adapter_output = json.loads(output.getvalue())
        self.assertEqual(adapter_output["content"], "")
        self.assertEqual(adapter_output["reasoning_content"], "unfinished reasoning")
        self.assertEqual(adapter_output["finish_reason"], "length")

    def test_analyst_retries_truncated_response_without_thinking(self):
        with tempfile.TemporaryDirectory() as tmp:
            role_call = TruncatedThenValidAnalystCall()
            ledger = CallLedger(Path(tmp), role_call=role_call)
            analyst = AceLLMAnalyst(
                workspace=REPO_ROOT,
                model="fake",
                timeout=10,
                ledger=ledger,
                max_tokens=8192,
                retry_max_tokens=2048,
            )
            task, plan, candidate, evidence = _analyst_inputs()
            diagnosis = analyst.analyze(
                task, plan, candidate, evidence, "ROLE: RTL Failure Analyst"
            )

            self.assertEqual(diagnosis.failure_class, "functional_timing")
            self.assertEqual(analyst.retries, 1)
            self.assertEqual(analyst.fallbacks, 0)
            self.assertEqual(role_call.calls[0]["thinking"], "enabled")
            self.assertEqual(role_call.calls[0]["reasoning_effort"], "low")
            self.assertEqual(role_call.calls[0]["max_tokens"], 8192)
            self.assertEqual(role_call.calls[1]["thinking"], "disabled")
            self.assertIsNone(role_call.calls[1]["reasoning_effort"])
            self.assertEqual(role_call.calls[1]["max_tokens"], 2048)
            self.assertEqual(ledger.records[0]["status"], "truncated")
            self.assertEqual(ledger.records[1]["status"], "ok")

    def test_analyst_repeated_invalid_json_degrades_branch_locally(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = CallLedger(Path(tmp), role_call=AlwaysInvalidAnalystCall())
            analyst = AceLLMAnalyst(
                workspace=REPO_ROOT, model="fake", timeout=10, ledger=ledger
            )
            task, plan, candidate, evidence = _analyst_inputs()
            diagnosis = analyst.analyze(
                task, plan, candidate, evidence, "ROLE: RTL Failure Analyst"
            )

            self.assertEqual(diagnosis.failure_class, "analyst_response_unavailable")
            self.assertIsNone(diagnosis.localized_failure)
            self.assertEqual(analyst.retries, 1)
            self.assertEqual(analyst.fallbacks, 1)
            self.assertEqual(
                [record["status"] for record in ledger.records],
                ["invalid_response", "invalid_response"],
            )

    def test_failed_role_call_is_charged_even_without_token_telemetry(self):
        def failing_call(**kwargs):
            raise RuntimeError("simulated provider exhaustion")

        with tempfile.TemporaryDirectory() as tmp:
            ledger = CallLedger(Path(tmp), role_call=failing_call)
            with self.assertRaisesRegex(RuntimeError, "provider exhaustion"):
                ledger.invoke(
                    workspace=REPO_ROOT,
                    role="planner",
                    component="coordinator",
                    model="fake-model",
                    system="strict json",
                    prompt="plan",
                    timeout=10,
                    temperature=0.1,
                )
            summary = ledger.summary()
            self.assertEqual(summary["calls"], 1)
            self.assertEqual(summary["failed_calls"], 1)
            self.assertEqual(summary["calls_with_token_telemetry"], 0)

    def test_generic_roles_drive_the_real_engine_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            fake = SequencedRoleCall()
            ledger = CallLedger(run_dir, role_call=fake)
            task = TaskContract.from_parts(
                task_id="mock_cid003",
                specification="Implement the public top module.",
                target_files=("rtl/top.sv",),
                immutable_context={"docs/spec.md": "Public behavior only."},
            )
            builder = AceLLMBuilder(
                workspace=REPO_ROOT,
                cid="cid003",
                model="fake-model",
                timeout=10,
                ledger=ledger,
                run_dir=run_dir,
            )
            roles = RoleSet(
                planner=AceLLMPlanner(
                    workspace=REPO_ROOT, model="fake-model", timeout=10, ledger=ledger
                ),
                builder=builder,
                verifier=MarkerVerifier(),
                analyst=AceLLMAnalyst(
                    workspace=REPO_ROOT, model="fake-model", timeout=10, ledger=ledger
                ),
                api_call_counter=lambda: ledger.count,
            )
            result = RTLSearchEngine(
                roles=roles,
                instructions=GeneralACEInstructions(),
                branch_width=2,
                max_steps=4,
                max_plans=2,
            ).run(task)

            self.assertEqual(result.status, "passed")
            self.assertEqual(result.llm_api_calls, 4)
            self.assertEqual(result.action_counts, {"expand": 1, "stop": 1})
            self.assertEqual(ledger.summary()["calls_with_token_telemetry"], 4)
            self.assertEqual(ledger.summary()["usage_totals"]["prompt_tokens"], 40)
            self.assertIn(result.selected_candidate_hash, builder.candidates)

    def test_multifile_builder_requires_and_extracts_full_target_sections(self):
        def multifile_call(**kwargs):
            content = (
                "// TARGET_FILE: rtl/a.sv\nmodule a; endmodule\n"
                "// TARGET_FILE: rtl/b.sv\nmodule b; endmodule\n"
            )
            Path(kwargs["out_json"]).write_text(json.dumps({"content": content}), encoding="utf-8")
            return content

        with tempfile.TemporaryDirectory() as tmp:
            ledger = CallLedger(Path(tmp), role_call=multifile_call)
            builder = AceLLMBuilder(
                workspace=REPO_ROOT,
                cid="cid003",
                model="fake",
                timeout=10,
                ledger=ledger,
                run_dir=Path(tmp),
            )
            task = TaskContract.from_parts(
                task_id="multi",
                specification="Implement both modules.",
                target_files=("rtl/a.sv", "rtl/b.sv"),
            )
            artifact = builder.build(
                task,
                PlanHypothesis("p1", "two complete modules", "separate modules"),
                "ROLE: RTL Builder",
                ActionKind.EXPAND,
                None,
                None,
            )
            self.assertEqual(set(artifact.as_dict()), {"rtl/a.sv", "rtl/b.sv"})
            self.assertEqual(artifact.contract_issues(task), ())


class CVDPLiveBoundaryTest(unittest.TestCase):
    def test_reference_patch_bodies_are_erased_and_initial_target_is_public_spec(self):
        row = {
            "id": "cvdp_agentic_demo_0001",
            "categories": ["cid003", "hard"],
            "prompt": "Repair the public target.",
            "context": {
                "rtl/top.sv": "module top; // buggy public input\nendmodule",
                "docs/spec.md": "public spec",
            },
            "patch": {"rtl/top.sv": "SECRET_REFERENCE_BODY"},
        }
        sanitized = sanitize_live_row(row)
        task = task_contract_from_row(sanitized)

        self.assertEqual(sanitized["patch"], {"rtl/top.sv": ""})
        self.assertEqual(sanitized["output"]["context"], {"rtl/top.sv": ""})
        self.assertNotIn("SECRET_REFERENCE_BODY", json.dumps(sanitized))
        self.assertIn("buggy public input", task.specification)
        self.assertEqual(task.context_map(), {"docs/spec.md": "public spec"})

    def test_prebuilt_model_injects_exact_single_and_multi_file_contracts(self):
        single = CandidateArtifact.from_mapping({"rtl/top.sv": "module top; endmodule"})
        payload, ok = PrebuiltCandidateModel(single, ("rtl/top.sv",)).prompt(
            "public", files=["rtl/top.sv"]
        )
        self.assertTrue(ok)
        self.assertEqual(payload["direct_text"], "module top; endmodule")

        multi = CandidateArtifact.from_mapping(
            {"rtl/a.sv": "module a; endmodule", "rtl/b.sv": "module b; endmodule"}
        )
        payload, ok = PrebuiltCandidateModel(multi, ("rtl/a.sv", "rtl/b.sv")).prompt(
            "public", files=["rtl/a.sv", "rtl/b.sv"]
        )
        self.assertTrue(ok)
        self.assertEqual([next(iter(item)) for item in payload["code"]], ["rtl/a.sv", "rtl/b.sv"])
        with self.assertRaisesRegex(ValueError, "differ"):
            PrebuiltCandidateModel(multi, ("rtl/a.sv", "rtl/b.sv")).prompt(
                "public", files=["rtl/a.sv"]
            )

    def test_evaluator_mapping_is_fail_closed(self):
        passed = evidence_from_result({"errors": 0, "tests": [{"result": 0}]})
        functional = evidence_from_result({"errors": 1, "tests": [{"result": 1}]})
        preflight = evidence_from_result(
            {
                "errors": 1,
                "preflight_failures": ["bad module"],
                "tests": [{"result": 1}],
            }
        )
        self.assertIs(passed.stage, EDAStage.PASS)
        self.assertIs(functional.stage, EDAStage.SIMULATION)
        self.assertIs(preflight.stage, EDAStage.INCOMPLETE)
        self.assertFalse(preflight.complete_bundle)


if __name__ == "__main__":
    unittest.main()
