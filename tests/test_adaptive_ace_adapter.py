import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = WORKSPACE / "skills" / "ace-rtl" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ace_cvdp_native.rtl_search.adaptive_ace_adapter import (  # noqa: E402
    AceAdaptiveAnalyst,
    AceAdaptiveCandidateJudge,
    AceAdaptiveDebater,
    AceAdaptiveDirector,
    AceAdaptivePlanner,
    _preserve_refine_compilation_unit_scaffold,
)
from ace_cvdp_native.rtl_search.ace_adapter import (  # noqa: E402
    CallLedger,
    ResourceBudgetExceeded,
)
from ace_cvdp_native.rtl_search.adaptive_contracts import (  # noqa: E402
    ActionCard,
    AdaptiveActionKind,
    BranchStatus,
    CandidateRecord,
    EvidenceRecord,
    SearchSnapshot,
)
from ace_cvdp_native.rtl_search.contracts import (  # noqa: E402
    CandidateArtifact,
    PlanHypothesis,
    TaskContract,
)
from ace_cvdp_native.rtl_search.node import EDAEvidence, EDAStage  # noqa: E402


class StubLedger:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.invalid = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def mark_last_invalid(self, error):
        self.invalid.append(str(error))


def _task():
    return TaskContract.from_parts(
        task_id="general_task",
        specification="Implement a public synchronous interface.",
        target_files=("dut.sv",),
        immutable_context={"public.md": "public timing notes"},
    )


def _snapshot(plan, *, candidates=(), evidence=(), round_index=1):
    return SearchSnapshot(
        round_index=round_index,
        plans=(plan,),
        candidates=tuple(candidates),
        evidence=tuple(evidence),
        knowledge=(),
        prior_round_events=(),
        physical_worker_limit=8,
        available_actions=tuple(action.value for action in AdaptiveActionKind),
    )


def test_adaptive_planner_selects_its_own_plan_count(tmp_path):
    ledger = StubLedger(
        [
            json.dumps(
                {
                    "adequacy_rationale": "the public contract is unambiguous",
                    "plans": [
                        {
                            "plan_id": "direct",
                            "summary": "direct implementation",
                            "distinguishing_assumption": "single clear timing interpretation",
                            "invariants": ["output changes only on the active edge"],
                        }
                    ],
                }
            )
        ]
    )
    planner = AceAdaptivePlanner(
        workspace=WORKSPACE,
        model="mock",
        timeout=1,
        ledger=ledger,
    )

    plans = planner.propose(_task(), "planner skill", (), ())

    assert [plan.plan_id for plan in plans] == ["direct"]
    prompt = ledger.calls[0]["prompt"]
    assert "Choose the number of plans from semantic uncertainty" in prompt
    assert "Return exactly" not in prompt
    assert "requested_count" not in prompt


def test_candidate_judge_receives_parent_evidence_and_candidate_delta():
    ledger = StubLedger(
        [
            json.dumps(
                {
                    "accepted": True,
                    "rationale": "the child changes only the named state partition",
                    "action_fidelity": "the obligation is implemented directly",
                    "semantic_novelty": "the output behavior changes for reset state",
                    "state_preservation": "the public interface and timing are unchanged",
                    "rtl_risk": "no duplicate declaration or driver is visible",
                    "revision_requirements": [],
                }
            )
        ]
    )
    judge = AceAdaptiveCandidateJudge(
        workspace=WORKSPACE,
        model="mock",
        timeout=1,
        ledger=ledger,
    )
    plan = PlanHypothesis("p1", "registered path", "reset state is observable")
    parent = CandidateArtifact.from_mapping(
        {"dut.sv": "module dut; assign out = 1'b0; endmodule\n"}
    )
    child = CandidateArtifact.from_mapping(
        {"dut.sv": "module dut; assign out = reset ? 1'b0 : in; endmodule\n"}
    )
    evidence = EDAEvidence(
        EDAStage.SIMULATION,
        "CANDIDATE_FUNCTIONAL",
        "functional_mismatch",
        "output mismatch after reset deassertion",
        True,
        public_feedback="public mismatch after reset deassertion",
    )
    snapshot = _snapshot(
        plan,
        candidates=(
            CandidateRecord(
                "c0001",
                "p1",
                (),
                "seed",
                parent,
                BranchStatus.ACTIVE,
                1,
                "e0001",
            ),
        ),
        evidence=(EvidenceRecord("e0001", "c0001", evidence),),
        round_index=2,
    )
    action = ActionCard(
        "refine-reset",
        AdaptiveActionKind.REFINE,
        source_node_ids=("c0001",),
        plan_id="p1",
        hypothesis="reset release behavior causes the mismatch",
        obligation="change only reset release output behavior",
        semantic_delta="drive input data after reset is deasserted",
        expected_observation="the post-reset mismatch disappears",
        falsifier="the same post-reset mismatch remains",
        evidence_refs=("e0001",),
    )

    judgement = judge.review_candidate(
        _task(), plan, action, (parent,), child, "judge skill", snapshot
    )

    assert judgement.accepted is True
    assert ledger.calls[0]["role"] == "adaptive_candidate_judge"
    prompt = ledger.calls[0]["prompt"]
    assert "PARENT EDA EVIDENCE" in prompt
    assert "output mismatch after reset deassertion" in prompt
    assert "CANDIDATE DELTA FROM PRIMARY PARENT" in prompt
    assert "drive input data after reset is deasserted" in prompt


def test_adaptive_planner_retries_invalid_final_response_compactly():
    ledger = StubLedger(
        [
            "{}",
            json.dumps(
                {
                    "adequacy_rationale": "one interpretation dominates",
                    "plans": [
                        {
                            "plan_id": "direct",
                            "summary": "direct implementation",
                            "distinguishing_assumption": "public contract is unambiguous",
                            "invariants": ["respect the declared interface"],
                        }
                    ],
                }
            ),
        ]
    )
    planner = AceAdaptivePlanner(
        workspace=WORKSPACE,
        model="mock",
        timeout=1,
        ledger=ledger,
    )

    plans = planner.propose(_task(), "planner skill", (), ())

    assert [plan.plan_id for plan in plans] == ["direct"]
    assert ledger.invalid and "at least one plan" in ledger.invalid[0]
    assert [call["role"] for call in ledger.calls] == [
        "adaptive_planner",
        "adaptive_planner_retry",
    ]
    assert ledger.calls[1]["thinking"] == "disabled"


def test_adaptive_planner_revision_receives_immutable_public_context():
    ledger = StubLedger(
        [
            json.dumps(
                {
                    "plan_id": "revised",
                    "summary": "context-grounded revision",
                    "distinguishing_assumption": "public timing notes are authoritative",
                    "invariants": ["preserve the declared timing"],
                }
            )
        ]
    )
    planner = AceAdaptivePlanner(
        workspace=WORKSPACE,
        model="mock",
        timeout=1,
        ledger=ledger,
    )
    base = PlanHypothesis("p1", "base", "initial assumption")
    action = ActionCard(
        "review",
        AdaptiveActionKind.REVIEW_SPEC,
        hypothesis="the public context resolves timing",
        obligation="review the context",
        expected_observation="one timing interpretation remains",
        falsifier="the context is ambiguous",
    )

    revised = planner.revise(_task(), "planner skill", action, _snapshot(base))

    assert revised.plan_id == "revised"
    prompt = ledger.calls[0]["prompt"]
    assert "IMMUTABLE PUBLIC CONTEXT" in prompt
    assert "public timing notes" in prompt


def test_adaptive_director_parses_variable_dispatch_without_worker_fill_rule():
    response = {
        "assessment": {
            "spec_certainty": "medium",
            "plan_concentration": "split",
            "implementation_risk": "medium",
            "verification_observability": "low",
            "rationale": "two incompatible timing assumptions remain",
        },
        "rationale": "test both materially different hypotheses",
        "dispatch": [
            {
                "action_id": "seed-a",
                "kind": "seed_plan",
                "source_node_ids": [],
                "plan_id": "p1",
                "hypothesis": "stateful",
                "obligation": "implement stateful path",
                "expected_observation": "stateful traces improve",
                "falsifier": "mismatch remains unchanged",
                "evidence_refs": [],
                "knowledge_refs": [],
            },
            {
                "action_id": "seed-b",
                "kind": "seed_plan",
                "source_node_ids": [],
                "plan_id": "p2",
                "hypothesis": "stateless",
                "obligation": "implement stateless path",
                "expected_observation": "direct traces improve",
                "falsifier": "mismatch remains unchanged",
                "evidence_refs": [],
                "knowledge_refs": [],
            },
        ],
        "knowledge_decisions": [],
    }
    ledger = StubLedger([json.dumps(response)])
    director = AceAdaptiveDirector(
        workspace=WORKSPACE,
        model="mock",
        timeout=1,
        ledger=ledger,
    )
    plan = PlanHypothesis("p1", "stateful", "state assumption")
    snapshot = SearchSnapshot(
        round_index=1,
        plans=(plan, PlanHypothesis("p2", "stateless", "no-state assumption")),
        candidates=(),
        evidence=(),
        knowledge=(),
        prior_round_events=(),
        physical_worker_limit=8,
        available_actions=("seed_plan", "stop"),
    )

    decision = director.decide(_task(), "director skill", snapshot)

    assert [action.plan_id for action in decision.dispatch] == ["p1", "p2"]
    prompt = ledger.calls[0]["prompt"]
    assert "fill worker slots merely" in prompt
    assert '"physical_worker_limit": 8' in prompt


def test_adaptive_director_repairs_one_unavailable_action():
    def response(kind, action_id):
        return json.dumps(
            {
                "assessment": {
                    "spec_certainty": "high",
                    "plan_concentration": "high",
                    "implementation_risk": "medium",
                    "verification_observability": "medium",
                    "rationale": "use the supplied public evidence",
                },
                "rationale": "choose one legal next action",
                "dispatch": [
                    {
                        "action_id": action_id,
                        "kind": kind,
                        "source_node_ids": [],
                        "plan_id": "p1",
                        "hypothesis": "exercise one implementation",
                        "obligation": "produce one complete candidate",
                        "expected_observation": "candidate reaches verification",
                        "falsifier": "candidate cannot reach verification",
                        "evidence_refs": [],
                        "knowledge_refs": [],
                    }
                ],
                "knowledge_decisions": [],
            }
        )

    ledger = StubLedger(
        [
            response("probe", "unavailable-probe"),
            response("seed_plan", "repaired-seed"),
        ]
    )
    director = AceAdaptiveDirector(
        workspace=WORKSPACE,
        model="mock",
        timeout=1,
        ledger=ledger,
    )
    plan = PlanHypothesis("p1", "direct", "one clear implementation")
    snapshot = SearchSnapshot(
        round_index=1,
        plans=(plan,),
        candidates=(),
        evidence=(),
        knowledge=(),
        prior_round_events=(),
        physical_worker_limit=1,
        available_actions=("seed_plan", "stop"),
    )

    decision = director.decide(_task(), "director skill", snapshot)

    assert decision.dispatch[0].kind is AdaptiveActionKind.SEED_PLAN
    assert ledger.invalid and "probe" in ledger.invalid[0]
    assert [call["role"] for call in ledger.calls] == [
        "adaptive_director",
        "adaptive_director_retry",
    ]
    assert "YOUR PREVIOUS RESPONSE WAS REJECTED" in ledger.calls[1]["prompt"]
    assert "RUNTIME LEGAL ACTIONS" in ledger.calls[0]["prompt"]
    assert '"kind": "seed_plan|stop"' in ledger.calls[0]["prompt"]


def test_refine_restores_required_parent_compilation_unit_scaffold():
    parent = CandidateArtifact.from_mapping(
        {
            "rtl.sv": (
                "`timescale 1ns/1ps\n"
                "`define XY_BITS 16\n"
                "function [16:0] atan_lut; input integer i; atan_lut=i; endfunction\n"
                "module cordic(input [`XY_BITS:0] x, output [`XY_BITS:0] y);\n"
                "assign y=x; endmodule\n"
            )
        }
    )
    child = CandidateArtifact.from_mapping(
        {
            "rtl.sv": (
                "module cordic(input [`XY_BITS:0] x, output [`XY_BITS:0] y);\n"
                "assign y=x+1'b1; endmodule\n"
            )
        }
    )
    action = ActionCard(
        "bounded-refine",
        AdaptiveActionKind.REFINE,
        source_node_ids=("c1",),
        plan_id="p1",
        hypothesis="the output stage is early",
        obligation="change only the output stage",
        semantic_delta="the final stage index changes while the interface remains fixed",
        expected_observation="functional mismatch changes",
        falsifier="the mismatch is unchanged",
    )

    repaired, actions = _preserve_refine_compilation_unit_scaffold(
        action, (parent,), child
    )
    source = repaired.as_dict()["rtl.sv"]

    assert source.startswith("`timescale 1ns/1ps")
    assert "`define XY_BITS 16" in source
    assert "function [16:0] atan_lut" in source
    assert "assign y=x+1'b1" in source
    assert actions[0]["restored_required_macros"] == ["XY_BITS"]


def test_adaptive_director_repairs_empty_dispatch():
    assessment = {
        "spec_certainty": "high",
        "plan_concentration": "high",
        "implementation_risk": "low",
        "verification_observability": "high",
        "rationale": "one implementation is sufficient",
    }
    ledger = StubLedger(
        [
            json.dumps(
                {
                    "assessment": assessment,
                    "rationale": "omitted dispatch accidentally",
                    "dispatch": [],
                    "knowledge_decisions": [],
                }
            ),
            json.dumps(
                {
                    "assessment": assessment,
                    "rationale": "seed the known plan",
                    "dispatch": [
                        {
                            "action_id": "seed",
                            "kind": "seed_plan",
                            "source_node_ids": [],
                            "plan_id": "p1",
                            "hypothesis": "direct implementation",
                            "obligation": "build one candidate",
                            "expected_observation": "verifier evidence is produced",
                            "falsifier": "candidate fails verification",
                            "evidence_refs": [],
                            "knowledge_refs": [],
                        }
                    ],
                    "knowledge_decisions": [],
                }
            ),
        ]
    )
    director = AceAdaptiveDirector(
        workspace=WORKSPACE, model="mock", timeout=1, ledger=ledger
    )
    snapshot = _snapshot(PlanHypothesis("p1", "direct", "clear public contract"))

    decision = director.decide(_task(), "director skill", snapshot)

    assert decision.dispatch[0].kind is AdaptiveActionKind.SEED_PLAN
    assert ledger.invalid and "non-empty list" in ledger.invalid[0]


def test_adaptive_director_repairs_source_free_stop_that_ignores_pass():
    assessment = {
        "spec_certainty": "high",
        "plan_concentration": "high",
        "implementation_risk": "low",
        "verification_observability": "high",
        "rationale": "fail-closed pass exists",
    }
    stop = lambda sources: json.dumps(
        {
            "assessment": assessment,
            "rationale": "deliver the verified candidate",
            "dispatch": [
                {
                    "action_id": "stop",
                    "kind": "stop",
                    "source_node_ids": sources,
                    "plan_id": None,
                    "hypothesis": "",
                    "obligation": "",
                    "expected_observation": "",
                    "falsifier": "",
                    "evidence_refs": ["e1"],
                    "knowledge_refs": [],
                }
            ],
            "knowledge_decisions": [],
        }
    )
    evidence = EDAEvidence(EDAStage.PASS, "PASS", "none", "verified", True)
    artifact = CandidateArtifact.from_mapping({"dut.sv": "module dut; endmodule"})
    candidate = CandidateRecord(
        "c1", "p1", (), "seed", artifact, BranchStatus.PASSED, 1, "e1"
    )
    plan = PlanHypothesis("p1", "direct", "clear contract")
    snapshot = _snapshot(
        plan,
        candidates=(candidate,),
        evidence=(EvidenceRecord("e1", "c1", evidence),),
        round_index=2,
    )
    ledger = StubLedger([stop([]), stop(["c1"])])
    director = AceAdaptiveDirector(
        workspace=WORKSPACE, model="mock", timeout=1, ledger=ledger
    )

    decision = director.decide(_task(), "director skill", snapshot)

    assert decision.dispatch[0].source_node_ids == ("c1",)
    assert "cannot ignore" in ledger.invalid[0]


def test_adaptive_director_repairs_final_round_replan_only_dispatch():
    assessment = {
        "spec_certainty": "medium",
        "plan_concentration": "low",
        "implementation_risk": "high",
        "verification_observability": "medium",
        "rationale": "the safety horizon is exhausted",
    }

    def response(kind, *, plan_id, action_id):
        return json.dumps(
            {
                "assessment": assessment,
                "rationale": "choose a horizon-aware action",
                "dispatch": [
                    {
                        "action_id": action_id,
                        "kind": kind,
                        "source_node_ids": [],
                        "plan_id": plan_id,
                        "hypothesis": "one bounded attempt remains",
                        "obligation": "produce evidence before the horizon",
                        "expected_observation": "new verifier evidence",
                        "falsifier": "no evidence is produced",
                        "evidence_refs": [],
                        "knowledge_refs": [],
                    }
                ],
                "knowledge_decisions": [],
            }
        )

    plan = PlanHypothesis("p1", "direct", "clear contract")
    snapshot = SearchSnapshot(
        round_index=6,
        plans=(plan,),
        candidates=(),
        evidence=(),
        knowledge=(),
        prior_round_events=(),
        physical_worker_limit=1,
        available_actions=("replan", "seed_plan", "stop"),
        max_rounds=6,
    )
    ledger = StubLedger(
        [
            response("replan", plan_id=None, action_id="late-replan"),
            response("seed_plan", plan_id="p1", action_id="final-seed"),
        ]
    )
    director = AceAdaptiveDirector(
        workspace=WORKSPACE, model="mock", timeout=1, ledger=ledger
    )

    decision = director.decide(_task(), "director skill", snapshot)

    assert decision.dispatch[0].kind is AdaptiveActionKind.SEED_PLAN
    assert "final-round dispatch" in ledger.invalid[0]
    assert '"rounds_remaining_including_current": 1' in ledger.calls[0]["prompt"]


def test_adaptive_analyst_receives_candidate_delta_not_repeated_context():
    parent = CandidateArtifact.from_mapping(
        {"dut.sv": "module dut; assign y = a; endmodule\n"}
    )
    candidate = CandidateArtifact.from_mapping(
        {"dut.sv": "module dut; assign y = ~a; endmodule\n"}
    )
    evidence = EDAEvidence(
        stage=EDAStage.SIMULATION,
        original_stage="CANDIDATE_FUNCTIONAL",
        failure_class="functional_mismatch",
        failure_signature="y mismatch",
        complete_bundle=True,
        total_subtests=1,
        public_feedback="mismatch at cycle 2",
    )
    plan = PlanHypothesis("p1", "direct logic", "combinational behavior", ("y follows a",))
    parent_record = CandidateRecord(
        node_id="c0001",
        plan_id="p1",
        parent_node_ids=(),
        producing_action_id="seed",
        artifact=parent,
        status=BranchStatus.ACTIVE,
        round_created=1,
        evidence_id="e0001",
    )
    snapshot = _snapshot(
        plan,
        candidates=(parent_record,),
        evidence=(EvidenceRecord("e0001", "c0001", evidence),),
        round_index=2,
    )
    action = ActionCard(
        "refine",
        AdaptiveActionKind.REFINE,
        source_node_ids=("c0001",),
        plan_id="p1",
        hypothesis="polarity is wrong",
        obligation="correct polarity only",
        semantic_delta="invert y for the input partition that currently mismatches",
        expected_observation="mismatch disappears",
        falsifier="same mismatch remains",
    )
    response = {
        "strengths": ["candidate compiles"],
        "weaknesses": ["functional mismatch remains"],
        "observations": ["y mismatches at cycle 2"],
        "root_causes": ["polarity hypothesis is unsupported"],
        "invalidated_assumptions": ["inversion fixes output"],
        "open_questions": ["is state required?"],
        "recommended_actions": ["replan state semantics"],
        "knowledge_proposals": [
            {
                "proposal_id": "kp1",
                "item_type": "rejected_hypothesis",
                "scope": "candidate_lineage",
                "scope_ref": "c0002",
                "claim": "simple output inversion does not resolve the mismatch",
                "evidence_refs": ["e0002"],
                "confidence": "high",
                "conflicts_with": [],
            }
        ],
    }
    ledger = StubLedger([json.dumps(response)])
    analyst = AceAdaptiveAnalyst(
        workspace=WORKSPACE,
        model="mock",
        timeout=1,
        ledger=ledger,
    )

    review = analyst.review(
        _task(),
        plan,
        action,
        candidate,
        evidence,
        "e0002",
        "c0002",
        "analyst skill",
        snapshot,
    )

    assert review.invalidated_assumptions == ("inversion fixes output",)
    prompt = ledger.calls[0]["prompt"]
    assert "--- parent/dut.sv" in prompt
    assert "+++ candidate/dut.sv" in prompt
    assert "IMMUTABLE PUBLIC CONTEXT" not in prompt
    assert "public timing notes" not in prompt


def test_adaptive_debater_compares_summaries_without_receiving_full_rtl():
    plan = PlanHypothesis("p1", "direct logic", "combinational behavior")
    evidence_a = EDAEvidence(
        EDAStage.SIMULATION,
        "CANDIDATE_FUNCTIONAL",
        "functional_mismatch",
        "branch a mismatch",
        True,
    )
    evidence_b = EDAEvidence(
        EDAStage.PASS,
        "PASS",
        "none",
        "none",
        True,
    )
    artifact_a = CandidateArtifact.from_mapping(
        {"dut.sv": "module dut; localparam SECRET_A = 1; endmodule"}
    )
    artifact_b = CandidateArtifact.from_mapping(
        {"dut.sv": "module dut; localparam SECRET_B = 1; endmodule"}
    )
    candidates = (
        CandidateRecord(
            "c1", "p1", (), "a", artifact_a, BranchStatus.ACTIVE, 1, "e1"
        ),
        CandidateRecord(
            "c2", "p1", (), "b", artifact_b, BranchStatus.PASSED, 1, "e2"
        ),
    )
    snapshot = _snapshot(
        plan,
        candidates=candidates,
        evidence=(EvidenceRecord("e1", "c1", evidence_a), EvidenceRecord("e2", "c2", evidence_b)),
        round_index=2,
    )
    action = ActionCard(
        "discuss",
        AdaptiveActionKind.DISCUSS,
        source_node_ids=("c1", "c2"),
        hypothesis="branches make incompatible claims",
        obligation="compare their evidence",
        expected_observation="one claim is better supported",
        falsifier="evidence remains non-discriminating",
    )
    response = {
        "summary": "c2 has stronger public evidence",
        "branch_advantages": ["c2: pass"],
        "branch_risks": ["c1: mismatch"],
        "disagreements": ["behavior differs"],
        "recommended_discriminators": ["retain fail-closed evidence"],
        "knowledge_proposals": [],
    }
    ledger = StubLedger([json.dumps(response)])
    debater = AceAdaptiveDebater(
        workspace=WORKSPACE,
        model="mock",
        timeout=1,
        ledger=ledger,
    )

    outcome = debater.discuss(_task(), "debater skill", action, snapshot)

    assert outcome.summary == "c2 has stronger public evidence"
    prompt = ledger.calls[0]["prompt"]
    assert "SECRET_A" not in prompt
    assert "SECRET_B" not in prompt
    assert '"candidate_hash"' in prompt


def test_call_ledger_assigns_unique_records_under_parallel_role_calls(tmp_path):
    def role_call(**kwargs):
        time.sleep(0.01)
        kwargs["out_json"].write_text(
            json.dumps({"content": "ok", "usage": {"total_tokens": 3}}),
            encoding="utf-8",
        )
        return "ok"

    ledger = CallLedger(tmp_path, role_call=role_call)

    def invoke(_index):
        return ledger.invoke(
            workspace=WORKSPACE,
            role="parallel",
            component="generator",
            model="mock",
            system="system",
            prompt="prompt",
            timeout=1,
            temperature=0,
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        assert list(pool.map(invoke, range(12))) == ["ok"] * 12

    summary = ledger.summary()
    assert summary["calls"] == 12
    assert summary["successful_calls"] == 12
    assert summary["usage_totals"]["total_tokens"] == 36
    assert len({record["sequence"] for record in summary["records"]}) == 12
    persisted = json.loads((tmp_path / "call_ledger.json").read_text(encoding="utf-8"))
    assert persisted["calls"] == 12


def test_call_ledger_blocks_calls_after_exact_resource_limit(tmp_path):
    def role_call(**kwargs):
        kwargs["out_json"].write_text(
            json.dumps({"content": "ok", "usage": {"total_tokens": 5}}),
            encoding="utf-8",
        )
        return "ok"

    ledger = CallLedger(tmp_path, role_call=role_call, max_calls=1, max_total_tokens=10)
    invoke_args = dict(
        workspace=WORKSPACE,
        role="bounded",
        component="generator",
        model="mock",
        system="system",
        prompt="prompt",
        timeout=1,
        temperature=0,
    )

    assert ledger.invoke(**invoke_args) == "ok"
    with pytest.raises(ResourceBudgetExceeded, match="LLM call budget exhausted"):
        ledger.invoke(**invoke_args)
    assert ledger.summary()["calls"] == 1


def test_adaptive_director_uses_compact_fallback_after_two_invalid_dispatches():
    assessment = {
        "spec_certainty": "medium",
        "plan_concentration": "high",
        "implementation_risk": "medium",
        "verification_observability": "medium",
        "rationale": "one known plan remains",
    }
    invalid = json.dumps(
        {
            "assessment": assessment,
            "rationale": "accidentally omitted work",
            "dispatch": [],
            "knowledge_decisions": [],
        }
    )
    valid = json.dumps(
        {
            "assessment": assessment,
            "rationale": "exercise the known plan",
            "dispatch": [
                {
                    "action_id": "fallback-seed",
                    "kind": "seed_plan",
                    "source_node_ids": [],
                    "plan_id": "p1",
                    "hypothesis": "the direct plan is viable",
                    "obligation": "build one complete candidate",
                    "semantic_delta": "",
                    "expected_observation": "verifier evidence is produced",
                    "falsifier": "the candidate fails verification",
                    "evidence_refs": [],
                    "knowledge_refs": [],
                }
            ],
            "knowledge_decisions": [],
        }
    )
    ledger = StubLedger([invalid, invalid, valid])
    director = AceAdaptiveDirector(
        workspace=WORKSPACE, model="mock", timeout=1, ledger=ledger
    )

    decision = director.decide(
        _task(), "director skill", _snapshot(PlanHypothesis("p1", "direct", "clear"))
    )

    assert decision.dispatch[0].action_id == "fallback-seed"
    assert [call["role"] for call in ledger.calls] == [
        "adaptive_director",
        "adaptive_director_retry",
        "adaptive_director_fallback",
    ]
    assert ledger.calls[-1]["thinking"] == "disabled"
    assert len(ledger.invalid) == 2


def test_adaptive_analyst_retries_truncated_response_with_bounded_facts():
    response = json.dumps(
        {
            "strengths": ["candidate reaches simulation"],
            "weaknesses": ["output mismatches"],
            "observations": ["mismatch at cycle 2"],
            "root_causes": ["edge alignment may be wrong"],
            "invalidated_assumptions": [],
            "open_questions": ["which edge is required"],
            "recommended_actions": ["test one edge-alignment correction"],
            "knowledge_proposals": [
                {
                    "proposal_id": "retry-fact",
                    "item_type": "observed_fact",
                    "scope": "candidate_lineage",
                    "scope_ref": "c0002",
                    "claim": "the candidate mismatches at cycle 2",
                    "evidence_refs": ["e0002"],
                    "confidence": "high",
                    "conflicts_with": [],
                }
            ],
        }
    )
    ledger = StubLedger(
        [RuntimeError("adaptive_analyst response exhausted max_tokens"), response]
    )
    analyst = AceAdaptiveAnalyst(
        workspace=WORKSPACE, model="mock", timeout=1, ledger=ledger
    )
    parent = CandidateArtifact.from_mapping(
        {"dut.sv": "module dut; assign y = a; endmodule\n"}
    )
    candidate = CandidateArtifact.from_mapping(
        {"dut.sv": "module dut; assign y = ~a; endmodule\n"}
    )
    plan = PlanHypothesis("p1", "direct", "one edge interpretation")
    action = ActionCard(
        "refine",
        AdaptiveActionKind.REFINE,
        source_node_ids=("c0001",),
        plan_id="p1",
        hypothesis="edge alignment is wrong",
        obligation="change one edge",
        semantic_delta="invert the sampled data path",
        expected_observation="mismatch disappears",
        falsifier="mismatch remains",
    )
    evidence = EDAEvidence(
        EDAStage.SIMULATION,
        "CANDIDATE_FUNCTIONAL",
        "functional_mismatch",
        "mismatch at cycle 2",
        True,
        public_feedback="mismatch at cycle 2",
    )
    parent_record = CandidateRecord(
        "c0001", "p1", (), "seed", parent, BranchStatus.ACTIVE, 1, "e0001"
    )

    review = analyst.review(
        _task(),
        plan,
        action,
        candidate,
        evidence,
        "e0002",
        "c0002",
        "analyst skill",
        _snapshot(plan, candidates=(parent_record,), round_index=2),
    )

    assert review.candidate_node_id == "c0002"
    assert [call["role"] for call in ledger.calls] == [
        "adaptive_analyst",
        "adaptive_analyst_retry",
    ]
    assert ledger.calls[-1]["thinking"] == "disabled"
    assert ledger.calls[-1]["max_tokens"] == 2048
    retry_prompt = ledger.calls[-1]["prompt"]
    assert "mismatch at cycle 2" in retry_prompt
    assert "c0002" in retry_prompt
