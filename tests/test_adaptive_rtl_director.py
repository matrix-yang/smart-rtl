import threading
import time
import sys
from pathlib import Path

import pytest


WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = WORKSPACE / "skills" / "ace-rtl" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ace_cvdp_native.rtl_search.adaptive_contracts import (
    ActionCard,
    AdaptiveActionKind,
    BranchReview,
    BranchStatus,
    CandidateJudgement,
    CandidateRecord,
    DebateOutcome,
    DirectorDecision,
    EvidenceRecord,
    KnowledgeDecision,
    KnowledgeProposal,
    TaskAssessment,
)
from ace_cvdp_native.rtl_search.adaptive_engine import AdaptiveRTLDirectorEngine
from ace_cvdp_native.rtl_search.adaptive_roles import AdaptiveRoleSet
from ace_cvdp_native.rtl_search.contracts import (
    CandidateArtifact,
    PlanHypothesis,
    TaskContract,
)
from ace_cvdp_native.rtl_search.node import EDAEvidence, EDAStage
from ace_cvdp_native.rtl_search.skill_instructions import BWRTLSkillInstructions
from ace_cvdp_native.rtl_search.knowledge import KnowledgeLedger


def _assessment(label="uncertain"):
    return TaskAssessment(label, label, label, label, "Evidence-dependent assessment")


def _action(action_id, kind, *, plan=None, sources=()):
    return ActionCard(
        action_id=action_id,
        kind=kind,
        plan_id=plan,
        source_node_ids=tuple(sources),
        hypothesis=f"hypothesis for {action_id}",
        obligation=f"obligation for {action_id}",
        semantic_delta=f"semantic delta for {action_id}",
        expected_observation=f"observation for {action_id}",
        falsifier=f"falsifier for {action_id}",
    )


class Planner:
    def propose(self, task, instruction, existing_plans, knowledge):
        assert not existing_plans
        return (
            PlanHypothesis("p1", "counter-state implementation", "stateful interpretation"),
            PlanHypothesis("p2", "direct combinational implementation", "stateless interpretation"),
        )

    def revise(self, task, instruction, action, snapshot):
        return PlanHypothesis("p3", "revised implementation", action.hypothesis)


class ConcurrentBuilder:
    def __init__(self, fail_plan=None):
        self.fail_plan = fail_plan
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def build(self, task, plan, instruction, action, parents, snapshot):
        if self.fail_plan == plan.plan_id:
            raise RuntimeError(f"intentional {plan.plan_id} failure")
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.04)
        with self.lock:
            self.active -= 1
        source = f"module dut; // {plan.plan_id} {action.action_id}\nendmodule\n"
        return CandidateArtifact.from_mapping({"dut.sv": source})


class Verifier:
    def verify(self, task, candidate):
        text = candidate.as_dict()["dut.sv"]
        passed = "p2" in text
        return EDAEvidence(
            stage=EDAStage.PASS if passed else EDAStage.SIMULATION,
            original_stage="PASS" if passed else "CANDIDATE_FUNCTIONAL",
            failure_class="none" if passed else "functional_mismatch",
            failure_signature="none" if passed else "output mismatch",
            complete_bundle=True,
            passed_subtests=1 if passed else 0,
            total_subtests=1,
            public_feedback="public pass" if passed else "public mismatch at cycle 4",
        )


class Analyst:
    def review(
        self,
        task,
        plan,
        action,
        candidate,
        evidence,
        evidence_id,
        candidate_node_id,
        instruction,
        snapshot,
    ):
        conflicts = ("k0001",) if candidate_node_id == "c0002" else ()
        claim = (
            "The implementation needs a stateless interpretation"
            if conflicts
            else "The implementation needs a stateful interpretation"
        )
        proposal = KnowledgeProposal(
            proposal_id=f"proposal-{candidate_node_id}",
            item_type="root_cause",
            scope="candidate_lineage",
            scope_ref=candidate_node_id,
            claim=claim,
            evidence_refs=(evidence_id,),
            confidence="medium",
            conflicts_with=conflicts,
        )
        return BranchReview(
            candidate_node_id,
            evidence_id,
            strengths=("compiles",),
            weaknesses=("functional mismatch",),
            observations=(evidence.failure_signature,),
            root_causes=(claim,),
            invalidated_assumptions=(),
            open_questions=("stateful or stateless?",),
            recommended_actions=("compare an alternative plan",),
            knowledge_proposals=(proposal,),
        )


class VariableDirector:
    def __init__(self):
        self.snapshots = []

    def decide(self, task, instruction, snapshot):
        self.snapshots.append(snapshot)
        if snapshot.round_index == 1:
            return DirectorDecision(
                _assessment("high"),
                "One dominant initial interpretation is sufficient to test first.",
                (_action("r1-seed", AdaptiveActionKind.SEED_PLAN, plan="p1"),),
            )
        if snapshot.round_index == 2:
            assert [candidate.node_id for candidate in snapshot.candidates] == ["c0001"]
            assert snapshot.knowledge[0]["status"] == "proposed"
            return DirectorDecision(
                _assessment("medium"),
                "The first result leaves incompatible explanations, so run both.",
                (
                    _action(
                        "r2-refine",
                        AdaptiveActionKind.REFINE,
                        plan="p1",
                        sources=("c0001",),
                    ),
                    _action("r2-seed-alt", AdaptiveActionKind.SEED_PLAN, plan="p2"),
                ),
                knowledge_decisions=(
                    KnowledgeDecision("k0001", "supported", "Grounded in e0001"),
                ),
            )
        if snapshot.round_index == 3:
            assert {event.action_id for event in snapshot.prior_round_events} == {
                "r2-refine",
                "r2-seed-alt",
            }
            assert {candidate.node_id for candidate in snapshot.candidates} == {
                "c0001",
                "c0002",
                "c0003",
            }
            assert {item["status"] for item in snapshot.knowledge} == {"conflicting"}
            return DirectorDecision(
                _assessment("contested"),
                "Compare the two incompatible branches before delivery.",
                (
                    _action(
                        "r3-discuss",
                        AdaptiveActionKind.DISCUSS,
                        sources=("c0002", "c0003"),
                    ),
                ),
            )
        assert snapshot.round_index == 4
        assert [event.action_id for event in snapshot.prior_round_events] == ["r3-discuss"]
        return DirectorDecision(
            _assessment("resolved"),
            "The alternative branch has fail-closed passing evidence.",
            (
                ActionCard(
                    "r3-stop",
                    AdaptiveActionKind.STOP,
                    source_node_ids=("c0003",),
                ),
            ),
        )


class Debater:
    def discuss(self, task, instruction, action, snapshot):
        return DebateOutcome(
            summary="c0003 is supported by pass evidence; c0002 retains a conflicting cause.",
            branch_advantages=("c0003: public evaluator pass",),
            branch_risks=("c0002: functional mismatch",),
            disagreements=("stateful versus stateless behavior",),
            recommended_discriminators=("retain both evidence records",),
        )


def _task():
    return TaskContract.from_parts(
        task_id="synthetic",
        specification="Implement module dut using only the public contract.",
        target_files=("dut.sv",),
    )


def _skill():
    return BWRTLSkillInstructions.from_workspace(WORKSPACE)


def test_variable_width_rounds_share_all_siblings_and_knowledge_conflicts(tmp_path):
    builder = ConcurrentBuilder()
    director = VariableDirector()
    engine = AdaptiveRTLDirectorEngine(
        roles=AdaptiveRoleSet(
            Planner(), builder, Verifier(), Analyst(), director, debater=Debater()
        ),
        instructions=_skill(),
        physical_workers=2,
        max_rounds=5,
        checkpoint_dir=tmp_path,
    )

    result = engine.run(_task())

    assert result.status == "passed"
    assert result.selected_candidate_node_id == "c0003"
    assert [len(decision.dispatch) for decision in result.decisions] == [1, 2]
    assert builder.max_active == 2
    assert {item["status"] for item in result.knowledge} == {"conflicting"}
    assert result.round_events[-1][-1].action_id == "automatic-pass-stop-r2"
    assert (tmp_path / "adaptive_state.json").is_file()


class FailureIsolatingDirector:
    def decide(self, task, instruction, snapshot):
        if snapshot.round_index == 1:
            return DirectorDecision(
                _assessment(),
                "Exercise independent alternatives.",
                (
                    _action("bad", AdaptiveActionKind.SEED_PLAN, plan="p1"),
                    _action("good", AdaptiveActionKind.SEED_PLAN, plan="p2"),
                ),
            )
        return DirectorDecision(
            _assessment("resolved"),
            "Deliver the independently passing branch.",
            (ActionCard("stop", AdaptiveActionKind.STOP, source_node_ids=("c0002",)),),
        )


def test_branch_local_role_failure_does_not_abort_sibling():
    result = AdaptiveRTLDirectorEngine(
        roles=AdaptiveRoleSet(
            Planner(),
            ConcurrentBuilder(fail_plan="p1"),
            Verifier(),
            Analyst(),
            FailureIsolatingDirector(),
        ),
        instructions=_skill(),
        physical_workers=2,
        max_rounds=3,
    ).run(_task())

    assert result.status == "passed"
    assert result.selected_candidate_node_id == "c0002"
    assert any("intentional p1 failure" in error["error"] for error in result.role_errors)
    assert [candidate.node_id for candidate in result.candidates] == ["c0002"]


def test_action_contract_rejects_fixed_width_style_unlabelled_work():
    with pytest.raises(ValueError, match="omitted decision fields"):
        ActionCard("unlabelled", AdaptiveActionKind.SEED_PLAN, plan_id="p1")


def test_source_transform_requires_explicit_semantic_delta():
    with pytest.raises(ValueError, match="semantic_delta"):
        ActionCard(
            "cosmetic-refine",
            AdaptiveActionKind.REFINE,
            source_node_ids=("c1",),
            plan_id="p1",
            hypothesis="flattening cases may help",
            obligation="rewrite nested cases as one flat case",
            expected_observation="the code is flatter",
            falsifier="the code remains nested",
        )


def test_knowledge_ledger_deduplicates_exact_claims_with_valid_provenance():
    ledger = KnowledgeLedger()
    first = KnowledgeProposal(
        "p1",
        "observed_fact",
        "global",
        None,
        "Output mismatches at cycle four",
        ("e1",),
        "high",
    )
    duplicate = KnowledgeProposal(
        "p2",
        "observed_fact",
        "global",
        None,
        "  output   mismatches at cycle FOUR ",
        ("e2",),
        "high",
    )

    first_ids = ledger.add_proposals(
        (first,), source_role="analyst", source_id="c1", valid_evidence_refs=("e1",)
    )
    duplicate_ids = ledger.add_proposals(
        (duplicate,), source_role="analyst", source_id="c2", valid_evidence_refs=("e2",)
    )

    assert first_ids == duplicate_ids == ("k0001",)
    assert len(ledger.items()) == 1


class CancellationPlanner(Planner):
    def __init__(self):
        self.revisions = 0

    def revise(self, task, instruction, action, snapshot):
        self.revisions += 1
        return super().revise(task, instruction, action, snapshot)


class PassAwareBuilder(ConcurrentBuilder):
    def build(self, task, plan, instruction, action, parents, snapshot):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.08 if plan.plan_id == "p1" else 0.01)
        with self.lock:
            self.active -= 1
        return CandidateArtifact.from_mapping(
            {"dut.sv": f"module dut; // {plan.plan_id} {action.action_id}\nendmodule\n"}
        )


class CountingAnalyst(Analyst):
    def __init__(self):
        self.calls = 0

    def review(self, *args, **kwargs):
        self.calls += 1
        return super().review(*args, **kwargs)


class CancellationDirector:
    def decide(self, task, instruction, snapshot):
        if snapshot.round_index == 1:
            return DirectorDecision(
                _assessment("split"),
                "Run two competing implementations and queue one spec review.",
                (
                    _action("slow-fail", AdaptiveActionKind.SEED_PLAN, plan="p1"),
                    _action("fast-pass", AdaptiveActionKind.SEED_PLAN, plan="p2"),
                    _action("queued-review", AdaptiveActionKind.REVIEW_SPEC),
                ),
            )
        statuses = {event.action_id: event.status for event in snapshot.prior_round_events}
        assert statuses["queued-review"] == "cancelled_after_sibling_pass"
        slow_event = next(
            event for event in snapshot.prior_round_events if event.action_id == "slow-fail"
        )
        assert "Analyst skipped" in slow_event.detail
        return DirectorDecision(
            _assessment("resolved"),
            "A sibling has fail-closed passing evidence.",
            (ActionCard("stop-pass", AdaptiveActionKind.STOP, source_node_ids=("c0002",)),),
        )


def test_sibling_pass_cancels_queued_work_and_skips_losing_analyst():
    planner = CancellationPlanner()
    analyst = CountingAnalyst()
    builder = PassAwareBuilder()
    result = AdaptiveRTLDirectorEngine(
        roles=AdaptiveRoleSet(
            planner,
            builder,
            Verifier(),
            analyst,
            CancellationDirector(),
        ),
        instructions=_skill(),
        physical_workers=2,
        max_rounds=3,
    ).run(_task())

    assert result.status == "passed"
    assert result.selected_candidate_node_id == "c0002"
    assert planner.revisions == 0
    assert analyst.calls == 0
    assert builder.max_active == 2
    assert result.role_errors == ()


class DuplicateBuilder:
    def build(self, task, plan, instruction, action, parents, snapshot):
        del task, plan, instruction, action, parents, snapshot
        return CandidateArtifact.from_mapping(
            {"dut.sv": "module dut; assign out = in; endmodule\n"}
        )


class CountingFailVerifier:
    def __init__(self):
        self.calls = 0

    def verify(self, task, candidate):
        del task, candidate
        self.calls += 1
        return EDAEvidence(
            stage=EDAStage.COMPILE,
            original_stage="CANDIDATE_COMPILE",
            failure_class="candidate_compile",
            failure_signature="syntax error",
            complete_bundle=True,
            total_subtests=1,
            public_feedback="public syntax error",
        )


class DuplicateRejectingDirector:
    def decide(self, task, instruction, snapshot):
        del task, instruction
        if snapshot.round_index == 1:
            return DirectorDecision(
                _assessment("initial"),
                "Create the initial candidate.",
                (_action("seed", AdaptiveActionKind.SEED_PLAN, plan="p1"),),
            )
        if snapshot.round_index == 2:
            return DirectorDecision(
                _assessment("localized"),
                "Attempt one bounded refinement.",
                (
                    _action(
                        "duplicate-refine",
                        AdaptiveActionKind.REFINE,
                        plan="p1",
                        sources=("c0001",),
                    ),
                ),
            )
        duplicate = next(
            candidate for candidate in snapshot.candidates if candidate.node_id == "c0002"
        )
        assert duplicate.status.value == "pruned"
        duplicate_evidence = next(
            item for item in snapshot.evidence if item.candidate_node_id == "c0002"
        )
        assert duplicate_evidence.evidence.failure_class == "candidate_no_change"
        return DirectorDecision(
            _assessment("exhausted"),
            "The normalized refinement made no progress.",
            (ActionCard("stop-no-progress", AdaptiveActionKind.STOP),),
        )


def test_duplicate_refine_is_pruned_before_verifier_and_analyst():
    verifier = CountingFailVerifier()
    analyst = CountingAnalyst()
    result = AdaptiveRTLDirectorEngine(
        roles=AdaptiveRoleSet(
            Planner(),
            DuplicateBuilder(),
            verifier,
            analyst,
            DuplicateRejectingDirector(),
        ),
        instructions=_skill(),
        physical_workers=1,
        max_rounds=3,
    ).run(_task())

    assert result.status == "stopped_without_correctness"
    assert verifier.calls == 1
    assert analyst.calls == 1
    assert [candidate.status.value for candidate in result.candidates] == ["active", "pruned"]
    assert result.evidence[1].evidence.failure_class == "candidate_no_change"


def test_lineage_diagnostics_expose_recurrence_without_selecting_an_action():
    artifact = CandidateArtifact.from_mapping({"dut.sv": "module dut; endmodule\n"})
    candidates = {
        "c1": CandidateRecord("c1", "p1", (), "seed", artifact, BranchStatus.ACTIVE, 1, "e1"),
        "c2": CandidateRecord(
            "c2", "p1", ("c1",), "refine", artifact, BranchStatus.ACTIVE, 2, "e2"
        ),
        "c3": CandidateRecord(
            "c3", "p1", ("c2",), "refine2", artifact, BranchStatus.ACTIVE, 3, "e3"
        ),
    }
    evidence = {
        "e1": EvidenceRecord(
            "e1", "c1", EDAEvidence(EDAStage.SIMULATION, "SIM", "functional", "zero output", True)
        ),
        "e2": EvidenceRecord(
            "e2", "c2", EDAEvidence(EDAStage.SIMULATION, "SIM", "functional", "zero   output", True)
        ),
        "e3": EvidenceRecord(
            "e3", "c3", EDAEvidence(EDAStage.COMPILE, "COMPILE", "syntax", "missing macro", True)
        ),
    }

    diagnostics = AdaptiveRTLDirectorEngine._lineage_diagnostics(candidates, evidence)
    by_node = {item["candidate_node_id"]: item for item in diagnostics}

    assert by_node["c2"]["current_failure_signature_occurrences"] == 2
    assert by_node["c2"]["consecutive_current_failure_signature"] == 2
    assert by_node["c3"]["consecutive_current_failure_signature"] == 1
    assert [item["eda_stage"] for item in by_node["c3"]["evidence_trajectory"]] == [
        "SIMULATION",
        "SIMULATION",
        "COMPILE",
    ]


class OneSeedDirector:
    def decide(self, task, instruction, snapshot):
        del task, instruction, snapshot
        return DirectorDecision(
            _assessment("focused"),
            "One candidate is sufficient.",
            (_action("seed", AdaptiveActionKind.SEED_PLAN, plan="p1"),),
        )


class AlwaysPassCountingVerifier:
    def __init__(self):
        self.calls = 0

    def verify(self, task, candidate):
        del task, candidate
        self.calls += 1
        return EDAEvidence(
            EDAStage.PASS,
            "PASS",
            "none",
            "fail-closed pass",
            True,
            passed_subtests=1,
            total_subtests=1,
        )


def test_pass_at_exact_verifier_budget_stops_without_extra_director_round():
    verifier = AlwaysPassCountingVerifier()
    roles = AdaptiveRoleSet(
        Planner(),
        ConcurrentBuilder(),
        verifier,
        Analyst(),
        OneSeedDirector(),
        resource_usage_callback=lambda: {
            "llm_calls": 0,
            "total_tokens": 0,
            "verifier_calls": verifier.calls,
        },
    )
    result = AdaptiveRTLDirectorEngine(
        roles=roles,
        instructions=_skill(),
        physical_workers=1,
        max_rounds=3,
        resource_budget={"verifier_calls": 1},
    ).run(_task())

    assert result.status == "passed"
    assert result.rounds == 1
    assert len(result.decisions) == 1
    assert result.resource_usage["verifier_calls"] == 1
    assert result.round_events[-1][-1].action_id == "automatic-pass-stop-r1"


def test_candidate_is_not_started_without_minimum_token_reservation():
    verifier = AlwaysPassCountingVerifier()
    roles = AdaptiveRoleSet(
        Planner(),
        ConcurrentBuilder(),
        verifier,
        Analyst(),
        OneSeedDirector(),
        resource_usage_callback=lambda: {
            "llm_calls": 0,
            "total_tokens": 0,
            "verifier_calls": verifier.calls,
        },
        resource_cost_callback=lambda: {
            "minimum_completion_reservation_total_tokens": {
                "candidate_action": 100,
            }
        },
    )
    result = AdaptiveRTLDirectorEngine(
        roles=roles,
        instructions=_skill(),
        physical_workers=1,
        max_rounds=3,
        resource_budget={"total_tokens": 50, "verifier_calls": 1},
    ).run(_task())

    assert result.status == "resource_limit"
    assert result.candidates == ()
    assert verifier.calls == 0
    assert result.round_events[-1][0].action_id == "automatic-reservation-stop-r1"


class DeltaBuilder:
    def __init__(self):
        self.calls = 0

    def build(self, task, plan, instruction, action, parents, snapshot):
        del task, plan, instruction, parents, snapshot
        self.calls += 1
        return CandidateArtifact.from_mapping(
            {
                "dut.sv": (
                    "module dut;\n"
                    f"  // behavior selected by {action.action_id}\n"
                    "endmodule\n"
                )
            }
        )


class CandidateJudge:
    def __init__(self, accepted):
        self.accepted = accepted
        self.calls = 0

    def review_candidate(
        self, task, plan, action, parents, candidate, instruction, snapshot
    ):
        del task, plan, candidate, instruction, snapshot
        self.calls += 1
        assert action.kind is AdaptiveActionKind.REFINE
        assert len(parents) == 1
        return CandidateJudgement(
            accepted=self.accepted,
            rationale=(
                "The bounded delta implements the requested discriminator."
                if self.accepted
                else "The child rewrites behavior outside the claimed discriminator."
            ),
            action_fidelity="bounded" if self.accepted else "action scope violated",
            semantic_novelty="named behavior changed",
            state_preservation="parent interface preserved",
            rtl_risk="no obvious structural risk",
            revision_requirements=(
                ()
                if self.accepted
                else ("Limit the child to the requested behavioral partition.",)
            ),
        )


class SeedThenRefineDirector:
    def decide(self, task, instruction, snapshot):
        del task, instruction
        if snapshot.round_index == 1:
            return DirectorDecision(
                _assessment("initial"),
                "Establish one executable parent.",
                (_action("seed", AdaptiveActionKind.SEED_PLAN, plan="p1"),),
            )
        if snapshot.round_index == 2:
            return DirectorDecision(
                _assessment("localized"),
                "Test one evidence-grounded local semantic delta.",
                (
                    _action(
                        "bounded-refine",
                        AdaptiveActionKind.REFINE,
                        plan="p1",
                        sources=("c0001",),
                    ),
                ),
            )
        return DirectorDecision(
            _assessment("complete"),
            "No further deterministic work is required by this fixture.",
            (ActionCard("stop", AdaptiveActionKind.STOP),),
        )


def test_candidate_judge_rejection_prunes_child_without_spending_eda():
    builder = DeltaBuilder()
    verifier = CountingFailVerifier()
    analyst = CountingAnalyst()
    judge = CandidateJudge(accepted=False)
    result = AdaptiveRTLDirectorEngine(
        roles=AdaptiveRoleSet(
            Planner(),
            builder,
            verifier,
            analyst,
            SeedThenRefineDirector(),
            judge=judge,
        ),
        instructions=_skill(),
        physical_workers=1,
        max_rounds=3,
    ).run(_task())

    assert result.status == "stopped_without_correctness"
    assert builder.calls == 2
    assert judge.calls == 1
    assert verifier.calls == 1
    assert analyst.calls == 1
    assert [candidate.status.value for candidate in result.candidates] == [
        "active",
        "pruned",
    ]
    assert result.evidence[1].evidence.failure_class == "candidate_action_contract"
    assert result.candidates[1].candidate_judgement is not None
    assert result.candidates[1].candidate_judgement.accepted is False


def test_candidate_judge_acceptance_preserves_normal_verifier_path():
    builder = DeltaBuilder()
    verifier = CountingFailVerifier()
    analyst = CountingAnalyst()
    judge = CandidateJudge(accepted=True)
    result = AdaptiveRTLDirectorEngine(
        roles=AdaptiveRoleSet(
            Planner(),
            builder,
            verifier,
            analyst,
            SeedThenRefineDirector(),
            judge=judge,
        ),
        instructions=_skill(),
        physical_workers=1,
        max_rounds=3,
    ).run(_task())

    assert result.status == "stopped_without_correctness"
    assert builder.calls == 2
    assert judge.calls == 1
    assert verifier.calls == 2
    assert analyst.calls == 2
    assert [candidate.status.value for candidate in result.candidates] == [
        "active",
        "active",
    ]
    assert result.evidence[1].evidence.failure_class == "candidate_compile"
    assert result.candidates[1].candidate_judgement is not None
    assert result.candidates[1].candidate_judgement.accepted is True


def test_refine_reservation_includes_candidate_judge_cost():
    builder = DeltaBuilder()
    verifier = CountingFailVerifier()
    judge = CandidateJudge(accepted=True)
    result = AdaptiveRTLDirectorEngine(
        roles=AdaptiveRoleSet(
            Planner(),
            builder,
            verifier,
            Analyst(),
            SeedThenRefineDirector(),
            judge=judge,
            resource_usage_callback=lambda: {
                "llm_calls": 0,
                "total_tokens": 0,
                "verifier_calls": verifier.calls,
            },
            resource_cost_callback=lambda: {
                "minimum_completion_reservation_total_tokens": {
                    "candidate_action": 20,
                    "candidate_judge_action": 40,
                }
            },
        ),
        instructions=_skill(),
        physical_workers=1,
        max_rounds=3,
        resource_budget={"total_tokens": 50, "verifier_calls": 3},
    ).run(_task())

    assert result.status == "resource_limit"
    assert builder.calls == 1
    assert judge.calls == 0
    assert verifier.calls == 1
    assert len(result.candidates) == 1
    assert result.round_events[-1][0].action_id == "automatic-reservation-stop-r2"
