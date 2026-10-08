from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "skills" / "ace-rtl" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ace_cvdp_native.rtl_search import (  # noqa: E402
    ActionKind,
    DecisionContext,
    EDAEvidence,
    EDAStage,
    SearchTree,
    TreePolicy,
)
from ace_cvdp_native.rtl_search.scoring import compare_evidence  # noqa: E402


def evidence(
    stage: EDAStage,
    signature: str,
    *,
    passed_subtests: int = 0,
    total_subtests: int = 1,
) -> EDAEvidence:
    return EDAEvidence(
        stage=stage,
        original_stage=stage.name,
        failure_class=stage.name.lower(),
        failure_signature=signature,
        complete_bundle=stage is not EDAStage.INCOMPLETE,
        passed_subtests=passed_subtests,
        total_subtests=total_subtests,
    )


def make_tree() -> tuple[SearchTree, str]:
    tree = SearchTree()
    root = tree.add_root("task")
    plan = tree.add_plan(root.node_id, "historical plan", node_id="plan")
    return tree, plan.node_id


class SearchTreeStateTest(unittest.TestCase):
    def test_nodes_are_immutable_and_lineage_is_explicit(self):
        tree, plan_id = make_tree()
        node = tree.add_candidate(
            plan_id,
            "hash-a",
            evidence(EDAStage.SIMULATION, "functional-a"),
            action_from_parent=ActionKind.EXPAND,
            node_id="a",
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            node.candidate_hash = "mutated"  # type: ignore[misc]
        self.assertEqual([item.node_id for item in tree.lineage("a")], ["root", "plan", "a"])

    def test_hard_eda_stage_dominates_soft_subtest_count(self):
        compile_failure = evidence(
            EDAStage.COMPILE,
            "compile",
            passed_subtests=99,
            total_subtests=100,
        )
        simulation_failure = evidence(
            EDAStage.SIMULATION,
            "simulation",
            passed_subtests=0,
            total_subtests=1,
        )
        self.assertGreater(compare_evidence(simulation_failure, compile_failure), 0)


class TreePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = TreePolicy()

    def test_structural_regression_backtracks_to_simulating_parent(self):
        tree, plan_id = make_tree()
        parent = tree.add_candidate(
            plan_id,
            "hash-good-structure",
            evidence(EDAStage.SIMULATION, "functional"),
            action_from_parent=ActionKind.EXPAND,
            node_id="iter_05",
        )
        child = tree.add_candidate(
            parent.node_id,
            "hash-incomplete",
            evidence(EDAStage.INCOMPLETE, "missing error_calc"),
            action_from_parent=ActionKind.REFINE,
            node_id="iter_06",
        )
        decision = self.policy.decide(tree, child.node_id)
        self.assertEqual(decision.action, ActionKind.BACKTRACK)
        self.assertEqual(decision.reason, "structural_regression")
        self.assertEqual(decision.destination_id, parent.node_id)

    def test_immediate_duplicate_returns_to_nearest_nonduplicate_ancestor(self):
        tree, plan_id = make_tree()
        iter7 = tree.add_candidate(
            plan_id,
            "hash-7",
            evidence(EDAStage.SIMULATION, "functional"),
            action_from_parent=ActionKind.EXPAND,
            node_id="iter_07",
        )
        iter8 = tree.add_candidate(
            iter7.node_id,
            "hash-8",
            evidence(EDAStage.SIMULATION, "functional"),
            action_from_parent=ActionKind.REFINE,
            node_id="iter_08",
        )
        iter9 = tree.add_candidate(
            iter8.node_id,
            "hash-8",
            evidence(EDAStage.SIMULATION, "functional"),
            action_from_parent=ActionKind.REFINE,
            node_id="iter_09",
        )
        decision = self.policy.decide(tree, iter9.node_id)
        self.assertEqual(decision.action, ActionKind.BACKTRACK)
        self.assertEqual(decision.reason, "duplicate_candidate_state")
        self.assertEqual(decision.destination_id, iter7.node_id)

    def test_abab_oscillation_returns_to_plan(self):
        tree, parent_id = make_tree()
        for index, signature in enumerate(["A", "B", "A", "B"], start=1):
            node = tree.add_candidate(
                parent_id,
                f"hash-{index}",
                evidence(EDAStage.SIMULATION, signature),
                action_from_parent=ActionKind.EXPAND if index == 1 else ActionKind.REFINE,
                node_id=f"n{index}",
            )
            parent_id = node.node_id
        decision = self.policy.decide(tree, parent_id)
        self.assertEqual(decision.action, ActionKind.BACKTRACK)
        self.assertEqual(decision.reason, "failure_signature_oscillation")
        self.assertEqual(decision.destination_id, "plan")

    def test_pass_stops_even_if_candidate_hash_repeats(self):
        tree, plan_id = make_tree()
        parent = tree.add_candidate(
            plan_id,
            "same-hash",
            evidence(EDAStage.SIMULATION, "failed"),
            action_from_parent=ActionKind.EXPAND,
            node_id="failed",
        )
        passed = tree.add_candidate(
            parent.node_id,
            "same-hash",
            evidence(EDAStage.PASS, "pass", passed_subtests=1, total_subtests=1),
            action_from_parent=ActionKind.REFINE,
            node_id="passed",
        )
        decision = self.policy.decide(tree, passed.node_id)
        self.assertEqual(decision.action, ActionKind.STOP)
        self.assertEqual(decision.reason, "fail_closed_pass")

    def test_unlocalized_failure_with_alternative_hypothesis_expands(self):
        tree, plan_id = make_tree()
        node = tree.add_candidate(
            plan_id,
            "hash-a",
            evidence(EDAStage.SIMULATION, "semantic ambiguity"),
            action_from_parent=ActionKind.EXPAND,
            node_id="a",
        )
        decision = self.policy.decide(
            tree,
            node.node_id,
            DecisionContext(
                localized_failure=False,
                distinct_hypotheses_available=True,
            ),
        )
        self.assertEqual(decision.action, ActionKind.EXPAND)

    def test_backtrack_rejects_non_ancestor_destination(self):
        tree, plan_id = make_tree()
        left = tree.add_candidate(
            plan_id,
            "left",
            evidence(EDAStage.SIMULATION, "left"),
            action_from_parent=ActionKind.EXPAND,
            node_id="left",
        )
        right = tree.add_candidate(
            plan_id,
            "right",
            evidence(EDAStage.SIMULATION, "right"),
            action_from_parent=ActionKind.EXPAND,
            node_id="right",
        )
        with self.assertRaisesRegex(ValueError, "must be an ancestor"):
            tree.backtrack(left.node_id, right.node_id, "invalid")


if __name__ == "__main__":
    unittest.main()
