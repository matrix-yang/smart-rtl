"""Deterministic breadth/depth/backtracking policy for RTL-Director v0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .node import ActionKind, EDAStage, NodeKind, SearchNode
from .scoring import has_subtest_regression, is_abab_oscillation, is_structural_regression
from .tree import SearchTree


@dataclass(frozen=True, slots=True)
class DecisionContext:
    localized_failure: Optional[bool] = None
    plan_conflict: bool = False
    distinct_hypotheses_available: bool = False
    search_exhausted: bool = False


@dataclass(frozen=True, slots=True)
class Decision:
    action: ActionKind
    reason: str
    destination_id: Optional[str] = None


class TreePolicy:
    """EDA-first policy with no token/budget or LLM-valued inputs."""

    def decide(
        self,
        tree: SearchTree,
        node_id: str,
        context: Optional[DecisionContext] = None,
    ) -> Decision:
        context = context or DecisionContext()
        node = tree.node(node_id)
        if node.kind is not NodeKind.CANDIDATE or node.evidence is None:
            raise ValueError("policy decisions require an evaluated candidate node")

        evidence = node.evidence
        if evidence.stage is EDAStage.PASS:
            return Decision(ActionKind.STOP, "fail_closed_pass")
        if evidence.stage is EDAStage.INFRASTRUCTURE:
            return Decision(ActionKind.STOP, "infrastructure_failure")
        if context.search_exhausted:
            return Decision(ActionKind.STOP, "search_exhausted")

        duplicate_destination = self._duplicate_destination(tree, node)
        if duplicate_destination is not None:
            return Decision(
                ActionKind.BACKTRACK,
                "duplicate_candidate_state",
                duplicate_destination.node_id,
            )

        parent = tree.node(node.parent_id) if node.parent_id is not None else None
        if (
            parent is not None
            and parent.kind is NodeKind.CANDIDATE
            and parent.evidence is not None
            and is_structural_regression(parent.evidence, evidence)
        ):
            return Decision(
                ActionKind.BACKTRACK,
                "structural_regression",
                parent.node_id,
            )

        candidate_lineage = tree.candidate_ancestors(node_id) + [node]
        signatures = [
            item.evidence.failure_signature
            for item in candidate_lineage
            if item.evidence is not None
        ]
        if is_abab_oscillation(signatures):
            plan = tree.owning_plan(node_id)
            destination = plan.node_id if plan is not None else tree.root_id
            return Decision(ActionKind.BACKTRACK, "failure_signature_oscillation", destination)

        if self._two_step_subtest_regression(candidate_lineage):
            destination = candidate_lineage[-3]
            return Decision(
                ActionKind.BACKTRACK,
                "two_step_subtest_regression",
                destination.node_id,
            )

        if context.plan_conflict:
            plan = tree.owning_plan(node_id)
            destination = plan.node_id if plan is not None else tree.root_id
            return Decision(ActionKind.BACKTRACK, "plan_or_spec_conflict", destination)

        if context.distinct_hypotheses_available or context.localized_failure is False:
            return Decision(ActionKind.EXPAND, "untried_distinct_hypothesis", node.parent_id)

        if evidence.stage is EDAStage.INCOMPLETE:
            return Decision(ActionKind.REFINE, "repair_incomplete_bundle")
        if evidence.stage in {EDAStage.COMPILE, EDAStage.ELABORATE}:
            return Decision(ActionKind.REFINE, "repair_structural_or_compile_failure")
        if evidence.stage is EDAStage.SIMULATION:
            reason = (
                "localized_functional_repair"
                if context.localized_failure is True
                else "functional_failure_default_refine"
            )
            return Decision(ActionKind.REFINE, reason)
        return Decision(ActionKind.EXPAND, "no_local_repair_signal", node.parent_id)

    @staticmethod
    def _duplicate_destination(tree: SearchTree, node: SearchNode) -> Optional[SearchNode]:
        ancestors = tree.candidate_ancestors(node.node_id)
        if not ancestors or ancestors[-1].candidate_hash != node.candidate_hash:
            return None
        for ancestor in reversed(ancestors[:-1]):
            if ancestor.candidate_hash != node.candidate_hash:
                return ancestor
        plan = tree.owning_plan(node.node_id)
        return plan

    @staticmethod
    def _two_step_subtest_regression(lineage: list[SearchNode]) -> bool:
        if len(lineage) < 3:
            return False
        grandparent, parent, child = lineage[-3:]
        if not grandparent.evidence or not parent.evidence or not child.evidence:
            return False
        return has_subtest_regression(
            grandparent.evidence, parent.evidence
        ) and has_subtest_regression(parent.evidence, child.evidence)
