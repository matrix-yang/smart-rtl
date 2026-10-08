"""Complete role-aware breadth/depth/backtracking mechanism loop."""

from __future__ import annotations

import hashlib
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional, Sequence

from .contracts import CandidateArtifact, Diagnosis, PlanHypothesis, TaskContract, WorkItem
from .instructions import RoleInstructionProvider
from .node import ActionKind, EDAEvidence, EDAStage, NodeKind
from .policy import Decision, TreePolicy
from .roles import RoleSet
from .scoring import evidence_rank
from .tree import SearchTree


def _instruction_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class LoopStep:
    step: int
    node_id: str
    parent_id: str
    plan_id: str
    candidate_hash: str
    producing_action: str
    eda_stage: str
    decision: str
    decision_reason: str
    destination_id: Optional[str]
    contract_issues: tuple[str, ...]
    builder_instruction_hash: str
    analyst_instruction_hash: Optional[str]


@dataclass(frozen=True, slots=True)
class LoopResult:
    task_id: str
    status: str
    selected_node_id: Optional[str]
    selected_candidate_hash: Optional[str]
    instruction_provider: str
    instruction_version: str
    director_instruction_hash: str
    plans_created: int
    steps: tuple[LoopStep, ...]
    action_counts: Dict[str, int]
    role_calls: Dict[str, int]
    contract_rejected_candidates: int
    llm_api_calls: int
    tree_nodes: tuple[Dict[str, object], ...]
    backtracks: tuple[Dict[str, str], ...]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["steps"] = [asdict(step) for step in self.steps]
        return payload


class RTLSearchEngine:
    """Model-independent control loop with injected roles and instructions."""

    def __init__(
        self,
        *,
        roles: RoleSet,
        instructions: RoleInstructionProvider,
        policy: Optional[TreePolicy] = None,
        branch_width: int = 2,
        max_steps: int = 12,
        max_plans: int = 4,
    ) -> None:
        if branch_width < 1 or max_steps < 1 or max_plans < branch_width:
            raise ValueError("invalid branch_width/max_steps/max_plans configuration")
        self.roles = roles
        self.instructions = instructions
        self.policy = policy or TreePolicy()
        self.branch_width = branch_width
        self.max_steps = max_steps
        self.max_plans = max_plans

    def run(self, task: TaskContract) -> LoopResult:
        tree = SearchTree()
        root = tree.add_root(task.task_id)
        plans: Dict[str, PlanHypothesis] = {}
        artifacts: Dict[str, CandidateArtifact] = {}
        pending: Deque[WorkItem] = deque()
        steps: List[LoopStep] = []
        action_counts: Counter[str] = Counter()
        role_calls: Counter[str] = Counter()
        contract_rejected = 0
        director_instruction = self.instructions.director(task)

        self._add_plans(
            task=task,
            tree=tree,
            plans=plans,
            pending=pending,
            requested_count=self.branch_width,
            role_calls=role_calls,
        )
        if not pending:
            return self._result(
                task,
                tree,
                plans,
                steps,
                action_counts,
                role_calls,
                contract_rejected,
                director_instruction,
                status="no_plan",
                selected_node_id=None,
                artifacts=artifacts,
            )

        for step_index in range(1, self.max_steps + 1):
            if not pending:
                break
            work = pending.popleft()
            plan = plans[work.plan_id]
            parent_artifact = artifacts.get(work.parent_id)
            builder_instruction = self.instructions.builder(
                task,
                plan,
                work.action,
                parent_artifact,
                work.diagnosis,
            )
            role_calls["builder"] += 1
            artifact = self.roles.builder.build(
                task,
                plan,
                builder_instruction,
                work.action,
                parent_artifact,
                work.diagnosis,
            )
            issues = artifact.contract_issues(task)
            if issues:
                contract_rejected += 1
                evidence = EDAEvidence(
                    stage=EDAStage.INCOMPLETE,
                    original_stage="LOCAL_CONTRACT",
                    failure_class="candidate_contract",
                    failure_signature=issues[0],
                    complete_bundle=False,
                    total_subtests=0,
                )
            else:
                role_calls["verifier"] += 1
                evidence = self.roles.verifier.verify(task, artifact)

            node_id = f"candidate_{step_index:03d}"
            node = tree.add_candidate(
                work.parent_id,
                artifact.candidate_hash,
                evidence,
                action_from_parent=work.action,
                node_id=node_id,
                source_iteration=step_index,
                source_ref=f"mechanism_step:{step_index}",
            )
            artifacts[node.node_id] = artifact

            diagnosis: Optional[Diagnosis] = None
            analyst_hash: Optional[str] = None
            if evidence.stage not in {
                EDAStage.INCOMPLETE,
                EDAStage.INFRASTRUCTURE,
                EDAStage.PASS,
            }:
                analyst_instruction = self.instructions.analyst(task, plan, artifact, evidence)
                analyst_hash = _instruction_hash(analyst_instruction)
                role_calls["analyst"] += 1
                diagnosis = self.roles.analyst.analyze(
                    task,
                    plan,
                    artifact,
                    evidence,
                    analyst_instruction,
                )

            context = diagnosis.to_decision_context() if diagnosis else None
            role_calls["director"] += 1
            decision = self.policy.decide(tree, node.node_id, context)
            action_counts[decision.action.value] += 1
            if decision.action is ActionKind.BACKTRACK:
                if decision.destination_id is None:
                    raise RuntimeError("BACKTRACK decision omitted destination")
                tree.backtrack(node.node_id, decision.destination_id, decision.reason)
                pending.appendleft(
                    WorkItem(
                        parent_id=decision.destination_id,
                        plan_id=work.plan_id,
                        action=ActionKind.EXPAND,
                        diagnosis=diagnosis,
                    )
                )
            elif decision.action is ActionKind.REFINE:
                pending.appendleft(
                    WorkItem(
                        parent_id=node.node_id,
                        plan_id=work.plan_id,
                        action=ActionKind.REFINE,
                        diagnosis=diagnosis,
                    )
                )
            elif decision.action is ActionKind.EXPAND:
                if not self._has_pending_plan(pending, tree):
                    self._add_plans(
                        task=task,
                        tree=tree,
                        plans=plans,
                        pending=pending,
                        requested_count=1,
                        role_calls=role_calls,
                    )

            steps.append(
                LoopStep(
                    step=step_index,
                    node_id=node.node_id,
                    parent_id=work.parent_id,
                    plan_id=work.plan_id,
                    candidate_hash=artifact.candidate_hash,
                    producing_action=work.action.value,
                    eda_stage=evidence.original_stage,
                    decision=decision.action.value,
                    decision_reason=decision.reason,
                    destination_id=decision.destination_id,
                    contract_issues=issues,
                    builder_instruction_hash=_instruction_hash(builder_instruction),
                    analyst_instruction_hash=analyst_hash,
                )
            )

            if decision.action is ActionKind.STOP:
                status = "passed" if evidence.stage is EDAStage.PASS else decision.reason
                return self._result(
                    task,
                    tree,
                    plans,
                    steps,
                    action_counts,
                    role_calls,
                    contract_rejected,
                    director_instruction,
                    status=status,
                    selected_node_id=node.node_id,
                    artifacts=artifacts,
                )

        selected = self._best_candidate(tree)
        return self._result(
            task,
            tree,
            plans,
            steps,
            action_counts,
            role_calls,
            contract_rejected,
            director_instruction,
            status="max_steps" if pending else "search_exhausted",
            selected_node_id=selected,
            artifacts=artifacts,
        )

    def _add_plans(
        self,
        *,
        task: TaskContract,
        tree: SearchTree,
        plans: Dict[str, PlanHypothesis],
        pending: Deque[WorkItem],
        requested_count: int,
        role_calls: Counter[str],
    ) -> None:
        capacity = self.max_plans - len(plans)
        requested_count = min(requested_count, capacity)
        if requested_count <= 0:
            return
        existing = tuple(plans.values())
        instruction = self.instructions.planner(task, existing, requested_count)
        role_calls["planner"] += 1
        proposed = tuple(
            self.roles.planner.propose(
                task,
                instruction,
                requested_count,
                existing,
            )
        )
        if len(proposed) > requested_count:
            raise ValueError("Planner returned more plans than requested")
        for plan in proposed:
            if plan.plan_id in plans:
                raise ValueError(f"Planner returned duplicate plan_id: {plan.plan_id}")
            plan_node = tree.add_plan(
                tree.root_id,
                plan.summary,
                plan_id=plan.plan_id,
                node_id=f"plan_{plan.plan_id}",
            )
            plans[plan.plan_id] = plan
            pending.append(
                WorkItem(
                    parent_id=plan_node.node_id,
                    plan_id=plan.plan_id,
                    action=ActionKind.EXPAND,
                )
            )

    @staticmethod
    def _has_pending_plan(pending: Deque[WorkItem], tree: SearchTree) -> bool:
        return any(tree.node(item.parent_id).kind is NodeKind.PLAN for item in pending)

    @staticmethod
    def _best_candidate(tree: SearchTree) -> Optional[str]:
        candidates = [
            node
            for node in tree.nodes.values()
            if node.kind is NodeKind.CANDIDATE and node.evidence is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda node: evidence_rank(node.evidence)).node_id

    def _result(
        self,
        task: TaskContract,
        tree: SearchTree,
        plans: Dict[str, PlanHypothesis],
        steps: Sequence[LoopStep],
        action_counts: Counter[str],
        role_calls: Counter[str],
        contract_rejected: int,
        director_instruction: str,
        *,
        status: str,
        selected_node_id: Optional[str],
        artifacts: Dict[str, CandidateArtifact],
    ) -> LoopResult:
        selected_hash = (
            artifacts[selected_node_id].candidate_hash
            if selected_node_id is not None and selected_node_id in artifacts
            else None
        )
        tree_nodes = tuple(
            {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "kind": node.kind.value,
                "plan_id": node.plan_id,
                "candidate_hash": node.candidate_hash,
                "stage": node.evidence.original_stage if node.evidence else None,
                "status": tree.status(node.node_id).value,
            }
            for node in tree.nodes.values()
        )
        backtracks = tuple(asdict(event) for event in tree.backtracks)
        return LoopResult(
            task_id=task.task_id,
            status=status,
            selected_node_id=selected_node_id,
            selected_candidate_hash=selected_hash,
            instruction_provider=self.instructions.name,
            instruction_version=self.instructions.version,
            director_instruction_hash=_instruction_hash(director_instruction),
            plans_created=len(plans),
            steps=tuple(steps),
            action_counts=dict(action_counts),
            role_calls=dict(role_calls),
            contract_rejected_candidates=contract_rejected,
            llm_api_calls=self.roles.api_calls,
            tree_nodes=tree_nodes,
            backtracks=backtracks,
        )
