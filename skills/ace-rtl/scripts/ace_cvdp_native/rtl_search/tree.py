"""Global tree storage, lineage queries, and explicit backtracking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

from .node import ActionKind, EDAEvidence, NodeKind, NodeStatus, SearchNode


@dataclass(frozen=True, slots=True)
class BacktrackEvent:
    source_id: str
    destination_id: str
    reason: str


class SearchTree:
    def __init__(self) -> None:
        self._nodes: Dict[str, SearchNode] = {}
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._status: Dict[str, NodeStatus] = {}
        self._tried_actions: Dict[str, Set[ActionKind]] = defaultdict(set)
        self._backtracks: List[BacktrackEvent] = []
        self._root_id: Optional[str] = None

    @property
    def root_id(self) -> str:
        if self._root_id is None:
            raise RuntimeError("search tree has no root")
        return self._root_id

    @property
    def nodes(self) -> Dict[str, SearchNode]:
        return dict(self._nodes)

    @property
    def backtracks(self) -> List[BacktrackEvent]:
        return list(self._backtracks)

    def node(self, node_id: str) -> SearchNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown search node: {node_id}") from exc

    def status(self, node_id: str) -> NodeStatus:
        self.node(node_id)
        return self._status[node_id]

    def set_status(self, node_id: str, status: NodeStatus) -> None:
        self.node(node_id)
        self._status[node_id] = status

    def add_root(self, task_id: str, *, node_id: str = "root") -> SearchNode:
        if self._root_id is not None:
            raise ValueError("search tree already has a root")
        root = SearchNode(
            node_id=node_id,
            parent_id=None,
            kind=NodeKind.ROOT,
            depth=0,
            hypothesis=task_id,
        )
        self._insert(root, NodeStatus.ACTIVE)
        self._root_id = root.node_id
        return root

    def add_plan(
        self,
        parent_id: str,
        hypothesis: str,
        *,
        plan_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> SearchNode:
        parent = self.node(parent_id)
        if parent.kind not in {NodeKind.ROOT, NodeKind.PLAN}:
            raise ValueError("plan nodes must descend from root or another plan")
        resolved_plan_id = plan_id or f"plan_{len(self._nodes)}"
        plan = SearchNode(
            node_id=node_id or resolved_plan_id,
            parent_id=parent_id,
            kind=NodeKind.PLAN,
            depth=parent.depth + 1,
            plan_id=resolved_plan_id,
            hypothesis=hypothesis,
            action_from_parent=ActionKind.EXPAND,
        )
        self._insert(plan, NodeStatus.ACTIVE)
        return plan

    def add_candidate(
        self,
        parent_id: str,
        candidate_hash: str,
        evidence: EDAEvidence,
        *,
        action_from_parent: ActionKind,
        node_id: str,
        source_iteration: Optional[int] = None,
        source_ref: Optional[str] = None,
    ) -> SearchNode:
        parent = self.node(parent_id)
        if action_from_parent not in {ActionKind.EXPAND, ActionKind.REFINE}:
            raise ValueError("candidate-producing action must be EXPAND or REFINE")
        owning_plan = parent.plan_id
        if parent.kind is NodeKind.ROOT:
            owning_plan = None
        candidate = SearchNode(
            node_id=node_id,
            parent_id=parent_id,
            kind=NodeKind.CANDIDATE,
            depth=parent.depth + 1,
            plan_id=owning_plan,
            candidate_hash=candidate_hash,
            evidence=evidence,
            action_from_parent=action_from_parent,
            source_iteration=source_iteration,
            source_ref=source_ref,
        )
        initial_status = NodeStatus.PASSED if evidence.passed else NodeStatus.ACTIVE
        self._insert(candidate, initial_status)
        return candidate

    def _insert(self, node: SearchNode, status: NodeStatus) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate node_id: {node.node_id}")
        if node.parent_id is not None and node.parent_id not in self._nodes:
            raise ValueError(f"unknown parent_id: {node.parent_id}")
        self._nodes[node.node_id] = node
        self._status[node.node_id] = status
        if node.parent_id is not None:
            self._children[node.parent_id].append(node.node_id)

    def children(self, node_id: str) -> List[SearchNode]:
        self.node(node_id)
        return [self.node(child_id) for child_id in self._children.get(node_id, [])]

    def lineage(self, node_id: str, *, include_self: bool = True) -> List[SearchNode]:
        current = self.node(node_id)
        reverse_lineage: List[SearchNode] = [current] if include_self else []
        parent_id = current.parent_id
        while parent_id is not None:
            parent = self.node(parent_id)
            reverse_lineage.append(parent)
            parent_id = parent.parent_id
        return list(reversed(reverse_lineage))

    def ancestors(self, node_id: str) -> List[SearchNode]:
        return self.lineage(node_id, include_self=False)

    def candidate_ancestors(self, node_id: str) -> List[SearchNode]:
        return [node for node in self.ancestors(node_id) if node.kind is NodeKind.CANDIDATE]

    def nearest_ancestor(
        self,
        node_id: str,
        predicate: Callable[[SearchNode], bool],
    ) -> Optional[SearchNode]:
        for node in reversed(self.ancestors(node_id)):
            if predicate(node):
                return node
        return None

    def owning_plan(self, node_id: str) -> Optional[SearchNode]:
        node = self.node(node_id)
        if node.kind is NodeKind.PLAN:
            return node
        return self.nearest_ancestor(node_id, lambda item: item.kind is NodeKind.PLAN)

    def mark_action_tried(self, node_id: str, action: ActionKind) -> None:
        self.node(node_id)
        self._tried_actions[node_id].add(action)

    def tried_actions(self, node_id: str) -> Set[ActionKind]:
        self.node(node_id)
        return set(self._tried_actions.get(node_id, set()))

    def backtrack(self, source_id: str, destination_id: str, reason: str) -> BacktrackEvent:
        source = self.node(source_id)
        destination = self.node(destination_id)
        if destination_id not in {node.node_id for node in self.ancestors(source_id)}:
            raise ValueError("backtrack destination must be an ancestor of the source")
        self._status[source.node_id] = NodeStatus.BACKTRACKED
        if self._status[destination.node_id] is not NodeStatus.PASSED:
            self._status[destination.node_id] = NodeStatus.ACTIVE
        event = BacktrackEvent(source.node_id, destination.node_id, reason)
        self._backtracks.append(event)
        return event

    def active_frontier(self) -> List[SearchNode]:
        frontier: List[SearchNode] = []
        for node_id, node in self._nodes.items():
            if node.kind is not NodeKind.CANDIDATE:
                continue
            if self._status[node_id] is not NodeStatus.ACTIVE:
                continue
            active_children = [
                child
                for child in self.children(node_id)
                if self._status[child.node_id] in {NodeStatus.ACTIVE, NodeStatus.PASSED}
            ]
            if not active_children:
                frontier.append(node)
        return frontier

    def __len__(self) -> int:
        return len(self._nodes)
