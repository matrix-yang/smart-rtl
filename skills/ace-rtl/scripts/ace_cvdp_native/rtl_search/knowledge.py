"""Evidence-linked shared memory for the adaptive BW-RTL controller."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .adaptive_contracts import KnowledgeDecision, KnowledgeProposal


class KnowledgeType(str, Enum):
    SPEC_CONSTRAINT = "spec_constraint"
    OBSERVED_FACT = "observed_fact"
    ROOT_CAUSE = "root_cause"
    FIX_PATTERN = "fix_pattern"
    REJECTED_HYPOTHESIS = "rejected_hypothesis"
    CONFLICT = "conflict"
    REUSABLE_ARTIFACT = "reusable_artifact"
    OPEN_QUESTION = "open_question"


class KnowledgeScope(str, Enum):
    GLOBAL = "global"
    PLAN_FAMILY = "plan_family"
    MODULE = "module"
    CANDIDATE_LINEAGE = "candidate_lineage"


class KnowledgeStatus(str, Enum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    CONFLICTING = "conflicting"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class KnowledgeItem:
    item_id: str
    item_type: KnowledgeType
    scope: KnowledgeScope
    scope_ref: Optional[str]
    claim: str
    evidence_refs: Tuple[str, ...]
    confidence: str
    status: KnowledgeStatus
    source_role: str
    source_id: str
    conflicts_with: Tuple[str, ...] = ()
    decision_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type.value,
            "scope": self.scope.value,
            "scope_ref": self.scope_ref,
            "claim": self.claim,
            "evidence_refs": list(self.evidence_refs),
            "confidence": self.confidence,
            "status": self.status.value,
            "source_role": self.source_role,
            "source_id": self.source_id,
            "conflicts_with": list(self.conflicts_with),
            "decision_rationale": self.decision_rationale,
        }


class KnowledgeLedger:
    """Validate provenance and transitions without judging semantic truth."""

    _ALLOWED_TRANSITIONS = {
        KnowledgeStatus.PROPOSED: {
            KnowledgeStatus.SUPPORTED,
            KnowledgeStatus.REFUTED,
            KnowledgeStatus.CONFLICTING,
            KnowledgeStatus.SUPERSEDED,
        },
        KnowledgeStatus.SUPPORTED: {
            KnowledgeStatus.REFUTED,
            KnowledgeStatus.CONFLICTING,
            KnowledgeStatus.SUPERSEDED,
        },
        KnowledgeStatus.CONFLICTING: {
            KnowledgeStatus.SUPPORTED,
            KnowledgeStatus.REFUTED,
            KnowledgeStatus.SUPERSEDED,
        },
        KnowledgeStatus.REFUTED: {KnowledgeStatus.SUPERSEDED},
        KnowledgeStatus.SUPERSEDED: set(),
    }

    def __init__(self) -> None:
        self._items: Dict[str, KnowledgeItem] = {}
        self._normalized_claims: Dict[Tuple[str, str, Optional[str], str], str] = {}
        self._counter = 0

    @staticmethod
    def _normalize_claim(claim: str) -> str:
        return re.sub(r"\s+", " ", claim.strip().lower())

    def add_proposals(
        self,
        proposals: Sequence[KnowledgeProposal],
        *,
        source_role: str,
        source_id: str,
        valid_evidence_refs: Iterable[str],
    ) -> Tuple[str, ...]:
        valid_evidence = set(valid_evidence_refs)
        created: list[str] = []
        for proposal in proposals:
            try:
                item_type = KnowledgeType(proposal.item_type)
                scope = KnowledgeScope(proposal.scope)
            except ValueError as exc:
                raise ValueError(
                    f"invalid knowledge taxonomy in {proposal.proposal_id}: {exc}"
                ) from exc
            missing_evidence = sorted(set(proposal.evidence_refs) - valid_evidence)
            if missing_evidence:
                raise ValueError(
                    f"knowledge proposal {proposal.proposal_id} references unknown evidence: "
                    + ", ".join(missing_evidence)
                )
            missing_conflicts = sorted(set(proposal.conflicts_with) - set(self._items))
            if missing_conflicts:
                raise ValueError(
                    f"knowledge proposal {proposal.proposal_id} references unknown conflicts: "
                    + ", ".join(missing_conflicts)
                )
            key = (
                item_type.value,
                scope.value,
                proposal.scope_ref,
                self._normalize_claim(proposal.claim),
            )
            existing = self._normalized_claims.get(key)
            if existing is not None:
                created.append(existing)
                continue
            self._counter += 1
            item_id = f"k{self._counter:04d}"
            status = (
                KnowledgeStatus.CONFLICTING
                if proposal.conflicts_with
                else KnowledgeStatus.PROPOSED
            )
            item = KnowledgeItem(
                item_id=item_id,
                item_type=item_type,
                scope=scope,
                scope_ref=proposal.scope_ref,
                claim=proposal.claim.strip(),
                evidence_refs=tuple(proposal.evidence_refs),
                confidence=proposal.confidence.strip() or "unspecified",
                status=status,
                source_role=source_role,
                source_id=source_id,
                conflicts_with=tuple(proposal.conflicts_with),
            )
            self._items[item_id] = item
            self._normalized_claims[key] = item_id
            created.append(item_id)
            if proposal.conflicts_with:
                for conflicting_id in proposal.conflicts_with:
                    other = self._items[conflicting_id]
                    reverse = tuple(dict.fromkeys((*other.conflicts_with, item_id)))
                    self._items[conflicting_id] = replace(
                        other,
                        status=KnowledgeStatus.CONFLICTING,
                        conflicts_with=reverse,
                    )
        return tuple(created)

    def apply_decisions(self, decisions: Sequence[KnowledgeDecision]) -> None:
        seen: set[str] = set()
        for decision in decisions:
            if decision.item_id in seen:
                raise ValueError(f"duplicate knowledge decision for {decision.item_id}")
            seen.add(decision.item_id)
            if decision.item_id not in self._items:
                raise ValueError(f"unknown knowledge item: {decision.item_id}")
            item = self._items[decision.item_id]
            try:
                status = KnowledgeStatus(decision.status)
            except ValueError as exc:
                raise ValueError(f"invalid knowledge status: {decision.status}") from exc
            if status is item.status:
                self._items[item.item_id] = replace(
                    item, decision_rationale=decision.rationale
                )
                continue
            if status not in self._ALLOWED_TRANSITIONS[item.status]:
                raise ValueError(
                    f"illegal knowledge transition {item.status.value}->{status.value} "
                    f"for {item.item_id}"
                )
            self._items[item.item_id] = replace(
                item,
                status=status,
                decision_rationale=decision.rationale,
            )

    def contains(self, item_id: str) -> bool:
        return item_id in self._items

    def snapshots(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(item.to_dict() for item in self._items.values())

    def items(self) -> Tuple[KnowledgeItem, ...]:
        return tuple(self._items.values())
