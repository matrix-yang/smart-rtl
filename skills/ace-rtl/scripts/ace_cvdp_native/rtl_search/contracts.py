"""Immutable public contracts shared by every RTL search role."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from .node import ActionKind
from .policy import DecisionContext


@dataclass(frozen=True, slots=True)
class TaskContract:
    task_id: str
    specification: str
    target_files: Tuple[str, ...]
    immutable_context: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.specification.strip():
            raise ValueError("public specification must be non-empty")
        if not self.target_files or any(not path.strip() for path in self.target_files):
            raise ValueError("at least one non-empty target file is required")
        if len(set(self.target_files)) != len(self.target_files):
            raise ValueError("target_files must be unique")
        context_paths = [path for path, _content in self.immutable_context]
        if len(set(context_paths)) != len(context_paths):
            raise ValueError("immutable context paths must be unique")
        overlap = set(self.target_files) & set(context_paths)
        if overlap:
            raise ValueError(
                "generated targets cannot also be immutable context: "
                + ", ".join(sorted(overlap))
            )

    @classmethod
    def from_parts(
        cls,
        *,
        task_id: str,
        specification: str,
        target_files: Tuple[str, ...],
        immutable_context: Optional[Mapping[str, str]] = None,
    ) -> "TaskContract":
        context = tuple(sorted((immutable_context or {}).items()))
        return cls(task_id, specification, tuple(target_files), context)

    def context_map(self) -> Dict[str, str]:
        return dict(self.immutable_context)


@dataclass(frozen=True, slots=True)
class PlanHypothesis:
    plan_id: str
    summary: str
    distinguishing_assumption: str
    invariants: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.plan_id.strip() or not self.summary.strip():
            raise ValueError("plan_id and summary must be non-empty")
        if not self.distinguishing_assumption.strip():
            raise ValueError("a plan must state its distinguishing assumption")


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    """Immutable generated files; no verifier or analyst may edit this object."""

    files: Tuple[Tuple[str, str], ...]

    @classmethod
    def from_mapping(cls, files: Mapping[str, str]) -> "CandidateArtifact":
        return cls(tuple(sorted((str(path), str(content)) for path, content in files.items())))

    def as_dict(self) -> Dict[str, str]:
        return dict(self.files)

    @property
    def candidate_hash(self) -> str:
        digest = hashlib.sha256()
        for path, content in self.files:
            digest.update(path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(content.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()[:16]

    def contract_issues(self, task: TaskContract) -> Tuple[str, ...]:
        actual = self.as_dict()
        expected = set(task.target_files)
        actual_paths = set(actual)
        issues = []
        for path in sorted(expected - actual_paths):
            issues.append(f"missing declared target file: {path}")
        for path in sorted(actual_paths - expected):
            issues.append(f"undeclared generated file: {path}")
        for path in sorted(expected & actual_paths):
            if not actual[path].strip():
                issues.append(f"empty generated target file: {path}")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class Diagnosis:
    failure_class: str
    rationale: str
    repair_obligation: str
    localized_failure: Optional[bool] = None
    plan_conflict: bool = False
    distinct_hypotheses_available: bool = False

    def to_decision_context(self) -> DecisionContext:
        return DecisionContext(
            localized_failure=self.localized_failure,
            plan_conflict=self.plan_conflict,
            distinct_hypotheses_available=self.distinct_hypotheses_available,
        )


@dataclass(frozen=True, slots=True)
class WorkItem:
    parent_id: str
    plan_id: str
    action: ActionKind
    diagnosis: Optional[Diagnosis] = None
