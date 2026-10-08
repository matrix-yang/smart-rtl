"""EDA-grounded search-tree primitives for RTL-Director v0.

The package is intentionally independent of the live ACE runner.  It can replay
persisted candidate/evaluator traces without importing an LLM client or issuing
API calls.
"""

from .node import ActionKind, EDAEvidence, EDAStage, NodeKind, NodeStatus, SearchNode
from .policy import Decision, DecisionContext, TreePolicy
from .tree import BacktrackEvent, SearchTree
from .contracts import CandidateArtifact, Diagnosis, PlanHypothesis, TaskContract
from .engine import LoopResult, RTLSearchEngine
from .instructions import GeneralACEInstructions, RoleInstructionProvider
from .roles import RoleSet
from .adaptive_contracts import AdaptiveActionKind, AdaptiveLoopResult
from .adaptive_engine import AdaptiveRTLDirectorEngine
from .adaptive_roles import AdaptiveRoleSet
from .skill_instructions import BWRTLSkillInstructions

__all__ = [
    "ActionKind",
    "BacktrackEvent",
    "CandidateArtifact",
    "Decision",
    "DecisionContext",
    "Diagnosis",
    "EDAEvidence",
    "EDAStage",
    "GeneralACEInstructions",
    "LoopResult",
    "NodeKind",
    "NodeStatus",
    "PlanHypothesis",
    "RTLSearchEngine",
    "RoleInstructionProvider",
    "RoleSet",
    "SearchNode",
    "SearchTree",
    "TaskContract",
    "TreePolicy",
    "AdaptiveActionKind",
    "AdaptiveLoopResult",
    "AdaptiveRTLDirectorEngine",
    "AdaptiveRoleSet",
    "BWRTLSkillInstructions",
]
