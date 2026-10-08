"""Fail-closed ChipVerilog boundary for immutable adaptive-search candidates.

The language-model roles receive only :func:`render_chipverilog_prompt` output.
Golden RTL, testbenches, proof scripts, and raw evaluator artifacts stay behind
``ChipVerilogIntegration`` and are never copied into ``EDAEvidence``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping

from ace_rtl_agent.utils.chipverilog_integration import (
    UNKNOWN_STATUSES,
    ChipVerilogIntegration,
    ChipVerilogTask,
    inspect_candidate_contract,
    render_chipverilog_prompt,
)

from ..processes import write_json
from .contracts import CandidateArtifact, TaskContract
from .node import EDAEvidence, EDAStage
from .scoring import normalize_failure_signature


CHIPVERILOG_TARGET_FILE = "rtl.sv"
_PASS_FORMAL_STATUSES = frozenset({"equivalent", "equivalent_bounded"})
_PASS_SIMULATION_PROOF_TYPES = frozenset({"simulation_tb"})
_STATUS_STAGE = {
    "invalid_candidate": EDAStage.INCOMPLETE,
    "compile_fail": EDAStage.COMPILE,
    "elaboration_fail": EDAStage.ELABORATE,
    "interface_fail": EDAStage.ELABORATE,
    "simulation_runtime_fail": EDAStage.SIMULATION,
    "function_fail": EDAStage.SIMULATION,
}
_FAILURE_CLASS = {
    "invalid_candidate": "candidate_contract",
    "compile_fail": "candidate_compile",
    "elaboration_fail": "candidate_elaboration",
    "interface_fail": "candidate_interface",
    "simulation_runtime_fail": "candidate_simulation_runtime",
    "function_fail": "candidate_functional",
}
_PRIVATE_BENCHMARK_PATH_RE = re.compile(
    r"(?:/(?:workspace|nfs|home|tmp)(?:/[A-Za-z0-9_.-]+)+)"
    r"(?:\.(?:sv|v|vh|svh|ys|py))?(?::(?P<line>\d+))?"
)
_PRIVATE_HIERARCHICAL_REFERENCE_RE = re.compile(
    r"\b(?:uut|dut|gold|golden|gate|reference|ref)"
    r"(?:\.[A-Za-z_$][A-Za-z0-9_$]*(?:\[[^\]\r\n]+\])?)+\b",
    re.IGNORECASE,
)
_PRIVATE_TESTBENCH_ID_RE = re.compile(
    r"\b(?:[A-Za-z0-9_$.-]+_tb_[A-Za-z0-9_$.-]+"
    r"|[A-Za-z0-9_$.-]+_tb\.(?:sv|v|svh|vh))\b",
    re.IGNORECASE,
)


def _redact_private_paths(value: Any) -> Any:
    """Remove benchmark filesystem identities while retaining diagnostic text."""

    if isinstance(value, str):
        def replacement(match: re.Match[str]) -> str:
            line = match.group("line")
            return f"<benchmark-private-artifact>:{line}" if line else "<benchmark-private-artifact>"

        return _PRIVATE_BENCHMARK_PATH_RE.sub(replacement, value)
    if isinstance(value, list):
        return [_redact_private_paths(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_private_paths(item) for key, item in value.items()}
    return value


def _redact_private_diagnostics(value: Any) -> Any:
    """Remove hidden testbench identities and hierarchical implementation refs."""

    if isinstance(value, str):
        value = _PRIVATE_HIERARCHICAL_REFERENCE_RE.sub(
            "<benchmark-private-hierarchical-reference>", value
        )
        return _PRIVATE_TESTBENCH_ID_RE.sub("<benchmark-private-testbench>", value)
    if isinstance(value, list):
        return [_redact_private_diagnostics(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _redact_private_diagnostics(item) for key, item in value.items()
        }
    return value


def task_contract_from_chipverilog_task(task: ChipVerilogTask) -> TaskContract:
    """Create a role-visible contract solely from the benchmark's public pack."""

    return TaskContract.from_parts(
        task_id=task.task_id,
        specification=render_chipverilog_prompt(task),
        target_files=(CHIPVERILOG_TARGET_FILE,),
        immutable_context={},
    )


def public_task_metadata(task: ChipVerilogTask, root: Path) -> Dict[str, Any]:
    """Return reproducibility metadata without reference or hidden artifacts."""

    root = root.resolve()
    prompt = render_chipverilog_prompt(task)
    return {
        "task_id": task.task_id,
        "source_family": task.source_family,
        "family": task.family,
        "target_top": task.target_top,
        "target_file": CHIPVERILOG_TARGET_FILE,
        "description_file": str(task.description_file.resolve().relative_to(root)),
        "include_files": [str(path.resolve().relative_to(root)) for path in task.include_files],
        "dependency_files": [
            str(path.resolve().relative_to(root)) for path in task.dependency_files
        ],
        "public_prompt_chars": len(prompt),
        "target_reference_exposed_to_roles": False,
    }


def _safe_feedback(result: Mapping[str, Any]) -> Dict[str, Any]:
    feedback = result.get("sanitized_feedback")
    if not isinstance(feedback, dict):
        return {
            "status": "evaluator_error",
            "verdict": "unknown",
            "failure_stage": "EVALUATOR_UNKNOWN",
            "reason": "sanitized evaluator feedback was unavailable",
            "formal_status": "unknown",
            "note": "This status is evaluator-inconclusive; do not infer correctness.",
        }
    # The incumbent integration constructs this allow-list.  Copy it rather
    # than forwarding any other result or raw report field.
    allowed = {
        "status",
        "verdict",
        "failure_stage",
        "flow",
        "proof_type",
        "reason",
        "interface_status",
        "formal_status",
        "counterexample_summary",
        "compile_gate_error_excerpt",
        "compile_gate_warning_excerpt",
        "generated_stub_modules",
        "expected_target_top",
        "candidate_top",
        "candidate_module_names",
        "note",
    }
    return {
        key: _redact_private_diagnostics(_redact_private_paths(feedback.get(key)))
        for key in sorted(allowed)
        if key in feedback
    }


def evidence_from_chipverilog_result(result: Mapping[str, Any]) -> EDAEvidence:
    """Translate one sanitized result; unknown or incomplete proof fails closed."""

    feedback = _safe_feedback(result)
    status = str(feedback.get("status") or "evaluator_error")
    verdict = str(feedback.get("verdict") or "unknown")
    formal_status = str(feedback.get("formal_status") or "unknown")
    proof_type = str(feedback.get("proof_type") or "unknown")
    flow = str(feedback.get("flow") or "unknown")
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    contract_valid = details.get("candidate_contract_valid") is True
    common_pass = (
        result.get("passed") is True
        and status == "pass"
        and verdict == "pass"
        and contract_valid
    )
    formal_proven = common_pass and formal_status in _PASS_FORMAL_STATUSES
    self_checking_simulation_pass = (
        common_pass
        and formal_status == "skip"
        and proof_type in _PASS_SIMULATION_PROOF_TYPES
        and flow == "simulation_tb"
    )
    proven_pass = formal_proven or self_checking_simulation_pass

    if proven_pass:
        stage = EDAStage.PASS
        failure_class = "none"
        signature = (
            "fail-closed formal equivalence pass"
            if formal_proven
            else "fail-closed self-checking simulation pass"
        )
        passed_subtests, total_subtests = 1, 1
    elif status in UNKNOWN_STATUSES or verdict == "unknown" or status == "pass":
        # A nominal pass with missing/unknown formal proof is also evaluator
        # infrastructure evidence, never candidate correctness.
        stage = EDAStage.INFRASTRUCTURE
        failure_class = "evaluator_unknown"
        signature = str(feedback.get("reason") or f"inconclusive evaluator status: {status}")
        passed_subtests, total_subtests = 0, 0
    else:
        stage = _STATUS_STAGE.get(status, EDAStage.INFRASTRUCTURE)
        failure_class = _FAILURE_CLASS.get(status, "evaluator_unknown")
        signature = str(feedback.get("reason") or f"ChipVerilog evaluator: {status}")
        passed_subtests, total_subtests = 0, 0 if status == "invalid_candidate" else 1

    return EDAEvidence(
        stage=stage,
        original_stage=str(feedback.get("failure_stage") or status).upper(),
        failure_class=failure_class,
        failure_signature=normalize_failure_signature(signature),
        complete_bundle=contract_valid and status != "invalid_candidate",
        passed_subtests=passed_subtests,
        total_subtests=total_subtests,
        public_feedback=json.dumps(feedback, indent=2, ensure_ascii=False),
    )


class ChipVerilogVerifier:
    """Serialize immutable candidates through the existing benchmark evaluator."""

    def __init__(
        self,
        *,
        chipverilog_root: Path,
        benchmark_task: ChipVerilogTask,
        run_dir: Path,
        evaluator_depth: int = 16,
        evaluator_timeout: int = 120,
        validate_tools: bool = True,
    ) -> None:
        self.chipverilog_root = chipverilog_root.resolve()
        self.benchmark_task = benchmark_task
        self.run_dir = run_dir.resolve()
        self.calls = 0
        self.evaluator_subprocess_calls = 0
        self.contract_rejections = 0
        self._lock = threading.Lock()
        self.integration = ChipVerilogIntegration(
            str(self.chipverilog_root),
            logging.getLogger("bw_rtl.chipverilog.verifier"),
            evaluator_depth=evaluator_depth,
            evaluator_timeout=evaluator_timeout,
            validate_tools=validate_tools,
        )
        self.integration.setup_datapoint_environment(
            benchmark_task.task_id,
            str(self.run_dir / "benchmark_boundary"),
            "bw-rtl-adaptive-verifier",
        )
        write_json(
            self.run_dir / "verifier_config.json",
            {
                "backend": "chipverilog_formal_equivalence_fail_closed",
                "task_id": benchmark_task.task_id,
                "target_top": benchmark_task.target_top,
                "target_file": CHIPVERILOG_TARGET_FILE,
                "evaluator_depth": evaluator_depth,
                "evaluator_timeout": evaluator_timeout,
                "evaluator_seed": None,
                "golden_or_hidden_exposed_to_roles": False,
                "pass_requires": {
                    "status": "pass",
                    "verdict": "pass",
                    "formal_status": sorted(_PASS_FORMAL_STATUSES),
                    "or_self_checking_simulation": {
                        "flow": "simulation_tb",
                        "proof_type": sorted(_PASS_SIMULATION_PROOF_TYPES),
                        "formal_status": "skip",
                    },
                    "candidate_contract_valid": True,
                },
            },
        )

    def verify(self, task: TaskContract, candidate: CandidateArtifact) -> EDAEvidence:
        with self._lock:
            return self._verify_locked(task, candidate)

    def _verify_locked(self, task: TaskContract, candidate: CandidateArtifact) -> EDAEvidence:
        if task.task_id != self.benchmark_task.task_id:
            raise ValueError(
                f"verifier task mismatch: {task.task_id} != {self.benchmark_task.task_id}"
            )
        self.calls += 1
        call_dir = self.run_dir / "verifier" / f"call_{self.calls:03d}_{candidate.candidate_hash}"
        call_dir.mkdir(parents=True, exist_ok=True)
        self._persist_candidate(call_dir, candidate)

        artifact_issues = candidate.contract_issues(task)
        candidate_text = candidate.as_dict().get(CHIPVERILOG_TARGET_FILE, "")
        module_contract = inspect_candidate_contract(
            candidate_text,
            self.benchmark_task.target_top,
        )
        write_json(
            call_dir / "candidate_contract.json",
            {
                "artifact_issues": list(artifact_issues),
                "module_contract": module_contract,
            },
        )
        if artifact_issues or not module_contract["valid"]:
            self.contract_rejections += 1
            reasons = [*artifact_issues, *module_contract["reasons"]]
            safe = {
                "status": "invalid_candidate",
                "verdict": "fail",
                "failure_stage": "CANDIDATE_CONTRACT",
                "reason": "; ".join(reasons),
                "formal_status": "not_run",
                "expected_target_top": self.benchmark_task.target_top,
                "candidate_top": module_contract["candidate_top"],
                "candidate_module_names": module_contract["candidate_module_names"],
            }
            result = {
                "passed": False,
                "details": {"candidate_contract_valid": False},
                "sanitized_feedback": safe,
            }
            write_json(call_dir / "sanitized_result.json", result)
            return evidence_from_chipverilog_result(result)

        self.evaluator_subprocess_calls += 1
        result = self.integration.test_rtl_code(task.task_id, candidate_text)
        safe_result = {
            "passed": result.get("passed") is True,
            "details": {
                "status": (result.get("details") or {}).get("status"),
                "verdict": (result.get("details") or {}).get("verdict"),
                "candidate_contract_valid": (result.get("details") or {}).get(
                    "candidate_contract_valid"
                ),
                "evaluation_index": (result.get("details") or {}).get("evaluation_index"),
            },
            "sanitized_feedback": _safe_feedback(result),
        }
        write_json(call_dir / "sanitized_result.json", safe_result)
        return evidence_from_chipverilog_result(safe_result)

    def close(self) -> None:
        self.integration.cleanup_global_resources()

    @staticmethod
    def _persist_candidate(call_dir: Path, candidate: CandidateArtifact) -> None:
        root = call_dir / "candidate"
        for name, code in candidate.files:
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe candidate path: {name}")
            path = root.joinpath(*rel.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code, encoding="utf-8")
        write_json(
            call_dir / "candidate_manifest.json",
            {
                "candidate_hash": candidate.candidate_hash,
                "files": [name for name, _ in candidate.files],
            },
        )
