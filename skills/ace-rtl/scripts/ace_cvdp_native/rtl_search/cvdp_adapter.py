"""Fail-closed native CVDP verifier for immutable search candidates."""

from __future__ import annotations

import copy
import os
import sys
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from ..cli import (
    normalize_agentic_row_for_native,
    resolve_target_files,
    select_generation_processor_class,
)
from ..diagnostics import (
    make_preflight_result,
    primary_failure_from_summary,
    summarize_result,
)
from ..harness import install_native_harness_patch, result_passed
from ..llm import build_evaluator_error_report
from ..processes import write_json
from ..sv import high_confidence_candidate_issues
from .contracts import CandidateArtifact, TaskContract
from .node import EDAEvidence, EDAStage
from .scoring import normalize_failure_signature, parse_eda_stage


def sanitize_live_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a row and erase any reference patch body before persistence."""

    normalized = normalize_agentic_row_for_native(row)
    target_files = resolve_target_files(normalized)
    if isinstance(normalized.get("patch"), dict):
        normalized["patch"] = {name: "" for name in target_files}
    if "response" in normalized:
        normalized["response"] = ""
    normalized.pop("subjective_reference", None)
    output = normalized.get("output")
    if isinstance(output, dict):
        output["context"] = {name: "" for name in target_files}
        if "response" in output:
            output["response"] = ""
    return normalized


def task_contract_from_row(row: Dict[str, Any]) -> TaskContract:
    """Build the public role contract without reading output/reference bodies."""

    target_files = tuple(resolve_target_files(row))
    input_data = row.get("input") if isinstance(row.get("input"), dict) else {}
    prompt = str(input_data.get("prompt") or row.get("prompt") or "").strip()
    raw_context = input_data.get("context")
    if not isinstance(raw_context, dict):
        raw_context = row.get("context") if isinstance(row.get("context"), dict) else {}

    immutable_context: Dict[str, str] = {}
    initial_target_parts: List[str] = []
    for raw_name, raw_content in raw_context.items():
        name = str(raw_name)
        content = str(raw_content)
        if name in target_files:
            initial_target_parts.append(
                f"# Initial public content for generated target: {name}\n"
                f"```systemverilog\n{content}\n```"
            )
        else:
            immutable_context[name] = content

    specification_parts = [prompt] if prompt else []
    specification_parts.extend(initial_target_parts)
    if not specification_parts:
        specification_parts.append("Implement every declared target using the supplied public context.")
    return TaskContract.from_parts(
        task_id=str(row.get("id", "")).strip(),
        specification="\n\n".join(specification_parts),
        target_files=target_files,
        immutable_context=immutable_context,
    )


class PrebuiltCandidateModel:
    """CVDP model protocol that injects one already-built candidate, with no LLM."""

    def __init__(self, candidate: CandidateArtifact, target_files: Tuple[str, ...]) -> None:
        self.candidate = candidate
        self.target_files = target_files
        self.requires_evaluation = True

    def prompt(
        self,
        prompt: str,
        schema: str = None,
        prompt_log: str = "",
        files: Optional[List[str]] = None,
        timeout: int = 60,
        category: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        requested = tuple(files or ())
        if requested != self.target_files:
            raise ValueError(
                "CVDP requested target files that differ from the immutable candidate contract: "
                f"expected={self.target_files}, requested={requested}"
            )
        if prompt_log:
            log_path = Path(prompt_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "Prebuilt RTL-Director candidate injected; no model call was made.\n\n" + prompt,
                encoding="utf-8",
            )
        mapping = self.candidate.as_dict()
        if len(requested) == 1:
            return {"direct_text": mapping[requested[0]]}, True
        return {"code": [{name: mapping[name]} for name in requested]}, True


def evidence_from_result(result: Dict[str, Any], *, public_feedback: str = "") -> EDAEvidence:
    summary = summarize_result(result)
    primary = primary_failure_from_summary(result, summary)
    tests = [test for test in result.get("tests", []) if isinstance(test, dict)]
    total = len(tests)
    passed_tests = sum(test.get("result", 1) == 0 for test in tests)
    if result_passed(result):
        raw_stage = "PASS"
        failure_class = "none"
        signature = "no failure detected"
    else:
        raw_stage = str(primary.get("stage") or "INFRA_SETUP")
        failure_class = {
            "LOCAL_PREFLIGHT": "candidate_preflight",
            "CANDIDATE_COMPILE": "candidate_compile",
            "CANDIDATE_FUNCTIONAL": "candidate_functional",
            "INFRA_SETUP": "infrastructure",
        }.get(raw_stage, "candidate_failure")
        signature = str(primary.get("signature") or primary.get("message") or raw_stage)
    return EDAEvidence(
        stage=parse_eda_stage(raw_stage),
        original_stage=raw_stage,
        failure_class=failure_class,
        failure_signature=normalize_failure_signature(signature),
        complete_bundle=raw_stage != "LOCAL_PREFLIGHT",
        passed_subtests=passed_tests,
        total_subtests=total,
        public_feedback=public_feedback,
    )


class CVDPNativeVerifier:
    """Run the existing CVDP Copilot processor on one frozen candidate."""

    def __init__(
        self,
        *,
        cvdp_repo: Path,
        row: Dict[str, Any],
        row_jsonl: Path,
        run_dir: Path,
        cid: str,
        evaluator_timeout: int,
        evaluator_seed: Optional[int] = 1,
    ) -> None:
        self.cvdp_repo = cvdp_repo
        self.row = copy.deepcopy(row)
        self.row_jsonl = row_jsonl
        self.run_dir = run_dir
        self.cid = cid
        self.evaluator_timeout = evaluator_timeout
        self.evaluator_seed = evaluator_seed
        self.calls = 0
        self._verify_lock = threading.Lock()
        write_json(
            self.run_dir / "verifier_config.json",
            {
                "backend": "native_cvdp",
                "evaluator_timeout": evaluator_timeout,
                "cocotb_random_seed": evaluator_seed,
            },
        )

    def verify(self, task: TaskContract, candidate: CandidateArtifact) -> EDAEvidence:
        # The native processor mutates process-wide environment variables and
        # assigns call directories.  Serialize only this EDA boundary; LLM
        # Builder/Analyst work in other branch pipelines may still overlap it.
        with self._verify_lock:
            return self._verify_locked(task, candidate)

    def _verify_locked(self, task: TaskContract, candidate: CandidateArtifact) -> EDAEvidence:
        self.calls += 1
        verify_dir = self.run_dir / "verifier" / f"call_{self.calls:03d}_{candidate.candidate_hash}"
        verify_dir.mkdir(parents=True, exist_ok=True)
        self._persist_candidate(verify_dir, candidate)

        preflight_issues: List[str] = []
        for name, code in candidate.files:
            preflight_issues.extend(
                f"{name}: {issue}" for issue in high_confidence_candidate_issues(code, self.cid)
            )
        if preflight_issues:
            result = make_preflight_result(preflight_issues, [])
            write_json(verify_dir / "preflight_result.json", result)
            summary = summarize_result(result)
            feedback = build_evaluator_error_report(result, summary)
            (verify_dir / "evaluator_error_report.txt").write_text(feedback, encoding="utf-8")
            return evidence_from_result(result, public_feedback=feedback)

        try:
            result = self._run_native(task, candidate, verify_dir)
        except Exception as exc:
            message = f"native CVDP evaluator infrastructure failure: {exc}"
            write_json(
                verify_dir / "infrastructure_error.json",
                {"stage": "INFRA_SETUP", "error": str(exc)},
            )
            return EDAEvidence(
                stage=EDAStage.INFRASTRUCTURE,
                original_stage="INFRA_SETUP",
                failure_class="infrastructure",
                failure_signature=normalize_failure_signature(message),
                complete_bundle=True,
                public_feedback=message,
            )

        write_json(verify_dir / "real_evaluator_result.json", result)
        summary = summarize_result(result)
        (verify_dir / "evaluator_summary.json").write_text(summary, encoding="utf-8")
        feedback = build_evaluator_error_report(result, summary)
        (verify_dir / "evaluator_error_report.txt").write_text(
            feedback, encoding="utf-8", errors="replace"
        )
        return evidence_from_result(result, public_feedback=feedback)

    def _run_native(
        self,
        task: TaskContract,
        candidate: CandidateArtifact,
        verify_dir: Path,
    ) -> Dict[str, Any]:
        if str(self.cvdp_repo) not in sys.path:
            sys.path.insert(0, str(self.cvdp_repo))
        install_native_harness_patch()
        from src import dataset_processor

        processor_cls = select_generation_processor_class(dataset_processor)
        processor = processor_cls(
            filename=str(self.row_jsonl),
            golden=False,
            threads=1,
            debug=False,
            host=False,
            prefix=str(verify_dir / "cvdp_work"),
            network_name=None,
            manage_network=False,
        )
        processor._sbj_model_config = None
        processor._sbj_model_instance = None
        processor.process_json()
        model = PrebuiltCandidateModel(candidate, tuple(task.target_files))
        prepared_id, obj, repo = processor.prepare(issue=task.task_id, model=model)
        if prepared_id != task.task_id:
            raise RuntimeError(f"unexpected prepared id {prepared_id} for {task.task_id}")

        previous_timeout = os.environ.get("CVDP_NATIVE_TIMEOUT")
        previous_seed = os.environ.get("COCOTB_RANDOM_SEED")
        os.environ["CVDP_NATIVE_TIMEOUT"] = str(self.evaluator_timeout)
        if self.evaluator_seed is not None:
            os.environ["COCOTB_RANDOM_SEED"] = str(self.evaluator_seed)
        try:
            result = processor.run(id=task.task_id, obj=obj, repo=repo, model=model)
        finally:
            if previous_timeout is None:
                os.environ.pop("CVDP_NATIVE_TIMEOUT", None)
            else:
                os.environ["CVDP_NATIVE_TIMEOUT"] = previous_timeout
            if previous_seed is None:
                os.environ.pop("COCOTB_RANDOM_SEED", None)
            else:
                os.environ["COCOTB_RANDOM_SEED"] = previous_seed
        return result

    @staticmethod
    def _persist_candidate(verify_dir: Path, candidate: CandidateArtifact) -> None:
        root = verify_dir / "candidate"
        for name, code in candidate.files:
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe candidate path: {name}")
            path = root.joinpath(*rel.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code, encoding="utf-8")
        write_json(
            verify_dir / "candidate_manifest.json",
            {"candidate_hash": candidate.candidate_hash, "files": [name for name, _ in candidate.files]},
        )
