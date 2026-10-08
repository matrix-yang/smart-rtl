# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ChipVerilog benchmark adapter for the reusable ACE-RTL agent loop.

The adapter keeps the benchmark evaluator behind a strict boundary: role models
receive the public task description, public macro definitions, selected fixed
dependency RTL, and a bounded evaluator summary.  The target reference RTL,
testbench, proof scripts, and raw evaluator artifacts are never added to a model
prompt.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


MODULE_RE = re.compile(
    r"(?m)^\s*module\s+(?:(?:automatic|static)\s+)?"
    r"([A-Za-z_][A-Za-z0-9_$]*)\b"
)
VERILOG_COMMENT_RE = re.compile(r"//[^\r\n]*|/\*.*?\*/", re.DOTALL)
CHIPVERILOG_TASK_KIND = "chipverilog_rtl_implementation"
CHIPVERILOG_TASK_MARKER = f"ACE_TASK_KIND={CHIPVERILOG_TASK_KIND}"
MODULE_NAME_FIELD_RE = re.compile(
    r"(?im)^\s*module\s+name\s*:\s*(?:\r?\n\s*)?[`'\"]?"
    r"([A-Za-z_][A-Za-z0-9_$]*)"
)
DOCUMENTED_MODULE_RE = re.compile(
    r"(?i)\bmodule\s+[`'\"]([A-Za-z_][A-Za-z0-9_$]*)[`'\"]"
)
INSTANCE_RE = re.compile(
    r"(?m)^\s*([A-Za-z_][A-Za-z0-9_$]*)\s*(?:#\s*\([^;]*?\))?\s+"
    r"[A-Za-z_][A-Za-z0-9_$]*\s*\("
)
KEYWORDS = {
    "always",
    "assign",
    "begin",
    "case",
    "casex",
    "casez",
    "else",
    "for",
    "function",
    "generate",
    "if",
    "initial",
    "module",
    "primitive",
    "task",
    "while",
}
SOURCE_TO_DES_FAMILY = {
    "verilog_cordic_core": "cordic_core",
    "or1200_hp": "or1200",
}
RESULT_NAME_OVERRIDES = {"fpu_double": "fpu"}
UNKNOWN_STATUSES = {
    "equivalence_timeout",
    "equivalence_error",
    "reference_fail",
    "evaluator_error",
}
STATUS_PROGRESS = {
    "pass": 0,
    "function_fail": 100,
    "interface_fail": 300,
    "elaboration_fail": 400,
    "compile_fail": 500,
    "simulation_runtime_fail": 600,
    "invalid_candidate": 900,
}
STATUS_STAGE = {
    "pass": "VERIFIED_PASS",
    "function_fail": "FUNCTIONAL",
    "interface_fail": "INTERFACE",
    "elaboration_fail": "ELABORATION",
    "compile_fail": "CANDIDATE_COMPILE",
    "simulation_runtime_fail": "SIMULATION_RUNTIME",
    "invalid_candidate": "CANDIDATE_CONTRACT",
    "equivalence_timeout": "EVALUATOR_UNKNOWN",
    "equivalence_error": "EVALUATOR_UNKNOWN",
    "reference_fail": "EVALUATOR_UNKNOWN",
    "evaluator_error": "EVALUATOR_UNKNOWN",
}


@dataclass(frozen=True)
class ChipVerilogTask:
    """Public task pack plus fixed dependency metadata."""

    task_id: str
    source_family: str
    family: str
    target_top: str
    description_file: Path
    include_files: tuple[Path, ...]
    dependency_files: tuple[Path, ...]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _target_top_from_description(description: Path) -> str:
    text = _read_text(description)
    match = (
        MODULE_RE.search(text)
        or MODULE_NAME_FIELD_RE.search(text)
        or DOCUMENTED_MODULE_RE.search(text)
    )
    if not match:
        raise ValueError(f"cannot resolve documented target module in {description}")
    return match.group(1)


def _task_identity(description: Path, src_root: Path) -> tuple[str, str, str]:
    relative = description.relative_to(src_root)
    source_family = relative.parts[0]
    family = SOURCE_TO_DES_FAMILY.get(source_family, source_family)
    leaf = description.parent.name
    task_id = RESULT_NAME_OVERRIDES.get(leaf, leaf)
    if family == "mips_16":
        task_id = f"mips_{leaf}"
    return task_id, source_family, family


def _reference_dir_name(task_id: str, family: str) -> str:
    if family == "mips_16" and task_id.startswith("mips_"):
        return task_id[len("mips_") :]
    return task_id


def _fixed_module_file(chipverilog_root: Path, task_id: str, family: str) -> Optional[Path]:
    """Resolve another task's fixed RTL without ever selecting the active target."""

    module_dir = chipverilog_root / "Des" / family / _reference_dir_name(task_id, family)
    if not module_dir.is_dir():
        return None
    preferred = module_dir / f"{module_dir.name}.v"
    if preferred.is_file():
        return preferred
    candidates = [
        path
        for path in sorted((*module_dir.glob("*.v"), *module_dir.glob("*.sv")))
        if "tb" not in path.stem.lower() and "test" not in path.stem.lower()
    ]
    return candidates[0] if candidates else None


def _family_include_files(chipverilog_root: Path, family: str) -> tuple[Path, ...]:
    family_root = chipverilog_root / "Des" / family
    return tuple(
        sorted(
            path
            for path in family_root.glob("*.v")
            if path.name == "timescale.v"
            or "defs" in path.name.lower()
            or "defines" in path.name.lower()
        )
    )


def _mentioned_module_names(text: str, module_names: Iterable[str]) -> List[str]:
    mentioned = []
    for name in sorted(module_names):
        if re.search(
            rf"(?<![A-Za-z0-9_$]){re.escape(name)}(?![A-Za-z0-9_$])",
            text,
            flags=re.IGNORECASE,
        ):
            mentioned.append(name)
    return mentioned


def _dependency_description_context(description_text: str) -> str:
    """Keep public paragraphs that explicitly describe instantiated hierarchy."""

    markers = re.compile(r"(?i)\b(?:instantiat(?:e|es|ed|ing)|submodules?)\b")
    if re.search(r"(?i)\btop-level\b", description_text) and markers.search(description_text):
        return description_text
    sentences = re.split(r"(?<=[.!?])\s+|\n+", description_text)
    return "\n".join(sentence for sentence in sentences if markers.search(sentence))


def _dependency_files(
    *,
    chipverilog_root: Path,
    target_task_id: str,
    target_top: str,
    family: str,
    description_text: str,
    module_catalog: Dict[str, List[tuple[str, str, Path]]],
) -> tuple[Path, ...]:
    """Find fixed dependencies without reading the target reference implementation.

    First-level dependencies must be named by the public description.  Recursive
    dependencies are then obtained only from already-selected fixed dependency
    files.  This deliberately avoids using the hidden target RTL as a retrieval
    oracle.
    """

    family_modules: Dict[str, tuple[str, str, Path]] = {}
    for name, records in module_catalog.items():
        if name == target_top:
            continue
        choices = [
            record
            for record in records
            if record[1] == family and record[0] != target_task_id
        ]
        if choices:
            # Prefer the task whose result id matches its module name. This
            # disambiguates the standalone fpu_mul task from fpu_mul_pipeline.
            choices.sort(key=lambda record: (record[0] != name, record[0]))
            family_modules[name] = choices[0]
    hierarchy_context = _dependency_description_context(description_text)
    frontier = _mentioned_module_names(hierarchy_context, family_modules)
    selected: List[Path] = []
    seen_modules = {target_top}
    target_reference_dir = (
        chipverilog_root
        / "Des"
        / family
        / _reference_dir_name(target_task_id, family)
    ).resolve()

    while frontier:
        module_name = frontier.pop(0)
        if module_name in seen_modules:
            continue
        seen_modules.add(module_name)
        record = family_modules.get(module_name)
        if record is None:
            continue
        dependency = record[2].resolve()
        if dependency.parent == target_reference_dir or dependency in selected:
            continue
        selected.append(dependency)
        instantiated = [
            name
            for name in INSTANCE_RE.findall(_read_text(dependency))
            if name not in KEYWORDS and name not in seen_modules and name in family_modules
        ]
        frontier.extend(instantiated)

    return tuple(sorted(selected))


def discover_chipverilog_tasks(chipverilog_root: Path | str) -> List[ChipVerilogTask]:
    """Discover the 64 public ChipVerilog task descriptions."""

    root = Path(chipverilog_root).resolve()
    src_root = root / "Src"
    des_root = root / "Des"
    if not src_root.is_dir() or not des_root.is_dir():
        raise FileNotFoundError(f"ChipVerilog Src/ and Des/ not found under {root}")

    description_files = sorted(src_root.glob("**/des/**/description.txt"))
    module_catalog: Dict[str, List[tuple[str, str, Path]]] = {}
    identities: Dict[Path, tuple[str, str, str, str]] = {}
    for description in description_files:
        task_id, source_family, family = _task_identity(description, src_root)
        target_top = _target_top_from_description(description)
        reference_file = _fixed_module_file(root, task_id, family)
        if reference_file is None:
            raise FileNotFoundError(f"cannot map public task {task_id} into Des/{family}")
        module_catalog.setdefault(target_top, []).append((task_id, family, reference_file))
        identities[description] = (task_id, source_family, family, target_top)

    tasks: List[ChipVerilogTask] = []
    for description in description_files:
        task_id, source_family, family, target_top = identities[description]
        description_text = _read_text(description)
        dependencies = _dependency_files(
            chipverilog_root=root,
            target_task_id=task_id,
            target_top=target_top,
            family=family,
            description_text=description_text,
            module_catalog=module_catalog,
        )
        tasks.append(
            ChipVerilogTask(
                task_id=task_id,
                source_family=source_family,
                family=family,
                target_top=target_top,
                description_file=description,
                include_files=_family_include_files(root, family),
                dependency_files=dependencies,
            )
        )
    return tasks


def _bounded_file_sections(
    files: Sequence[Path],
    *,
    tag: str,
    character_budget: int,
) -> tuple[str, List[str]]:
    sections: List[str] = []
    omitted: List[str] = []
    used = 0
    for path in files:
        content = _read_text(path)
        section = f'\n<{tag} name="{path.name}">\n{content}\n</{tag}>\n'
        if used + len(section) > character_budget:
            omitted.append(path.name)
            continue
        sections.append(section)
        used += len(section)
    return "".join(sections), omitted


def render_chipverilog_prompt(
    task: ChipVerilogTask,
    *,
    dependency_character_budget: int = 240_000,
) -> str:
    """Render only public/allowed model context for one task."""

    include_text, omitted_includes = _bounded_file_sections(
        task.include_files,
        tag="include_file",
        character_budget=max(20_000, dependency_character_budget // 4),
    )
    dependency_text, omitted_dependencies = _bounded_file_sections(
        task.dependency_files,
        tag="fixed_dependency_rtl",
        character_budget=max(0, dependency_character_budget),
    )
    omitted = omitted_includes + omitted_dependencies
    omitted_text = ", ".join(omitted) if omitted else "(none)"
    description = _read_text(task.description_file).strip()
    return f"""{CHIPVERILOG_TASK_MARKER}
ACE_BENCHMARK=chipverilog
ACE_ARTIFACT_KIND=synthesizable_rtl
ACE_EXPECTED_TARGET_TOP={task.target_top}

ChipVerilog RTL generation contract:
- Generate one complete Verilog/SystemVerilog implementation of module `{task.target_top}`.
- Return complete replacement RTL only; do not return a diff or explanatory prose.
- This is never a testbench, stimulus, checker, coverage, or assertion-generation task.
- Preserve every documented port, width, macro branch, reset behavior, latency, and cycle contract.
- Do not redefine fixed dependency modules.
- Do not instantiate or refer to a reference/golden implementation.
- The target reference RTL and benchmark testbench are hidden from all role models.
- Evaluator feedback is a bounded status summary and may be inconclusive; timeout/tool/reference
  failures are UNKNOWN and must not be treated as proof that the candidate is functionally wrong.

<target_specification task_id="{task.task_id}" family="{task.family}">
{description}
</target_specification>

<public_macro_context>
{include_text or '(none)'}
</public_macro_context>

<fixed_dependency_context>
{dependency_text or '(none; evaluator supplies any required fixed support RTL)'}
</fixed_dependency_context>

<dependency_context_omitted_due_to_budget>
{omitted_text}
</dependency_context_omitted_due_to_budget>
"""


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "unknown"


def _bounded_text(value: Any, limit: int = 4_000) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _bounded_reason(status: str, value: Any) -> str:
    text = _bounded_text(value)
    if status != "function_fail":
        return text
    fragments = [fragment.strip() for fragment in text.split("|") if fragment.strip()]
    if len(fragments) <= 1:
        return text
    return fragments[0] + f" | ...[{len(fragments) - 1} additional mismatch excerpts withheld]"


def _json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _utc_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{os.getpid()}"


def summarize_llm_telemetry(path: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "usage_available_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": 0.0,
        "by_role": {},
    }
    if not path.is_file():
        return summary
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("event") != "llm_call":
            continue
        role = str(event.get("role") or "unknown")
        role_summary = summary["by_role"].setdefault(
            role,
            {
                "calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
                "total_tokens": 0,
                "latency_seconds": 0.0,
            },
        )
        summary["calls"] += 1
        role_summary["calls"] += 1
        success = event.get("success") is True
        outcome_key = "successful_calls" if success else "failed_calls"
        summary[outcome_key] += 1
        role_summary[outcome_key] += 1
        if event.get("usage_available") is True:
            summary["usage_available_calls"] += 1
        usage = event.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        for key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "total_tokens",
        ):
            value = usage.get(key, 0)
            value = int(value) if isinstance(value, (int, float)) else 0
            summary[key] += value
            role_summary[key] += value
        latency = event.get("latency_seconds", 0.0)
        latency = float(latency) if isinstance(latency, (int, float)) else 0.0
        summary["latency_seconds"] += latency
        role_summary["latency_seconds"] += latency
    return summary


def inspect_candidate_contract(candidate_text: str, expected_target_top: str) -> Dict[str, Any]:
    """Validate the public candidate artifact before any golden-backed tool runs.

    ChipVerilog asks for one replacement target module.  A missing or different
    top must fail closed; otherwise the benchmark dependency resolver can satisfy
    a testbench-only candidate with the hidden golden target.
    """

    text = str(candidate_text or "")
    uncommented_text = VERILOG_COMMENT_RE.sub("", text)
    module_names = MODULE_RE.findall(uncommented_text)
    target_definition_count = sum(name == expected_target_top for name in module_names)
    unexpected_modules = [name for name in module_names if name != expected_target_top]
    reasons: List[str] = []
    if not module_names:
        reasons.append("candidate contains no module declaration")
    if target_definition_count == 0:
        reasons.append(f"candidate does not define expected target `{expected_target_top}`")
    elif target_definition_count > 1:
        reasons.append(
            f"candidate defines expected target `{expected_target_top}` {target_definition_count} times"
        )
    if unexpected_modules:
        reasons.append(
            "candidate defines unexpected module(s): " + ", ".join(unexpected_modules)
        )
    return {
        "valid": not reasons,
        "task_kind": CHIPVERILOG_TASK_KIND,
        "expected_target_top": expected_target_top,
        "candidate_top": module_names[0] if module_names else "",
        "candidate_module_names": module_names,
        "target_definition_count": target_definition_count,
        "unexpected_module_names": unexpected_modules,
        "candidate_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "reasons": reasons,
    }


class ChipVerilogIntegration:
    """ACE-RTL integration backed by ChipVerilog's real golden evaluator."""

    def __init__(
        self,
        chipverilog_path: str,
        logger,
        max_retries: int = 3,
        initial_backoff: float = 5.0,
        *,
        evaluator_depth: int = 16,
        evaluator_timeout: int = 120,
        dependency_character_budget: Optional[int] = None,
        validate_tools: bool = True,
    ) -> None:
        del max_retries, initial_backoff  # Kept for the common integration constructor API.
        self.chipverilog_path = Path(chipverilog_path).resolve()
        self.logger = logger
        self.evaluator_depth = max(1, int(evaluator_depth))
        self.evaluator_timeout = max(1, int(evaluator_timeout))
        if dependency_character_budget is None:
            dependency_character_budget = int(
                os.environ.get("ACE_RTL_CHIPVERILOG_DEPENDENCY_CHARS", "240000")
            )
        self.dependency_character_budget = max(0, dependency_character_budget)
        self.evaluator_script = self.chipverilog_path / "tools" / "formal_equivalence.py"
        self.iverilog = os.environ.get("ACE_RTL_IVERILOG", "iverilog")
        self.vvp = os.environ.get("ACE_RTL_VVP", "vvp")
        self.yosys = os.environ.get("ACE_RTL_YOSYS", "yosys")
        self.run_id = _safe_component(os.environ.get("ACE_RTL_RUN_ID", _utc_run_id()))
        self.tasks = {task.task_id: task for task in discover_chipverilog_tasks(self.chipverilog_path)}
        self.current_task: Optional[ChipVerilogTask] = None
        self.current_prompt: Optional[str] = None
        self.evaluation_index = 0
        self.last_result: Optional[Dict[str, Any]] = None
        self.telemetry_file: Optional[Path] = None
        self._validate_environment(validate_tools=validate_tools)

    def _validate_environment(self, *, validate_tools: bool) -> None:
        if not self.evaluator_script.is_file():
            raise FileNotFoundError(f"ChipVerilog evaluator not found: {self.evaluator_script}")
        if len(self.tasks) != 64:
            raise RuntimeError(
                f"expected 64 ChipVerilog tasks, discovered {len(self.tasks)} under {self.chipverilog_path}"
            )
        if not validate_tools:
            return
        missing = [
            executable
            for executable in (self.iverilog, self.vvp, self.yosys)
            if shutil.which(executable) is None and not Path(executable).is_file()
        ]
        if missing:
            raise RuntimeError(
                "ChipVerilog evaluator tools missing from PATH: " + ", ".join(missing)
            )

    def get_problem_list(self) -> List[str]:
        return sorted(self.tasks)

    def load_problem(self, problem_name: str) -> Dict[str, Any]:
        try:
            task = self.tasks[problem_name]
        except KeyError as exc:
            raise KeyError(
                f"unknown ChipVerilog task {problem_name!r}; available tasks: "
                + ", ".join(self.get_problem_list())
            ) from exc
        prompt = render_chipverilog_prompt(
            task,
            dependency_character_budget=self.dependency_character_budget,
        )
        self.current_task = task
        self.current_prompt = prompt
        return {
            "problem_name": problem_name,
            "task_id": task.task_id,
            "family": task.family,
            "target_top": task.target_top,
            "prompt": prompt,
        }

    def setup_datapoint_environment(
        self,
        problem_name: str,
        base_output_dir: str,
        model_name: str,
    ) -> None:
        task_data = self.load_problem(problem_name)
        self.base_output_dir = Path(base_output_dir).resolve()
        self.model_name = model_name
        self.model_slug = _safe_component(model_name)
        self.problem_output_dir = (
            self.base_output_dir
            / "chipverilog"
            / self.run_id
            / self.model_slug
            / problem_name
        )
        self.problem_output_dir.mkdir(parents=True, exist_ok=True)
        self.base_path = self.problem_output_dir
        self.reports_dir = self.problem_output_dir / "evaluator_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.expected_rtl_filename = "rtl.sv"
        self.telemetry_file = self.problem_output_dir / "telemetry.jsonl"
        os.environ["ACE_RTL_LLM_TELEMETRY_FILE"] = str(self.telemetry_file)
        os.environ["ACE_RTL_TASK_ID"] = task_data["task_id"]
        self.evaluation_index = 0
        self.last_result = None
        (self.problem_output_dir / "prompt.md").write_text(
            self.current_prompt or "", encoding="utf-8"
        )
        public_metadata = {
            **task_data,
            "prompt": None,
            "task_kind": CHIPVERILOG_TASK_KIND,
            "artifact_kind": "synthesizable_rtl",
            "run_id": self.run_id,
            "model": model_name,
            "description_file": str(
                self.current_task.description_file.relative_to(self.chipverilog_path)
            ),
            "include_files": [
                str(path.relative_to(self.chipverilog_path))
                for path in self.current_task.include_files
            ],
            "dependency_files": [
                str(path.relative_to(self.chipverilog_path))
                for path in self.current_task.dependency_files
            ],
            "target_reference_exposed_to_roles": False,
        }
        (self.problem_output_dir / "task.json").write_text(
            json.dumps(public_metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def get_problem_prompt(self, problem_name: str) -> str:
        if self.current_task is None or self.current_task.task_id != problem_name:
            self.load_problem(problem_name)
        return self.current_prompt or ""

    def get_sanitized_testbench(self) -> None:
        """ChipVerilog testbenches remain entirely behind the evaluator boundary."""

        return None

    def _record_telemetry(self, payload: Dict[str, Any]) -> None:
        if self.telemetry_file is None:
            return
        with self.telemetry_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _sanitized_feedback(self, row: Dict[str, Any]) -> Dict[str, Any]:
        status = str(row.get("status") or "evaluator_error")
        return {
            "status": status,
            "verdict": "pass" if status == "pass" else ("unknown" if status in UNKNOWN_STATUSES else "fail"),
            "failure_stage": STATUS_STAGE.get(status, "EVALUATOR_UNKNOWN"),
            "flow": _bounded_text(row.get("flow"), 200),
            "proof_type": _bounded_text(row.get("proof_type"), 200),
            "reason": _bounded_reason(status, row.get("reason")),
            "interface_status": _bounded_text(row.get("interface_status"), 500),
            "formal_status": _bounded_text(row.get("formal_status"), 500),
            "counterexample_summary": _bounded_text(row.get("counterexample_summary")),
            "compile_gate_error_excerpt": _bounded_text(row.get("compile_gate_error_excerpt")),
            "compile_gate_warning_excerpt": _bounded_text(row.get("compile_gate_warning_excerpt"), 2_000),
            "generated_stub_modules": _json_list(row.get("generated_stub_modules")),
            "expected_target_top": _bounded_text(row.get("expected_target_top"), 200),
            "candidate_top": _bounded_text(row.get("candidate_top"), 200),
            "candidate_module_names": _json_list(row.get("candidate_module_names")),
            "note": (
                "This status is evaluator-inconclusive; do not infer candidate correctness."
                if status in UNKNOWN_STATUSES
                else ""
            ),
        }

    def _evaluator_error_result(
        self,
        message: str,
        *,
        elapsed: float,
        stdout: str = "",
        stderr: str = "",
    ) -> Dict[str, Any]:
        row = {
            "status": "evaluator_error",
            "flow": "evaluator",
            "reason": _bounded_text(message),
            "formal_status": "unknown",
            "counterexample_summary": "",
            "compile_gate_error_excerpt": _bounded_text(stderr or stdout, 2_000),
        }
        return self._result_from_row(row, elapsed=elapsed)

    def _result_from_row(self, row: Dict[str, Any], *, elapsed: float) -> Dict[str, Any]:
        feedback = self._sanitized_feedback(row)
        status = feedback["status"]
        progress_metric = STATUS_PROGRESS.get(status, -1)
        error_report = json.dumps(feedback, indent=2, ensure_ascii=False)
        compilation_success = status not in {
            "compile_fail",
            "invalid_candidate",
            "evaluator_error",
        }
        return {
            "passed": status == "pass",
            "compilation_success": compilation_success,
            "error_message": None if status == "pass" else f"ChipVerilog evaluator: {status}",
            "error_report": None if status == "pass" else error_report,
            "details": {
                "status": status,
                "verdict": feedback["verdict"],
                "failure_stage": feedback["failure_stage"],
                # ACE-RTL's incumbent controller expects a lower-is-better integer.
                # Unknown evaluator states deliberately use -1 and do not enter
                # mismatch-based best-candidate comparisons.
                "mismatches": progress_metric,
                "progress_metric": progress_metric,
                "per_signal_mismatches": [],
                "proof_type": feedback["proof_type"],
                "elapsed_seconds": elapsed,
            },
            "sanitized_feedback": feedback,
        }

    def _finalize_evaluation_result(
        self,
        *,
        result: Dict[str, Any],
        candidate_text: str,
        eval_index: int,
        eval_dir: Path,
        candidate_contract: Dict[str, Any],
    ) -> Dict[str, Any]:
        result["rtl_code"] = candidate_text
        result["details"]["evaluation_index"] = eval_index
        result["details"]["evaluator_run_dir"] = str(eval_dir)
        result["details"]["candidate_contract_valid"] = candidate_contract["valid"]
        result["details"]["candidate_sha256"] = candidate_contract["candidate_sha256"]
        feedback_path = self.reports_dir / f"evaluation_{eval_index:03d}.json"
        feedback_path.write_text(
            json.dumps(result["sanitized_feedback"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.reports_dir / f"evaluation_{eval_index:03d}.txt").write_text(
            result["error_report"] or "ChipVerilog evaluator: pass\n",
            encoding="utf-8",
        )
        self.last_result = result
        self._record_telemetry(
            {
                "event": "chipverilog_evaluation",
                "timestamp": time.time(),
                "task_id": self.current_task.task_id if self.current_task else None,
                "evaluation_index": eval_index,
                "status": result["details"]["status"],
                "verdict": result["details"]["verdict"],
                "passed": result["passed"],
                "elapsed_seconds": result["details"]["elapsed_seconds"],
                "evaluator_run_dir": str(eval_dir),
                "candidate_contract_valid": candidate_contract["valid"],
                "expected_target_top": candidate_contract["expected_target_top"],
                "candidate_top": candidate_contract["candidate_top"],
                "candidate_module_names": candidate_contract["candidate_module_names"],
                "candidate_sha256": candidate_contract["candidate_sha256"],
            }
        )
        return result

    def test_rtl_code(self, problem_name: str, rtl_code: str) -> Dict[str, Any]:
        if self.current_task is None or self.current_task.task_id != problem_name:
            self.load_problem(problem_name)
        if not hasattr(self, "problem_output_dir"):
            raise RuntimeError("setup_datapoint_environment must be called before test_rtl_code")

        self.evaluation_index += 1
        eval_index = self.evaluation_index
        candidate_text = str(rtl_code or "")
        current_rtl = self.problem_output_dir / "rtl.sv"
        current_rtl.write_text(candidate_text, encoding="utf-8")
        history_dir = self.problem_output_dir / "candidate_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        (history_dir / f"candidate_{eval_index:03d}.sv").write_text(
            candidate_text, encoding="utf-8"
        )

        eval_dir = self.problem_output_dir / "evaluator_runs" / f"eval_{eval_index:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        candidate_contract = inspect_candidate_contract(
            candidate_text,
            self.current_task.target_top,
        )
        (eval_dir / "candidate_contract.json").write_text(
            json.dumps(candidate_contract, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not candidate_contract["valid"]:
            invocation = {
                "evaluation_index": eval_index,
                "skipped": True,
                "skip_reason": "candidate contract failed before golden-backed evaluation",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            (eval_dir / "evaluator_invocation.json").write_text(
                json.dumps(invocation, indent=2) + "\n", encoding="utf-8"
            )
            reason = "; ".join(candidate_contract["reasons"])
            row = {
                "status": "invalid_candidate",
                "flow": "candidate_contract",
                "proof_type": "candidate_contract",
                "reason": reason,
                "compile_gate_error_excerpt": reason,
                "expected_target_top": candidate_contract["expected_target_top"],
                "candidate_top": candidate_contract["candidate_top"],
                "candidate_module_names": candidate_contract["candidate_module_names"],
            }
            result = self._result_from_row(row, elapsed=0.0)
            return self._finalize_evaluation_result(
                result=result,
                candidate_text=candidate_text,
                eval_index=eval_index,
                eval_dir=eval_dir,
                candidate_contract=candidate_contract,
            )

        result_root = eval_dir / "Result"
        candidate_dir = result_root / self.model_slug / problem_name
        report_dir = eval_dir / "reports"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_file = candidate_dir / f"{problem_name}_t1.v"
        candidate_file.write_text(candidate_text, encoding="utf-8")
        command = [
            sys.executable,
            str(self.evaluator_script),
            "verify",
            str(candidate_dir),
            "--result-root",
            str(result_root),
            "--report-dir",
            str(report_dir),
            "--depth",
            str(self.evaluator_depth),
            "--timeout",
            str(self.evaluator_timeout),
            "--iverilog",
            self.iverilog,
            "--vvp",
            self.vvp,
            "--yosys",
            self.yosys,
        ]
        invocation = {
            "evaluation_index": eval_index,
            "command": command,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (eval_dir / "evaluator_invocation.json").parent.mkdir(parents=True, exist_ok=True)
        (eval_dir / "evaluator_invocation.json").write_text(
            json.dumps(invocation, indent=2) + "\n", encoding="utf-8"
        )

        start = time.monotonic()
        try:
            process = subprocess.run(
                command,
                cwd=self.chipverilog_path,
                text=True,
                capture_output=True,
                timeout=max(60, self.evaluator_timeout * 4 + 30),
                check=False,
            )
            elapsed = time.monotonic() - start
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - start
            stdout = _bounded_text(exc.stdout, 8_000)
            stderr = _bounded_text(exc.stderr, 8_000)
            (eval_dir / "evaluator.log").write_text(
                f"stdout:\n{stdout}\n\nstderr:\n{stderr}\n", encoding="utf-8"
            )
            result = self._evaluator_error_result(
                f"evaluator wrapper timed out after {elapsed:.1f}s",
                elapsed=elapsed,
                stdout=stdout,
                stderr=stderr,
            )
        else:
            (eval_dir / "evaluator.log").write_text(
                f"return_code: {process.returncode}\nstdout:\n{process.stdout}\n\nstderr:\n{process.stderr}\n",
                encoding="utf-8",
            )
            report_path = report_dir / self.model_slug / f"{problem_name}.json"
            if process.returncode != 0 or not report_path.is_file():
                result = self._evaluator_error_result(
                    f"evaluator exited with status {process.returncode}; report_present={report_path.is_file()}",
                    elapsed=elapsed,
                    stdout=process.stdout,
                    stderr=process.stderr,
                )
            else:
                try:
                    payload = json.loads(report_path.read_text(encoding="utf-8"))
                    rows = payload.get("results")
                    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
                        raise ValueError(f"expected exactly one evaluator result, got {type(rows).__name__}")
                    evaluator_row = rows[0]
                    reported_top = str(evaluator_row.get("candidate_top") or "")
                    if reported_top != self.current_task.target_top:
                        reason = (
                            "evaluator candidate-top postcondition failed: expected "
                            f"`{self.current_task.target_top}`, got `{reported_top or '(missing)'}`"
                        )
                        evaluator_row = {
                            "status": "invalid_candidate",
                            "flow": "candidate_contract",
                            "proof_type": "candidate_contract",
                            "reason": reason,
                            "compile_gate_error_excerpt": reason,
                            "expected_target_top": self.current_task.target_top,
                            "candidate_top": reported_top,
                            "candidate_module_names": candidate_contract[
                                "candidate_module_names"
                            ],
                        }
                    result = self._result_from_row(evaluator_row, elapsed=elapsed)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    result = self._evaluator_error_result(
                        f"invalid evaluator report: {exc}", elapsed=elapsed
                    )

        return self._finalize_evaluation_result(
            result=result,
            candidate_text=candidate_text,
            eval_index=eval_index,
            eval_dir=eval_dir,
            candidate_contract=candidate_contract,
        )

    def write_summary(self, final_report: Dict[str, Any]) -> Path:
        task = self.current_task
        status = None
        verdict = None
        proof_type = None
        if self.last_result:
            status = self.last_result["details"].get("status")
            verdict = self.last_result["details"].get("verdict")
            proof_type = self.last_result["details"].get("proof_type")
        summary = {
            "task_id": task.task_id if task else final_report.get("datapoint_name"),
            "family": task.family if task else None,
            "target_top": task.target_top if task else None,
            "run_id": self.run_id,
            "model": self.model_name,
            "success": bool(final_report.get("success", False)),
            "status": status,
            "verdict": verdict,
            "proof_type": proof_type,
            "total_iterations": final_report.get("total_iterations", 0),
            "total_time": final_report.get("total_time", 0.0),
            "evaluator_calls": self.evaluation_index,
            "llm_usage": summarize_llm_telemetry(self.telemetry_file),
            "unknown": status in UNKNOWN_STATUSES if status else True,
            "target_reference_exposed_to_roles": False,
        }
        summary_path = self.problem_output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._record_telemetry({"event": "task_complete", **summary})
        return summary_path

    def cleanup_test_environment(self, problem_name: str) -> None:
        self.logger.debug("ChipVerilog task cleanup complete: %s", problem_name)

    def cleanup_global_resources(self) -> None:
        if self.telemetry_file is not None:
            configured = os.environ.get("ACE_RTL_LLM_TELEMETRY_FILE")
            if configured == str(self.telemetry_file):
                os.environ.pop("ACE_RTL_LLM_TELEMETRY_FILE", None)
                os.environ.pop("ACE_RTL_TASK_ID", None)
        self.logger.debug("ChipVerilog integration cleanup complete")
