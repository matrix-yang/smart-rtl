from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .agent import TopAgent
from .log_utils import get_logger, set_log_dir
from .rtl_editor import RTLEditor
from .rtl_generator import RTLGenerator
from .token_counter import TokenCounter, TokenCounterCached

logger = get_logger(__name__)

MODULE_RE = re.compile(r"(?m)^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b")
INSTANCE_RE = re.compile(
    r"(?m)^\s*([A-Za-z_][A-Za-z0-9_$]*)\s*(?:#\s*\([^;]*?\))?\s+"
    r"([A-Za-z_][A-Za-z0-9_$]*)\s*\("
)
RESULT_NAME_OVERRIDES = {"fpu_double": "fpu"}
KEYWORDS = {
    "module", "if", "else", "for", "while", "case", "assign", "always",
    "initial", "begin", "end", "function", "task", "generate",
}


@dataclass(frozen=True)
class ChipVerilogTask:
    task_id: str
    family: str
    target_top: str
    description_file: Path
    reference_dir: Path
    reference_file: Path
    dependency_files: tuple[Path, ...]
    include_files: tuple[Path, ...]


@dataclass
class EvaluationResult:
    passed: bool
    score: int
    status: str
    feedback: str
    report: dict
    elapsed_seconds: float


def _defined_modules(path: Path) -> set[str]:
    return set(MODULE_RE.findall(path.read_text(encoding="utf-8", errors="ignore")))


def _instantiated_modules(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {name for name, _ in INSTANCE_RE.findall(text) if name not in KEYWORDS}


def _reference_file(reference_dir: Path) -> Path:
    same_name = reference_dir / f"{reference_dir.name}.v"
    if same_name.is_file():
        return same_name
    candidates = sorted(reference_dir.glob("*.v")) + sorted(reference_dir.glob("*.sv"))
    for path in candidates:
        if "tb" not in path.stem.lower() and "test" not in path.stem.lower():
            return path
    raise FileNotFoundError(f"No reference RTL under {reference_dir}")


def _reference_top(reference: Path, reference_dir: Path) -> str:
    modules = _defined_modules(reference)
    if reference_dir.name in modules:
        return reference_dir.name
    testbenches = [
        path for path in reference_dir.glob("*.v")
        if path != reference and ("tb" in path.stem.lower() or "test" in path.stem.lower())
    ]
    instantiated = set().union(*(_instantiated_modules(path) for path in testbenches)) if testbenches else set()
    intersection = modules & instantiated
    if len(intersection) == 1:
        return next(iter(intersection))
    if len(modules) == 1:
        return next(iter(modules))
    raise ValueError(f"Cannot resolve DUT top in {reference}")


def discover_tasks(chipverilog_root: Path) -> list[ChipVerilogTask]:
    src_root = chipverilog_root / "Src"
    des_root = chipverilog_root / "Des"
    tasks: list[ChipVerilogTask] = []
    all_reference_files = [
        path for path in des_root.rglob("*.v")
        if "tb" not in path.stem.lower() and "test" not in path.stem.lower()
    ]
    module_index: dict[str, list[Path]] = {}
    for path in all_reference_files:
        for module in _defined_modules(path):
            module_index.setdefault(module, []).append(path)

    for description in sorted(src_root.glob("**/des/**/description.txt")):
        relative = description.relative_to(src_root)
        family_src = relative.parts[0]
        family = {
            "verilog_cordic_core": "cordic_core",
            "or1200_hp": "or1200",
        }.get(family_src, family_src)
        leaf = description.parent.name
        task_id = RESULT_NAME_OVERRIDES.get(leaf, leaf)
        if family == "mips_16":
            task_id = f"mips_{leaf}"
        reference_leaf = task_id[len("mips_"):] if family == "mips_16" else task_id
        reference_dir = des_root / family / reference_leaf
        reference = _reference_file(reference_dir)
        target_top = _reference_top(reference, reference_dir)

        dependencies: list[Path] = []
        seen_modules = {target_top}
        frontier = list(_instantiated_modules(reference))
        while frontier:
            module = frontier.pop()
            if module in seen_modules:
                continue
            seen_modules.add(module)
            choices = module_index.get(module, [])
            if not choices:
                continue
            same_family = [p for p in choices if p.is_relative_to(des_root / family)]
            dependency = (same_family or choices)[0]
            if dependency == reference or dependency in dependencies:
                continue
            dependencies.append(dependency)
            frontier.extend(_instantiated_modules(dependency) - seen_modules)

        includes = tuple(
            sorted(
                path for path in (des_root / family).glob("*.v")
                if path.name == "timescale.v" or "defs" in path.name.lower() or "defines" in path.name.lower()
            )
        )
        tasks.append(
            ChipVerilogTask(
                task_id=task_id,
                family=family,
                target_top=target_top,
                description_file=description,
                reference_dir=reference_dir,
                reference_file=reference,
                dependency_files=tuple(sorted(set(dependencies))),
                include_files=includes,
            )
        )
    return tasks


def _clip_public_text(text: str, budget: int, label: str) -> tuple[str, bool]:
    if len(text) <= budget:
        return text, False
    marker = f"\n/* ... {label} truncated to fit model context ... */\n"
    if budget <= len(marker) + 80:
        return text[: max(0, budget)], True
    head = (budget - len(marker)) * 2 // 3
    tail = budget - len(marker) - head
    return text[:head] + marker + text[-tail:], True


def render_public_spec(task: ChipVerilogTask, max_chars: int | None = None) -> str:
    description = task.description_file.read_text(encoding="utf-8")
    header = f"""ChipVerilog target contract:
- Generate exactly one target module named `{task.target_top}`.
- Do not redefine dependency modules.
- Match all described ports, widths, timing, macros, and submodule connections.
- The reference implementation is hidden. Golden feedback may report compiler,
  simulation, interface, or equivalence failures, but never exposes reference RTL.

"""
    if max_chars is None:
        dependency_sections = [
            f"\n<dependency_file name=\"{path.name}\">\n"
            f"{path.read_text(encoding='utf-8', errors='ignore')}\n"
            "</dependency_file>"
            for path in task.dependency_files
        ]
        include_sections = [
            f"\n<include_file name=\"{path.name}\">\n"
            f"{path.read_text(encoding='utf-8', errors='ignore')}\n"
            "</include_file>"
            for path in task.include_files
        ]
        return header + f"""<target_specification>
{description.strip()}
</target_specification>

<available_includes>
{''.join(include_sections) or '(none)'}
</available_includes>

<available_dependency_rtl>
{''.join(dependency_sections) or '(none; target is standalone)'}
</available_dependency_rtl>
"""

    if max_chars < 4000:
        raise ValueError("max_chars must be at least 4000")
    fixed_overhead = len(header) + 500
    content_budget = max_chars - fixed_overhead
    description_budget = min(len(description), max(1800, content_budget * 45 // 100))
    clipped_description, description_truncated = _clip_public_text(
        description.strip(), description_budget, "target specification"
    )
    remaining = content_budget - len(clipped_description)

    include_sections: list[str] = []
    include_budget = min(max(0, remaining // 4), 10000)
    for index, path in enumerate(task.include_files):
        slots = len(task.include_files) - index
        allowance = max(200, include_budget // max(1, slots))
        content, clipped = _clip_public_text(
            path.read_text(encoding="utf-8", errors="ignore"),
            allowance,
            f"include {path.name}",
        )
        include_sections.append(
            f"\n<include_file name=\"{path.name}\" truncated=\"{str(clipped).lower()}\">\n"
            f"{content}\n</include_file>"
        )
        include_budget -= min(include_budget, len(content))
    remaining -= sum(len(section) for section in include_sections)

    dependency_sections: list[str] = []
    dependency_budget = max(0, remaining - 400)
    for index, path in enumerate(task.dependency_files):
        slots = len(task.dependency_files) - index
        allowance = max(400, dependency_budget // max(1, slots))
        content, clipped = _clip_public_text(
            path.read_text(encoding="utf-8", errors="ignore"),
            allowance,
            f"dependency {path.name}",
        )
        dependency_sections.append(
            f"\n<dependency_file name=\"{path.name}\" truncated=\"{str(clipped).lower()}\">\n"
            f"{content}\n</dependency_file>"
        )
        dependency_budget -= min(dependency_budget, len(content))

    truncation_note = (
        "Public context was deterministically clipped for the 32768-token model. "
        "Target specification, include files, and every dependency retain their head/tail."
        if description_truncated or any('truncated=\"true\"' in section for section in include_sections + dependency_sections)
        else "No public context was clipped."
    )
    rendered = header + f"""<context_note>{truncation_note}</context_note>

<target_specification>
{clipped_description}
</target_specification>

<available_includes>
{''.join(include_sections) or '(none)'}
</available_includes>

<available_dependency_rtl>
{''.join(dependency_sections) or '(none; target is standalone)'}
</available_dependency_rtl>
"""
    if len(rendered) > max_chars:
        rendered = rendered[: max_chars - 80] + "\n/* final deterministic context cap */\n"
    return rendered


class ChipVerilogGoldenReviewer:
    SCORE = {
        "pass": 0,
        "function_fail": 100,
        "interface_fail": 200,
        "elaboration_fail": 300,
        "compile_fail": 400,
        "simulation_runtime_fail": 500,
        "equivalence_timeout": 600,
        "equivalence_error": 700,
        "reference_fail": 800,
    }

    def __init__(
        self,
        task: ChipVerilogTask,
        output_dir: Path,
        chipverilog_root: Path,
        model_name: str,
        telemetry_file: Path,
        timeout: int = 120,
        depth: int = 16,
    ) -> None:
        self.task = task
        self.output_dir = output_dir
        self.chipverilog_root = chipverilog_root
        self.model_name = model_name
        self.telemetry_file = telemetry_file
        self.timeout = timeout
        self.depth = depth
        self.evaluation_index = 0
        self.last_result: EvaluationResult | None = None

    def _record(self, payload: dict) -> None:
        self.telemetry_file.parent.mkdir(parents=True, exist_ok=True)
        with self.telemetry_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def evaluate(self) -> EvaluationResult:
        self.evaluation_index += 1
        rtl_path = self.output_dir / "rtl.sv"
        candidate_dir = self.chipverilog_root / "Result" / self.model_name / self.task.task_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_dir / f"{self.task.task_id}_t1.v"
        shutil.copy2(rtl_path, candidate_path)
        snapshot_dir = self.output_dir / "candidate_history"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rtl_path, snapshot_dir / f"candidate_{self.evaluation_index:03d}.sv")

        report_dir = self.output_dir / "golden_reports"
        command = [
            "python3", str(self.chipverilog_root / "tools" / "formal_equivalence.py"),
            "verify", str(candidate_dir),
            "--result-root", str(self.chipverilog_root / "Result"),
            "--report-dir", str(report_dir),
            "--depth", str(self.depth), "--timeout", str(self.timeout),
        ]
        start = time.monotonic()
        process = subprocess.run(command, text=True, capture_output=True)
        elapsed = time.monotonic() - start
        report_path = report_dir / self.model_name / f"{self.task.task_id}.json"
        if process.returncode or not report_path.is_file():
            report = {"status": "evaluator_error", "stdout": process.stdout, "stderr": process.stderr}
            result = EvaluationResult(False, 900, "evaluator_error", json.dumps(report), report, elapsed)
        else:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            row = report["results"][0]
            status = row["status"]
            feedback_payload = {
                "status": status,
                "flow": row.get("flow"),
                "reason": row.get("reason"),
                "interface_status": row.get("interface_status"),
                "formal_status": row.get("formal_status"),
                "counterexample_summary": row.get("counterexample_summary"),
                "compile_gate_error_excerpt": row.get("compile_gate_error_excerpt"),
                "compile_gate_warning_excerpt": row.get("compile_gate_warning_excerpt"),
            }
            result = EvaluationResult(
                passed=status == "pass",
                score=self.SCORE.get(status, 850),
                status=status,
                feedback=json.dumps(feedback_payload, ensure_ascii=False, indent=2),
                report=report,
                elapsed_seconds=elapsed,
            )
        self.last_result = result
        self._record({
            "event": "tool_call", "timestamp": time.time(), "tool": "chipverilog_golden_evaluator",
            "task_id": self.task.task_id, "evaluation_index": self.evaluation_index,
            "command": command, "return_code": process.returncode, "latency_seconds": elapsed,
            "status": result.status, "score": result.score,
        })
        return result

    def review(self) -> tuple[bool, int, str]:
        result = self.evaluate()
        return result.passed, result.score, result.feedback

    def review_for_edit(self) -> dict:
        """Evaluate an edit atomically with the benchmark's full build context."""
        result = self.evaluate()
        syntax_failed = result.status in {
            "compile_fail", "elaboration_fail", "interface_fail", "evaluator_error"
        }
        return {
            "is_syntax_pass": not syntax_failed,
            "is_sim_pass": result.passed,
            "error_msg": "" if result.passed else result.feedback,
            "sim_mismatch_cnt": result.score,
        }


class ChipVerilogMageAgent(TopAgent):
    """MAGE candidate generation/edit loop grounded by ChipVerilog's golden evaluator.

    TBGenerator and SimJudge are intentionally bypassed because the evaluator is
    authoritative. RTLGenerator and RTLEditor remain the LLM agents; compilation,
    simulation and equivalence are deterministic tool calls.
    """

    def __init__(self, llm, max_candidates: int = 5, editor_steps: int = 4):
        super().__init__(llm)
        self.rtl_max_candidates = max_candidates
        self.rtl_selected_candidates = 2
        self.editor_steps = editor_steps

    def run_chipverilog(
        self,
        task: ChipVerilogTask,
        spec: str,
        chipverilog_root: Path,
        model_name: str,
        original_spec_chars: int | None = None,
    ) -> tuple[bool, str]:
        output_dir = Path(self.output_path) / f"chipverilog_{task.task_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir_per_run = str(output_dir)
        log_dir = Path(self.log_path) / f"chipverilog_{task.task_id}"
        set_log_dir(str(log_dir))
        telemetry_file = output_dir / "telemetry.jsonl"
        os.environ["MAGE_TELEMETRY_FILE"] = str(telemetry_file)
        self.token_counter.reset()
        self._spec_chars = len(spec)
        self._original_spec_chars = original_spec_chars or len(spec)
        reviewer = ChipVerilogGoldenReviewer(
            task, output_dir, chipverilog_root, model_name, telemetry_file
        )
        generator = RTLGenerator(self.token_counter)
        # MAGE's legacy single-file syntax retry cannot see benchmark macros and
        # dependencies. Generate exactly five proposals, then let the golden
        # evaluator judge them in the complete build context.
        generator.max_trials = 1
        self.token_counter.set_cur_tag("rtl_generator")
        first = generator.chat(spec, "", "", str(output_dir / "rtl.sv"), enable_cache=True)
        candidates = [first]
        if self.rtl_max_candidates > 1:
            self.token_counter.set_cur_tag("rtl_candidate_sampling")
            candidates += generator.gen_candidates(
                spec, "", "", str(output_dir / "rtl.sv"),
                candidates_num=self.rtl_max_candidates - 1, enable_cache=True,
            )

        ranked: list[tuple[int, str, str]] = []
        final_rtl = ""
        for index, (syntax_ok, rtl) in enumerate(candidates, start=1):
            if not rtl.strip():
                continue
            (output_dir / "rtl.sv").write_text(rtl, encoding="utf-8")
            evaluation = reviewer.evaluate()
            self._record_selection(telemetry_file, index, evaluation, syntax_ok)
            if evaluation.passed:
                final_rtl = rtl
                return self._finish(task, output_dir, final_rtl, reviewer, telemetry_file)
            ranked.append((evaluation.score, rtl, evaluation.feedback))

        ranked.sort(key=lambda item: item[0])
        for rank, (score, rtl, feedback) in enumerate(ranked[: self.rtl_selected_candidates], start=1):
            (output_dir / "rtl.sv").write_text(rtl, encoding="utf-8")
            editor = RTLEditor(self.token_counter, reviewer)
            editor.max_trials = self.editor_steps
            self.token_counter.set_cur_tag(f"rtl_editor_rank_{rank}")
            passed, edited = editor.chat(
                spec=spec,
                output_dir_per_run=str(output_dir),
                sim_failed_log=feedback,
                sim_mismatch_cnt=score,
            )
            final_rtl = edited
            if passed:
                return self._finish(task, output_dir, final_rtl, reviewer, telemetry_file)

        if ranked and not final_rtl:
            final_rtl = ranked[0][1]
            (output_dir / "rtl.sv").write_text(final_rtl, encoding="utf-8")
        return self._finish(task, output_dir, final_rtl, reviewer, telemetry_file)

    @staticmethod
    def _record_selection(
        path: Path, index: int, result: EvaluationResult, legacy_syntax_ok: bool
    ) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "event": "candidate_evaluation", "timestamp": time.time(),
                "candidate_index": index, "status": result.status,
                "score": result.score, "latency_seconds": result.elapsed_seconds,
                "legacy_single_file_syntax_ok": legacy_syntax_ok,
            }) + "\n")

    def _finish(
        self,
        task: ChipVerilogTask,
        output_dir: Path,
        rtl: str,
        reviewer: ChipVerilogGoldenReviewer,
        telemetry_file: Path,
    ) -> tuple[bool, str]:
        if rtl:
            (output_dir / "rtl.sv").write_text(rtl, encoding="utf-8")
        final = reviewer.evaluate() if rtl else EvaluationResult(False, 999, "empty", "empty RTL", {}, 0)
        tokens = self.token_counter.get_sum_count()
        summary = {
            "task_id": task.task_id,
            "family": task.family,
            "target_top": task.target_top,
            "passed": final.passed,
            "status": final.status,
            "input_tokens": tokens.in_token_cnt,
            "output_tokens": tokens.out_token_cnt,
            "llm_calls_by_role": {key: len(value) for key, value in self.token_counter.token_cnts.items()},
            "golden_tool_calls": reviewer.evaluation_index,
            "dependency_files": [str(path) for path in task.dependency_files],
            "include_files": [str(path) for path in task.include_files],
            "public_spec_chars": self._spec_chars,
            "public_spec_original_chars": self._original_spec_chars,
            "public_spec_truncated": self._spec_chars < self._original_spec_chars,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        with telemetry_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "task_complete", **summary}) + "\n")
        return final.passed, rtl
