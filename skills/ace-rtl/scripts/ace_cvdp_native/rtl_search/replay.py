"""Offline replay of persisted ACE-RTL CVDP traces.

Example:

    python -m ace_cvdp_native.rtl_search.replay \
      --result-json /path/to/datapoint/result.json \
      --attempt 3 \
      --expect-backtrack 6:5 \
      --expect-backtrack 9:7 \
      --expect-stop 10 \
      --output replay.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .node import ActionKind, EDAEvidence
from .policy import TreePolicy
from .scoring import normalize_failure_signature, parse_eda_stage
from .tree import SearchTree


@dataclass(frozen=True, slots=True)
class ReplayStep:
    iteration: int
    node_id: str
    parent_id: str
    candidate_hash: str
    stage: str
    action: str
    reason: str
    destination_id: Optional[str]
    source_ref: str


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _select_attempt(payload: Dict[str, Any], attempt_index: int) -> Dict[str, Any]:
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("result JSON has no attempts list")
    for attempt in attempts:
        if isinstance(attempt, dict) and attempt.get("attempt") == attempt_index:
            history = attempt.get("history")
            if not isinstance(history, list) or not history:
                raise ValueError(f"attempt {attempt_index} has no replayable history")
            return attempt
    raise ValueError(f"attempt {attempt_index} not found")


def _load_postprocess(datapoint_root: Path, attempt_index: int, iteration: int) -> Dict[str, Any]:
    path = datapoint_root / f"attempt_{attempt_index}" / f"iter_{iteration:02d}" / "postprocess_result.json"
    return _read_json(path) if path.exists() else {}


def evidence_from_history_item(
    item: Dict[str, Any],
    postprocess: Dict[str, Any],
) -> EDAEvidence:
    primary_failure = item.get("primary_failure") or {}
    raw_stage = str(primary_failure.get("stage") or "")
    stage = parse_eda_stage(raw_stage)
    result = item.get("result") or {}
    tests = result.get("tests") or []
    recorded_tests = [test for test in tests if isinstance(test, dict)]
    passed_subtests = sum(1 for test in recorded_tests if test.get("result") == 0)
    preflight_issues = postprocess.get("preflight_issues") or []
    complete_bundle = not bool(preflight_issues)
    signature = normalize_failure_signature(
        str(primary_failure.get("signature") or primary_failure.get("message") or raw_stage)
    )
    return EDAEvidence(
        stage=stage,
        original_stage=raw_stage,
        failure_class=raw_stage.lower(),
        failure_signature=signature,
        complete_bundle=complete_bundle,
        passed_subtests=passed_subtests,
        total_subtests=len(recorded_tests),
    )


def replay_attempt(result_json: Path, attempt_index: int) -> Dict[str, Any]:
    payload = _read_json(result_json)
    attempt = _select_attempt(payload, attempt_index)
    task_id = str(payload.get("id") or result_json.parent.name)
    history = sorted(attempt["history"], key=lambda item: int(item["iteration"]))

    tree = SearchTree()
    root = tree.add_root(task_id)
    plan = tree.add_plan(
        root.node_id,
        f"historical ACE attempt {attempt_index}",
        plan_id=f"historical_attempt_{attempt_index}",
        node_id=f"plan_attempt_{attempt_index}",
    )
    policy = TreePolicy()
    parent_id = plan.node_id
    steps: List[ReplayStep] = []

    for offset, item in enumerate(history):
        iteration = int(item["iteration"])
        postprocess = _load_postprocess(result_json.parent, attempt_index, iteration)
        evidence = evidence_from_history_item(item, postprocess)
        candidate_hash = str(
            postprocess.get("candidate_bundle_hash")
            or item.get("candidate_hash")
            or ""
        )
        if not candidate_hash:
            raise ValueError(f"iteration {iteration} has no candidate hash")
        node_id = f"iter_{iteration:02d}"
        iteration_dir = result_json.parent / f"attempt_{attempt_index}" / node_id
        node = tree.add_candidate(
            parent_id,
            candidate_hash,
            evidence,
            action_from_parent=ActionKind.EXPAND if offset == 0 else ActionKind.REFINE,
            node_id=node_id,
            source_iteration=iteration,
            source_ref=str(iteration_dir),
        )
        decision = policy.decide(tree, node.node_id)
        if decision.action is ActionKind.BACKTRACK:
            if decision.destination_id is None:
                raise RuntimeError("BACKTRACK decision omitted destination")
            tree.backtrack(node.node_id, decision.destination_id, decision.reason)
        steps.append(
            ReplayStep(
                iteration=iteration,
                node_id=node.node_id,
                parent_id=parent_id,
                candidate_hash=candidate_hash,
                stage=evidence.original_stage,
                action=decision.action.value,
                reason=decision.reason,
                destination_id=decision.destination_id,
                source_ref=str(iteration_dir),
            )
        )
        parent_id = node.node_id

    duplicate_steps = [step for step in steps if step.reason == "duplicate_candidate_state"]
    structural_steps = [step for step in steps if step.reason == "structural_regression"]
    stop_steps = [step for step in steps if step.action == ActionKind.STOP.value]
    backtrack_steps = [step for step in steps if step.action == ActionKind.BACKTRACK.value]
    return {
        "schema_version": 1,
        "experiment_tier": "auxiliary/dev",
        "replay_mode": "observational_policy_audit",
        "task_id": task_id,
        "result_json": str(result_json.resolve()),
        "attempt": attempt_index,
        "policy": "deterministic_rtl_tree_v0",
        "budget_used_by_policy": False,
        "llm_calls": 0,
        "metrics": {
            "iterations_loaded": len(steps),
            "candidate_nodes": len(steps),
            "backtrack_decisions": len(backtrack_steps),
            "duplicate_candidate_states": len(duplicate_steps),
            "structural_regressions": len(structural_steps),
            "pass_stop_detections": sum(1 for step in stop_steps if step.reason == "fail_closed_pass"),
        },
        "steps": [asdict(step) for step in steps],
    }


def _parse_expected_backtrack(raw: str) -> Tuple[int, int]:
    try:
        source, destination = raw.split(":", 1)
        return int(source), int(destination)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected ITERATION:DESTINATION_ITERATION") from exc


def attach_expectations(
    report: Dict[str, Any],
    expected_backtracks: Sequence[Tuple[int, int]],
    expected_stops: Sequence[int],
) -> bool:
    expectations_supplied = bool(expected_backtracks or expected_stops)
    actual_backtracks = {
        (int(step["iteration"]), int(str(step["destination_id"]).split("_")[-1]))
        for step in report["steps"]
        if step["action"] == ActionKind.BACKTRACK.value
        and str(step.get("destination_id", "")).startswith("iter_")
    }
    expected_backtrack_set = set(expected_backtracks)
    actual_stops = {
        int(step["iteration"])
        for step in report["steps"]
        if step["action"] == ActionKind.STOP.value
    }
    expected_stop_set = set(expected_stops)
    true_positive = len(actual_backtracks & expected_backtrack_set)
    recall = (
        true_positive / len(expected_backtrack_set)
        if expected_backtrack_set
        else 1.0
    )
    unexpected = actual_backtracks - expected_backtrack_set if expectations_supplied else set()
    missing = expected_backtrack_set - actual_backtracks
    missing_stops = expected_stop_set - actual_stops
    unexpected_stops = (
        actual_stops - expected_stop_set if expectations_supplied else set()
    )
    passed = not missing and not unexpected and not missing_stops and not unexpected_stops
    report["expectations"] = {
        "expectations_supplied": expectations_supplied,
        "expected_backtracks": sorted([list(item) for item in expected_backtrack_set]),
        "actual_backtracks": sorted([list(item) for item in actual_backtracks]),
        "expected_backtrack_recall": recall,
        "unexpected_backtracks": sorted([list(item) for item in unexpected]),
        "missing_backtracks": sorted([list(item) for item in missing]),
        "expected_stops": sorted(expected_stop_set),
        "actual_stops": sorted(actual_stops),
        "missing_stops": sorted(missing_stops),
        "unexpected_stops": sorted(unexpected_stops),
        "passed": passed,
    }
    return passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument(
        "--expect-backtrack",
        type=_parse_expected_backtrack,
        action="append",
        default=[],
        metavar="ITER:DEST",
    )
    parser.add_argument("--expect-stop", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = replay_attempt(args.result_json, args.attempt)
    expectations_passed = attach_expectations(
        report,
        args.expect_backtrack,
        args.expect_stop,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if expectations_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
