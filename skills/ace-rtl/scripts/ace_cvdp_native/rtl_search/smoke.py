"""Run the complete role-aware mechanism loop without an LLM or benchmark API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .contracts import TaskContract
from .engine import RTLSearchEngine
from .instructions import GeneralACEInstructions
from .scripted import make_scripted_roles


def run_smoke() -> dict[str, object]:
    task = TaskContract.from_parts(
        task_id="mechanism_smoke_task",
        specification=(
            "Implement the declared synthesizable synchronous RTL target. Preserve the "
            "public module interface and active-low reset behavior; do not modify verification assets."
        ),
        target_files=("rtl/top.sv",),
        immutable_context={"docs/public_spec.md": "Public timing and interface notes."},
    )
    engine = RTLSearchEngine(
        roles=make_scripted_roles(),
        instructions=GeneralACEInstructions(),
        branch_width=2,
        max_steps=8,
        max_plans=3,
    )
    result = engine.run(task)
    payload = result.to_dict()
    steps = payload["steps"]
    backtracks = payload["backtracks"]
    structural_step = next(
        (step for step in steps if step["decision_reason"] == "structural_regression"),
        None,
    )
    post_backtrack_step = None
    if structural_step is not None:
        structural_index = int(structural_step["step"])
        post_backtrack_step = next(
            (step for step in steps if int(step["step"]) == structural_index + 1),
            None,
        )
    required_actions = {"refine", "backtrack", "expand", "stop"}
    action_counts = payload["action_counts"]
    checks = {
        "status_passed": payload["status"] == "passed",
        "two_plans_created": int(payload["plans_created"]) >= 2,
        "all_controller_actions_exercised": required_actions <= set(action_counts),
        "structural_backtrack_recorded": structural_step is not None and bool(backtracks),
        "post_backtrack_child_uses_destination": (
            structural_step is not None
            and post_backtrack_step is not None
            and post_backtrack_step["parent_id"] == structural_step["destination_id"]
            and post_backtrack_step["producing_action"] == "expand"
        ),
        "contract_rejection_exercised": int(payload["contract_rejected_candidates"]) >= 1,
        "zero_llm_api_calls": int(payload["llm_api_calls"]) == 0,
    }
    payload["acceptance"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }
    payload["experiment_tier"] = "auxiliary/dev mechanism smoke"
    payload["claim_boundary"] = (
        "Control-flow executability only; no benchmark effectiveness, instruction quality, or generalization claim."
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_smoke()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
