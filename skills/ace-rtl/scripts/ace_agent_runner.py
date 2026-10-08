#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACE-RTL wrapper for the vendored reusable RTL agent package."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional


def workspace_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def pop_workspace_arg(argv: List[str]) -> tuple[List[str], Optional[Path]]:
    cleaned: List[str] = [argv[0]]
    workspace: Optional[Path] = None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--workspace":
            if i + 1 >= len(argv):
                raise SystemExit("--workspace requires a value")
            workspace = Path(argv[i + 1]).resolve()
            i += 2
            continue
        if arg.startswith("--workspace="):
            workspace = Path(arg.split("=", 1)[1]).resolve()
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned, workspace


def has_option(argv: List[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv[1:])


def is_non_cvdp_mode(argv: List[str]) -> bool:
    return (
        has_option(argv, "--verilogeval-problem")
        or has_option(argv, "--chipbench-problem")
        or has_option(argv, "--chipverilog-problem")
    )


def default_cvdp_repo(workspace: Path) -> Path:
    env_path = os.environ.get("CVDP_REPO_ROOT")
    if env_path:
        candidate = Path(env_path).resolve()
        try:
            candidate.relative_to(workspace)
            return candidate
        except ValueError:
            pass
    return (workspace / "external" / "cvdp_benchmark").resolve()


def main() -> None:
    argv, explicit_workspace = pop_workspace_arg(sys.argv)
    workspace = explicit_workspace or Path(os.environ.get("ACE_RTL_WORKSPACE", workspace_from_script())).resolve()

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("ACE_RTL_WORKSPACE", str(workspace))

    if not has_option(argv, "--output-dir"):
        argv.extend(["--output-dir", str(workspace / "outputs" / "ace_rtl_agent")])

    if not is_non_cvdp_mode(argv) and not has_option(argv, "--cvdp-benchmark-path"):
        argv.extend(["--cvdp-benchmark-path", str(default_cvdp_repo(workspace))])

    sys.argv = argv

    from ace_rtl_agent.main import main as agent_main

    agent_main()


if __name__ == "__main__":
    main()
