#!/usr/bin/env python3
"""Unified public entry point for Smart-RTL benchmark adapters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
ACE_RTL_SCRIPTS = REPO_ROOT / "skills" / "ace-rtl" / "scripts"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-rtl",
        description="Run one fail-closed Smart-RTL adaptive search",
    )
    commands = parser.add_subparsers(dest="benchmark", metavar="{cvdp,chipverilog}")
    commands.add_parser("cvdp", help="run exactly one CVDP datapoint")
    commands.add_parser("chipverilog", help="run or list ChipVerilog public tasks")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    benchmark, benchmark_args = args[0], args[1:]
    if benchmark not in {"cvdp", "chipverilog"}:
        parser.error(f"unknown benchmark: {benchmark}")

    sys.path.insert(0, str(ACE_RTL_SCRIPTS))
    if benchmark == "cvdp":
        from ace_cvdp_native.rtl_search.adaptive_live import main as benchmark_main
    else:
        from ace_cvdp_native.rtl_search.adaptive_chipverilog_live import (
            main as benchmark_main,
        )
    return benchmark_main(benchmark_args)


if __name__ == "__main__":
    raise SystemExit(main())
