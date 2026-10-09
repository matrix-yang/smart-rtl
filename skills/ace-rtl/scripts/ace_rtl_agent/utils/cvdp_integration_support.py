# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants and helpers for CVDP integration."""

from typing import Dict

MAX_REPORT_BYTES = 2 * 1024 * 1024          # Never read/keep more than 2MB of a report
REPORT_HEAD_BYTES = 1 * 1024 * 1024          # Keep the first 1MB (setup + first failures)
REPORT_TAIL_BYTES = 256 * 1024               # Keep the last 256KB (final error/traceback)
MAX_NATIVE_RUNDIR_LOG_BYTES = 96 * 1024
NATIVE_RUNDIR_LOG_NAMES = (
    "lint_results.log",
    "synth.log",
    "sim.log",
    "coverage.log",
)


def _summarize_runaway_report(head_text: str, tail_text: str, original_size: int) -> str:
    """
    Detect a hung/spinning testbench and produce a concise, actionable report.

    When a report is truncated because it grew enormous, the usual cause is a
    deadlocked DUT that never makes progress, so the testbench repeats a single
    "waiting" message endlessly. We surface that diagnosis explicitly so the
    debugging LLM fixes the deadlock instead of drowning in millions of lines.
    """
    # Find the most-repeated non-trivial line in the head sample.
    counts: Dict[str, int] = {}
    for line in head_text.splitlines():
        stripped = line.strip()
        if len(stripped) < 8:
            continue
        counts[stripped] = counts.get(stripped, 0) + 1

    hang_note = ""
    if counts:
        top_line, top_count = max(counts.items(), key=lambda kv: kv[1])
        if top_count >= 100:
            hang_note = (
                "\n\n================ AUTOMATED HANG DIAGNOSIS ================\n"
                f"The testbench produced a runaway log ({original_size / (1024*1024):.0f} MB). "
                f"The line below repeated at least {top_count} times, which means the "
                "simulation DEADLOCKED / never made forward progress:\n"
                f"    \"{top_line}\"\n"
                "ROOT CAUSE: the DUT almost certainly never asserts the handshake/valid/"
                "ready signal the testbench is waiting on (e.g. m_axis_tvalid, *_ready, "
                "*_done), or a state machine is stuck. FIX the RTL so that signal is "
                "driven correctly and the protocol completes; do not change the testbench.\n"
                "==========================================================\n"
            )

    return (
        head_text
        + hang_note
        + "\n... [REPORT TRUNCATED: exceeded size cap to protect disk/context] ...\n"
        + tail_text
    )
