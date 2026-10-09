# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACE-RTL native CVDP runner internals."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import logging
import multiprocessing
import os
import queue
import re
import shutil
import shlex
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


MAX_REPORT_BYTES = 2 * 1024 * 1024
DEFAULT_EVALUATOR_TIMEOUT_SECONDS = 300
MAX_NATIVE_RUNDIR_LOG_BYTES = 96 * 1024
MAX_SERVICE_LOG_BYTES = 384 * 1024
NATIVE_RUNDIR_REPORT_RESERVE_BYTES = 384 * 1024
NATIVE_RUNDIR_LOG_NAMES = (
    "lint_results.log",
    "synth.log",
    "sim.log",
    "xrun.log",
    "irun.log",
    "build.log",
    "simulation.log",
    "coverage.log",
    "coverage",
    "imc.log",
    "mdv.log",
)
CADENCE_DEFAULT_TIMESCALE_ARGS = "-timescale 1ns/1ps"
CADENCE_TIMESCALE_RE = re.compile(r"(?:^|\s)-(?:override_)?timescale(?:\s|=|$)", re.IGNORECASE)
CADENCE_COMMAND_RE = re.compile(
    r"(?<![\w./-])((?:/[^\s'\"`;]+/)?(?:xrun|irun))(?![\w./-])",
    re.IGNORECASE,
)
CADENCE_LICENSE_ENV_VARS = (
    "CDS_LIC_FILE",
    "LM_LICENSE_FILE",
    "SNPSLMD_LICENSE_FILE",
    "MGLS_LICENSE_FILE",
)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


RTL_IMPLEMENTATION_GUIDANCE = """\
This is a CVDP RTL implementation/modification task. Return the complete target RTL file(s).

Preserve module name, parameter list, port names, directions, widths, reset polarity,
clocking, and existing latency unless the public prompt explicitly changes it.

Implement only behavior requested by the public prompt and context. Do not use
hardcoded test-vector outputs or hidden reference behavior. Drive all outputs
deterministically on reset and every branch.

If the public prompt supplies an existing RTL body and asks to identify/fix a
bug, treat the task as a bug fix: preserve the supplied architecture and make the
smallest complete-code correction needed to satisfy the public specification and
observed/expected behavior. Do not replace a mostly complete supplied design with
a new clean-room implementation unless the prompt explicitly asks for a rewrite
or the supplied code is structurally unusable. In a bug-fix task, observed output
tables are public symptoms for debugging timing/control/datapath issues, not
constants to hardcode.

The generated RTL must compile with Icarus Verilog using `iverilog -g2012`.
It must also be clean under strict Verilator lint where the harness runs lint:
avoid unused helper signals/bits, implicit width expansion/truncation, blocking
assignments in clocked sequential logic, mixed sync/async driving, and stale
state across repeated `start`/`go` transactions. Use explicit casts or scoped
helper logic instead of relying on simulator truncation.
For Icarus compatibility, do not unpack or pack byte lanes with packed-vector
constant or indexed part-selects inside `always_*` processes. Use continuous
`assign` statements, wires, or helper functions before procedural use.
For multi-cycle designs, capture operation-select sideband controls with the
accepted input data and use the delayed copy at the output stage; do not use
live input controls to choose behavior for delayed results.
For fixed-latency datapaths, align the result register and valid flag to the
same public output cycle. If an output is still the reset/default value when the
valid flag reaches the checked cycle, inspect off-by-one valid-pipe indexing and
whether the result register is updated one clock too late.
Do not clear a registered result output merely because the current cycle is not
valid when a separate delayed valid/done flag may expose that cleared register on
the next checked cycle. Hold the last computed result or update/clear it in the
same clocked condition that controls the public valid/done flag.
For arithmetic specified by public constants or polynomial coefficients, derive
terms from the bits explicitly set in the public coefficient and do not invent
extra terms or copy private reference formulas. For finite-field or polynomial
constant multipliers, include the shifted/multiplied term for each set
coefficient bit and omit every unset-bit term.

Return raw source only: no Markdown fences, no prose, no JSON. If multiple
target files are listed, emit one complete section per file headed exactly by
`// TARGET_FILE: <path>`. Keep helper functions inside the module and declare
them before procedural blocks that use them. Do not emit top-level functions.
Declare every temporary signal, payload, byte alias, loop variable, and helper
variable at module scope; do not declare variables inside `always_comb`,
`always_ff`, `case`, `if`, `for`, or nested `begin` blocks.
"""


OPTIMIZATION_GUIDANCE = """\
This is a CVDP RTL lint/area/power-performance optimization task. Return the
complete target RTL file only.

Preserve module name, parameters, ports, reset polarity, clocking, checked output
timing, and externally visible behavior unless the public prompt explicitly
allows a latency or protocol change. First keep the sanity/functional service
passing, then reduce only the requested synthesis metric.

For registered protocols, preserve the cycle relationship among input capture,
internal state updates, result outputs, and `done`/valid flags. If the original
design registers `done` or the result, the optimized design must keep equivalent
registered timing and must clear/reinitialize transaction state on reset and on
each new public start/valid command. Do not convert a registered completion flag
to purely combinational logic unless the prompt explicitly requests that timing
change.

Keep the output Verilator-lint-clean: avoid unused helper signals/bits, implicit
width expansion/truncation, blocking assignments inside clocked sequential
logic, and overlapping state encodings/case arms.
Keep it Icarus-compatible as well: do not unpack or pack byte lanes with
packed-vector constant or indexed part-selects inside `always_*` blocks; use
continuous `assign` extraction/packing or helper functions before procedural
logic.
If the optimized RTL has multi-cycle behavior, pipeline operation-select
sideband controls with the data and use the delayed controls for result
selection, keying, and valid/done output behavior.
For fixed-latency outputs, keep result storage and the public valid/done flag on
the same checked cycle; avoid optimizations that leave the result at its default
value for the first valid cycle.
Do not clear a registered result output in an idle/default branch if a separately
pipelined valid/done flag can assert on the next cycle and expose the cleared
value. Tie result update/clear timing to the same stage as the public valid/done
signal, or hold the previous result until the transaction is no longer visible.

Return raw synthesizable SystemVerilog only: no Markdown fences, no prose, no
testbench code, no assertions, no delays, and no file or shell operations.
"""


COVERAGE_GUIDANCE = """\
This is a CVDP coverage/stimulus-generation task.

Return the complete generated `verif/` SystemVerilog testbench file. The clean
DUT must run finite legal stimulus and call `$finish` cleanly. Do not modify
immutable RTL.

Do not add `$error`, `$fatal`, `$stop`, reference-model checkers, fake coverage
files, shell/system calls, file writes, force/release, bind, or hierarchical
assignments to DUT internals.

Do not call simulator coverage system tasks. The CVDP harness owns coverage
collection and reporting.

Increase coverage by directed legal public input stimulus: reset, idle cycles,
small enumerations, boundary values, alternating patterns, valid/invalid
handshakes, back-to-back transactions, FSM transitions, and legal flush paths.
"""


CHECKER_GUIDANCE = """\
This is a CVDP checker/testbench-generation task.

Return the complete generated `verif/` SystemVerilog checker testbench file.
The clean DUT must finish without errors, while real mismatches against the
public specification must make injected buggy DUT variants fail simulation.

Use finite stimulus and bounded waits. Preserve the DUT instantiation and
harness-visible module/file role. Do not modify immutable RTL, do not compare a
DUT output to itself as the expected value, and do not suppress real checker
failures.
"""


VERIFICATION_GUIDANCE = """\
This is a CVDP verification/testbench-generation task.

Return the complete generated `verif/` SystemVerilog testbench file. Preserve
the requested testbench module name, DUT instance name, public DUT ports, and
harness-visible file role.

Use finite deterministic stimulus, bounded waits/timeouts, and an explicit
`$finish`. Drive legal public inputs only. Do not modify immutable RTL, do not
use shell/file operations, and do not emit Markdown or prose.
"""


ASSERTION_GUIDANCE = """\
This is a CVDP assertion-insertion task.

Return the complete target SystemVerilog `rtl/` or `verif/` file requested by
the datapoint. Preserve original module names, ports, parameters, assignments,
latency, reset behavior, and outputs. Assertion code must observe behavior only
and must not replace or drive the design.

Prefer simple simulator-compatible procedural immediate assertions gated by
reset/valid state. Do not emit Markdown, prose, fake pass logic, shell/file
operations, or unrelated testbench code.
"""


GENERIC_CVDP_GUIDANCE = """\
This is a CVDP code-generation task. Return the complete generated target file(s)
requested by the public prompt and context.

Preserve public interfaces, module/file roles, reset/clocking behavior, and
harness-visible timing unless the public prompt explicitly changes them. Do not
hardcode datapoint-specific expected outputs, hidden reference behavior, fake
coverage reports, shell/file operations, or private-test workarounds.

Generated RTL must be strict-lint-friendly when lint is part of the harness:
avoid unused helper signals/bits, implicit width warnings, and clocked blocking
assignments unless the public prompt explicitly requires a nonstandard style.
Generated RTL must also avoid Icarus-sensitive byte unpacking/packing with
packed-vector constant or indexed part-selects inside `always_*` blocks; extract
subfields with continuous `assign` statements, wires, or helper logic before
procedural use. For multi-cycle behavior, pipeline sideband controls together
with data and use delayed controls for result selection.
For fixed-latency datapaths, align result registers with their valid/done flags;
an all-default output at the expected valid cycle usually indicates off-by-one
valid-pipe gating or a result register updated one clock too late.
If the result output is registered, do not clear it in a default/idle branch that
can run one cycle before the delayed valid/done flag is checked. Either hold the
computed result or update/clear the result under the same pipeline stage that
drives the public valid/done flag.

Return raw SystemVerilog/Verilog source only: no Markdown fences, no prose, no
JSON. If multiple target files are listed, emit one complete section per file
headed exactly by `// TARGET_FILE: <path>`.
"""


ASSERTION_TASK_MARKERS = (
    "assertion-insertion",
    "assertion insertion",
    "create systemverilog assertions",
    "generate systemverilog assertions",
    "add systemverilog assertions",
    "insert systemverilog assertions",
    "add assertions",
    "insert assertions",
    "assertion requirements",
    "assertion properties",
    "assert properties",
    "assert property",
    "sva assertion",
    "sva assertions",
)

CHECKER_TASK_MARKERS = (
    "checker generation",
    "generate a checker",
    "generate checker",
    "testbench checker",
    "checker testbench",
    "checker functionality",
    "integrate a checker",
    "add a checker",
    "mutation",
    "mutant",
    "injected bug",
    "bug_",
    "num_bugs",
    "test_errors",
)

COVERAGE_TASK_MARKERS = (
    "coverage",
    "coverage.cmd",
    "test_coverage",
    "test_report",
    "covergroup",
    "cover property",
)

STIMULUS_TASK_MARKERS = (
    "stimulus generation",
    "generate stimulus",
    "directed stimulus",
    "test plan to testbench",
)

OPTIMIZATION_TASK_MARKERS = (
    "optimization",
    "optimize",
    "lint code review",
    "area optimization",
    "latency optimization",
    "power-performance",
    "power performance",
    "number of cells",
    "number of wires",
    "percent_cells",
    "percent_wires",
    "yosys",
)


def marker_in_text(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def classify_task(target_file: str, prompt: str) -> str:
    text = f"{target_file}\n{prompt}".lower()
    target = target_file.lower()
    is_rtl_target = target.startswith("rtl/")
    is_verif_target = target.startswith("verif/")

    # Ordinary RTL specs often say that a signal is "asserted". That should
    # not change an implementation task into an assertion-insertion task.
    if marker_in_text(text, ASSERTION_TASK_MARKERS):
        return "assertion"

    if marker_in_text(text, OPTIMIZATION_TASK_MARKERS):
        return "optimization"

    if is_rtl_target:
        return "rtl"

    # Checker prompts often also request stimulus and may mention coverage in
    # harness text. Mutation/checker intent must win over generic coverage.
    if marker_in_text(text, CHECKER_TASK_MARKERS):
        return "checker"

    if marker_in_text(text, COVERAGE_TASK_MARKERS) or (
        is_verif_target and marker_in_text(text, STIMULUS_TASK_MARKERS)
    ):
        return "coverage"

    if is_verif_target:
        return "verification"

    return "generic"


def guidance_for_task(task_kind: str) -> str:
    if task_kind == "coverage":
        return COVERAGE_GUIDANCE
    if task_kind == "checker":
        return CHECKER_GUIDANCE
    if task_kind == "verification":
        return VERIFICATION_GUIDANCE
    if task_kind == "assertion":
        return ASSERTION_GUIDANCE
    if task_kind == "optimization":
        return OPTIMIZATION_GUIDANCE
    if task_kind == "rtl":
        return RTL_IMPLEMENTATION_GUIDANCE
    return GENERIC_CVDP_GUIDANCE


def response_label_for_task(task_kind: str, target_file: str, files: List[str]) -> str:
    if len(files) > 1:
        return "target SystemVerilog/Verilog files"
    if task_kind == "coverage":
        return "coverage/stimulus SystemVerilog testbench file"
    if task_kind == "checker":
        return "checker SystemVerilog testbench file"
    if task_kind == "assertion":
        return "SystemVerilog file with requested assertions"
    if target_file.startswith("verif/"):
        return "SystemVerilog verification file"
    return "target SystemVerilog/Verilog file"


def system_prompt_for_task(task_kind: str, multi_file: bool = False) -> str:
    if task_kind in {"coverage", "checker", "verification"}:
        role = "Produce raw SystemVerilog verification/testbench source only."
    elif task_kind == "assertion":
        role = "Produce raw SystemVerilog source with the requested assertions only."
    else:
        role = "Produce raw Icarus-compatible synthesizable SystemVerilog/Verilog source only."
    multi = (
        " If multiple target files are requested, separate them with exact "
        "`// TARGET_FILE: <path>` header comments."
        if multi_file
        else ""
    )
    return (
        "You are the ACE-RTL generator. "
        f"{role}{multi} Do not wrap the answer in Markdown fences or add commentary."
    )


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
