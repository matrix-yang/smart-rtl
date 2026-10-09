---
name: ace-rtl
description: Consolidated ACE-RTL skill for running an iterative generator-reflector-coordinator RTL agent on CVDP and ChipVerilog. Use for benchmark setup, no-cheating runs, role-based external LLM routing, unique output logging, task-aware prompt guidance, and real evaluator pass-rate reporting.
---

# ACE-RTL

Use this skill as the entry point for ACE-RTL work. It contains the agent
protocol, CVDP and ChipVerilog runbooks, role guidance, optional external LLM
bridge, and task-driven benchmark prompt guidance.

## Core Rules

- Clone the official CVDP benchmark code from
  `https://github.com/NVlabs/cvdp_benchmark.git` into this workspace and follow
  that checkout's own setup instructions. Do not use a CVDP checkout, setup,
  environment, or data outside the current workspace unless the user explicitly
  authorizes it.
- Report final pass/fail only from the real CVDP evaluator.
- Report ChipVerilog pass/fail only from its real `formal_equivalence.py`
  evaluator. Keep the target reference RTL, testbench, proof scripts, and raw
  evaluator artifacts outside all role-model prompts.
- In ChipVerilog mode, treat evaluator/reference failures and equivalence
  timeout/errors as unknown rather than candidate functional failures.
- When Docker is unavailable or the daemon socket is inaccessible, use the
  checked-in native runner `scripts/ace_cvdp_native_runner.py` after
  `setup-native.md`. Do not hand-roll Docker-to-native path translation.
- Native CVDP runs must execute under the repo-local `.cvdp-native` Python
  environment when it exists. Missing Python packages, pytest
  collection/import failures, or missing harness env variables are
  `INFRA_SETUP`, not candidate RTL failures.
- Before starting any native CVDP datapoint workers or LLM calls, verify the
  selected rows' harness services have their required simulator tools on
  `PATH`. Rows that request Xcelium/Cadence services require a configured
  simulator command such as `irun` or `xrun` and usually `imc`; if either is
  missing, use the `eda_tool_setup` skill to configure site-provided EDA tools
  before classifying the run as setup-blocked.
- For native commercial/Cadence rows, copied harness command files must have
  container paths, configured Cadence command names, and a default Cadence
  timescale option rewritten before execution. The runner should inject
  `-timescale 1ns/1ps` into `xrun`/`irun` commands unless the harness already
  sets a timescale.
- Do not classify candidate RTL reset/register/output initialization failures
  as `INFRA_SETUP` just because a simulator or assertion line says "failed to
  initialize". Setup classification is for missing tools, missing Python
  modules, invalid harness environment variables, license/database/tool
  initialization failures, collection failures, or native path/setup breakage.
- The native runner must post-process generated SystemVerilog before CVDP
  evaluation, save bounded evaluator reports, and feed reports through the
  reusable `FocusedDebugger` reflector and `FreshStartCoordinator` coordinator
  components.
- Preserve every generated target path declared by the selected CVDP row. For
  multi-file targets, generator prompts must list every target file and require
  one complete `// TARGET_FILE: <path>` section per file.
- Native-runner generator prompts must be task-aware and data-driven from the
  selected CVDP row, target paths, prompt, and harness services. Do not
  hard-code a partial list of task IDs or inject one task shape's guidance into
  unrelated CVDP task types.
- Reusable RTL-agent components live under `scripts/ace_rtl_agent/` and must be
  launched through `scripts/ace_agent_runner.py`.
- Do not use hidden solutions, golden outputs, private reference code,
  expected answer tables, or injected bug internals.
- Do not hardcode datapoint-specific answers or recognizable test vectors.
- Do not modify immutable harness, source, or helper files unless the task
  explicitly lists them as generated targets.
- During native-runner iteration, role agents and external models may receive
  public prompt/context, generated targets, current candidates, role history,
  and bounded evaluator reports emitted by the real CVDP run. Do not separately
  open, copy, or inject private scorer files, real test files,
  expected-output code, mutation definitions, or harness internals.
- Default CVDP scheduling is 4 datapoints in parallel, with 30 max iterations
  per process and 5 parallel independent processes per datapoint.
- Native evaluator subprocesses must be bounded and reaped after timeout or
  service exit.
- Every CVDP run must create a unique directory under `outputs/`, stream run
  logs to `run.log`, and write a Markdown result report with overall pass rate
  for the requested runtime filter, unique solved datapoint IDs, failed
  datapoint IDs, and useful run metadata.
- By default, do not call external LLM APIs. Use coding-agent reasoning and
  spawned role agents unless the user explicitly assigns a model to a role.
- If the user assigns a model to a role, use `scripts/llm_call.py` and
  `scripts/extract_solution.py`.

## Role Agents

Spawn three role agents for non-trivial runs:

- generator: produces the next complete target RTL, testbench, assertion, or
  optimization artifact.
- reflector: use the reusable `FocusedDebugger` component to analyze the latest
  evaluator report and create focused fix guidance.
- coordinator: use the reusable `FreshStartCoordinator` component to maintain
  history and decide restart.

Read `references/agent-workflow.md` for the full loop,
`references/role-guidance.md` for role prompts, and
`references/agent-components.md` when selecting or modifying the local reusable
implementation.

## CVDP Setup

- Dataset setup: `references/benchmarks/cvdp/setup-datasets.md`
- Native no-Docker setup: `references/benchmarks/cvdp/setup-native.md`
- Docker setup: `references/benchmarks/cvdp/setup-docker.md`
- Concrete run checklist: `references/benchmarks/cvdp/run-cvdp.md`
- EDA tool setup skill: `../eda_tool_setup/SKILL.md`

## CVDP Task Guidance

Read `references/benchmarks/cvdp/cid-index.md`, inspect the JSONL row, target
paths, prompt, and harness services, then choose guidance based on task behavior
rather than hard-coded task IDs.

## ChipVerilog

- Concrete run checklist: `references/benchmarks/chipverilog/run-chipverilog.md`
- Reusable adapter: `scripts/ace_rtl_agent/utils/chipverilog_integration.py`
- One-task/full-suite runner: `scripts/ace_chipverilog_runner.py`

## Optional LLM Calls

Read `references/llm-routing.md` only when the user explicitly assigns external
models to roles. The default hosted provider is NVIDIA Inference API with model
`nvidia/nemotron-3-ultra-550b-a55b`; users may replace that path with
`ACE_RTL_LLM_SCRIPT`.
