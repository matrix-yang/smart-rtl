# Run ACE-RTL on ChipVerilog

## Boundary

- Use the real `tools/formal_equivalence.py verify` evaluator for every candidate.
- Role models may receive `Src/**/description.txt`, public family macro files,
  selected fixed dependency RTL, current candidates, and bounded evaluator feedback.
- Never send the target reference RTL, testbench, proof scripts, or raw evaluator
  artifacts to Generator, Reflector, or Coordinator.
- Fail closed before invoking the golden-backed evaluator unless the candidate
  defines the expected target module exactly once and defines no other module.
  A testbench-only, wrapper-only, wrong-top, or mixed RTL/testbench response is
  recorded as `invalid_candidate` and cannot be completed with golden RTL.
- Treat `equivalence_timeout`, `equivalence_error`, `reference_fail`, and
  `evaluator_error` as unknown, not as confirmed functional failures.
- Record every LLM attempt in per-task `telemetry.jsonl`, including the agent
  role, latency, provider usage fields, and normalized input/output/reasoning/
  cached/total tokens. Per-task and suite summaries aggregate those counters.

## Container preflight

The ACE image contains `iverilog`, `vvp`, and `yosys`. Mount ChipVerilog
read-only; evaluator candidates and artifacts are written under ACE-RTL's
`outputs/` tree.

```bash
docker compose -f compose.ace-rtl.yaml run --rm ace-rtl-agentic \
  bash -lc 'command -v iverilog vvp yosys && \
  test -f "$CHIPVERILOG_ROOT/tools/formal_equivalence.py"'
```

## Inspect task packs without an LLM call

```bash
docker compose -f compose.ace-rtl.yaml run --rm ace-rtl-agentic \
  python3 skills/ace-rtl/scripts/ace_chipverilog_runner.py \
  --chipverilog-root "$CHIPVERILOG_ROOT" \
  --modules mips_alu or1200_alu --dry-run
```

## One-task smoke

```bash
export DEEPSEEK_API_KEY='<key>'
docker compose -f compose.ace-rtl.yaml run --rm ace-rtl-agentic \
  python3 skills/ace-rtl/scripts/ace_chipverilog_runner.py \
  --chipverilog-root "$CHIPVERILOG_ROOT" \
  --modules mips_alu \
  --model deepseek-flash \
  --max-iterations 1 \
  --parallel-experiments 1
```

## Full suite

The runner discovers all 64 tasks. ACE-RTL's paper-style parallel trajectories
can be enabled per task with `--parallel-experiments 5`; start with one task and
one trajectory before allocating a full run.

```bash
docker compose -f compose.ace-rtl.yaml run --rm ace-rtl-agentic \
  python3 skills/ace-rtl/scripts/ace_chipverilog_runner.py \
  --chipverilog-root "$CHIPVERILOG_ROOT" \
  --model deepseek-flash \
  --max-iterations 30 \
  --parallel-experiments 5
```

Every invocation gets a unique UTC run id unless `--run-id` is supplied.
Outputs include per-iteration candidates and prompts, isolated evaluator runs,
sanitized feedback, telemetry, per-task `summary.json`, and suite-level
`run_summary.partial.json` / `run_summary.json`.
