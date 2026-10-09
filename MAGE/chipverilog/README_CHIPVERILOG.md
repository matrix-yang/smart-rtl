# MAGE + DeepSeek V4 on ChipVerilog

This is an isolated adaptation of MAGE. The original repository at
`/nfs/FM/yangyifan/rtl_bench/MAGE` is unchanged.

The adapted loop uses `RTLGenerator -> five candidates -> ChipVerilog golden
evaluation -> RTLEditor`. `TBGenerator` and `SimJudge` are bypassed because the
golden evaluator is authoritative. Multi-module prompts include recursively
discovered dependency RTL and macro files, but never include the target
reference implementation.

Build:

```bash
docker compose -f compose.chipverilog.yaml build
```

Inspect task packs without an API call:

```bash
docker compose -f compose.chipverilog.yaml run --rm mage-chipverilog \
  python3 scripts/run_chipverilog.py \
  --chipverilog-root /chipverilog \
  --output-root /experiments/dry/output \
  --log-root /experiments/dry/log \
  --modules mips_alu or1200_cpu fpu \
  --dry-run
```

One-task smoke:

```bash
docker compose -f compose.chipverilog.yaml run --rm \
  -e DEEPSEEK_API_KEY mage-chipverilog \
  python3 scripts/run_chipverilog.py \
  --chipverilog-root /chipverilog \
  --output-root /experiments/smoke/output \
  --log-root /experiments/smoke/log \
  --model-name mage_deepseek_v4_smoke \
  --modules mips_alu \
  --max-candidates 1 \
  --editor-steps 0
```

The default main configuration uses five candidates and four editing steps.
Every LLM and verifier invocation is appended to each task's `telemetry.jsonl`.
Full-suite run:

```bash
docker compose -f compose.chipverilog.yaml run --rm \
  -e DEEPSEEK_API_KEY mage-chipverilog \
  python3 scripts/run_chipverilog.py \
  --chipverilog-root /chipverilog \
  --output-root /experiments/main/output \
  --log-root /experiments/main/log \
  --model-name mage_deepseek_v4
```

Per-task outputs include `rtl.sv`, all candidate snapshots, golden reports,
`summary.json`, `telemetry.jsonl`, and the ordinary MAGE prompt/response logs.

In golden-only mode no `tb.sv` is generated or required. `RTLEditor` receives
the private evaluator's sanitized feedback in `sim_failed_log`; its prompt marks
the generated testbench as unavailable instead of attempting to read a file.

Runs resume at task granularity by default. A task with a valid, matching
`summary.json` and non-empty `rtl.sv` is skipped and included in the final
manifest. An incomplete task directory is moved to
`output/archived_attempts/<task>__<UTC timestamp>` before a clean retry. Use
`--rerun-completed` only when completed tasks should intentionally be archived
and regenerated.

`RTLEditor` requests at most 4096 output tokens, accepts only one local
`replace_content_by_matching` patch (maximum 6000 characters and 120 lines per
old/new fragment), and retries one malformed or truncated JSON response. LLM
telemetry records `finish_reason`, the requested output cap, truncation state,
and structured `editor_output_error` events.
