# BW-RTL live ACE/CVDP adapter

## Scope

The live entry point runs exactly one datapoint through:

```text
ACE Planner -> ACE Builder -> native CVDP Verifier -> ACE Analyst
                                              |
                              deterministic TreePolicy
```

It does not launch ACE's five parallel attempts and does not use a learned
Director or trace-derived repair skill. Breadth, depth, and backtracking are
controlled by the general tree mechanism.

## cid003 shakedown command

The API key and optional proxy headers are inherited by name from the host; no
secret is written into the command.

```bash
docker run -d \
  --name bw-rtl-cvdp-cid003-dynamic-equalizer \
  --entrypoint python3 \
  -v "$PWD":/workspace/ace \
  -w /workspace/ace \
  -e PYTHONPATH=/workspace/ace/skills/ace-rtl/scripts:/workspace/ace \
  -e CVDP_NATIVE_HARNESS=1 \
  -e DEEPSEEK_API_KEY \
  -e DEEPSEEK_BASE_URL \
  -e DEEPSEEK_EXTRA_HEADERS \
  -e ACE_RTL_LLM_TIMEOUT=1800 \
  -e ACE_RTL_MAX_TOKENS=192000 \
  ace-rtl-agentic:deepseek \
  -m ace_cvdp_native.rtl_search.live \
  --workspace /workspace/ace \
  --cvdp-repo /workspace/ace/external/cvdp_benchmark \
  --dataset /workspace/ace/datasets/cvdp-benchmark-dataset/cvdp_v1.1.0_agentic_code_generation_no_commercial.jsonl \
  --outputs-root /workspace/ace/outputs \
  --cid cid003 \
  --id cvdp_agentic_dynamic_equalizer_0001 \
  --planner-model deepseek-flash \
  --generator-model deepseek-flash \
  --analyst-model deepseek-flash \
  --branch-width 2 \
  --max-plans 4 \
  --max-steps 12 \
  --llm-timeout 1800 \
  --evaluator-timeout 300 \
  --evaluator-seed 1
```

Monitor with:

```bash
docker logs -f bw-rtl-cvdp-cid003-dynamic-equalizer
```

The first output line is `RUN_DIR=...`. That directory contains:

- `metadata.json`: task, models, search settings, and fixed evaluator seed;
- `call_ledger.json`: every successful/failed role call and available provider
  token usage;
- `candidates/`: immutable complete Builder outputs;
- `verifier/`: candidate manifests, native CVDP worktrees, raw evaluator
  results, and bounded feedback;
- `result.json`: tree actions, selected hash, terminal status, and telemetry;
- `selected_candidate/`: delivered candidate files and manifest.

Exit code `0` means fail-closed PASS, `1` means the legal search ended without a
PASS, and `2` means setup/adapter infrastructure failure.
