# Smart-RTL

Smart-RTL is an evidence-driven adaptive search controller for RTL generation.
It keeps multiple behavioral hypotheses when a specification is ambiguous,
chooses a variable-length set of actions from the live search state, and treats
only a fail-closed benchmark evaluator as proof of success.

The same controller runtime supports two public benchmark adapters:

- [CVDP](https://github.com/NVlabs/cvdp_benchmark), using its native
  simulation/evaluation harness.
- [ChipVerilog](https://github.com/HKUSTGZ-MICS-LYU/ChipVerilog), using its
  compile and formal-equivalence flow while keeping target references out of
  model prompts.

Smart-RTL is derived from NVIDIA ACE-RTL. Historical `BW-RTL`, `BW_*`, and
`skills/bw-rtl` names remain in a few internal interfaces so older experiment
artifacts and environment variables continue to work.

## How it works

1. A Planner extracts competing implementation hypotheses and invariants from
   the public task contract.
2. A Director selects the next evidence-seeking actions. It may seed, refine,
   compare, fuse, roll back, or stop branches; there is no fixed top-K search
   width.
3. Builders generate candidate RTL. Analyst, Debater, and Judge roles provide
   bounded evidence and action-fidelity checks without seeing hidden targets.
4. The benchmark adapter runs a real simulator or formal evaluator, sanitizes
   its feedback, and records a durable evidence ledger.
5. A verified pass stops immediately. Unknown or incomplete evaluator results
   are never promoted to a pass.

`--physical-workers` controls execution concurrency only. It does not constrain
the Director's algorithmic branch width. See
[the adaptive-loop design](docs/BW_RTL_ADAPTIVE_LOOP_V1.md) and
[method specification](docs/BW_RTL_PAPER_METHOD_AND_FLOW.md) for details.

## Validation status

The 2026-10-08 release gate used the DeepSeek API model ID `deepseek-flash`
(DeepSeek-V4.1-Flash) with one physical worker and fixed call/token budgets.

| Benchmark | Sample | Result | Rounds | LLM calls | Evaluator calls | Tokens |
|---|---|---:|---:|---:|---:|---:|
| CVDP | `cid003/cvdp_agentic_binary_to_gray_0003` | passed | 1 | 3 | 1 | 10,359 |
| ChipVerilog | `mips_IF_stage` | passed | 1 | 4 | 2 | 23,626 |

The CVDP sample passed its native simulation harness. The selected ChipVerilog
candidate passed formal equivalence by temporal induction at depth 16. See
[the validation report](docs/VALIDATION.md) for the protocol, evidence summary,
historical reference, and exact claim boundary.

This is an end-to-end two-sample release check, not evidence of aggregate
benchmark superiority or statistical robustness.

## Requirements

The recommended path uses Docker and Docker Compose. You also need:

- a DeepSeek API key with access to `deepseek-flash`;
- local CVDP and ChipVerilog checkouts;
- the public CVDP dataset files for CVDP runs;
- enough CPU, memory, and disk for Icarus Verilog, Verilator, and Yosys.

The supplied image builds the open-source simulator toolchain and installs the
pinned packages from `requirements.txt`. Commercial EDA-dependent CVDP rows
still require your site-provided tools and licenses.

## Setup

From the repository root:

```bash
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY. The populated file is ignored by Git.

mkdir -p external datasets outputs artifacts/experiment
git clone https://github.com/NVlabs/cvdp_benchmark.git external/cvdp_benchmark
git clone https://github.com/HKUSTGZ-MICS-LYU/ChipVerilog.git external/ChipVerilog

# Creates .cvdp-native, installs pinned Python dependencies, and downloads the
# public CVDP dataset files. Existing checkouts and datasets are reused.
./skills/ace-rtl/scripts/setup_cvdp_native_env.sh

docker compose -f compose.ace-rtl.yaml build
docker compose -f compose.ace-rtl.yaml config --quiet
```

You may export `DEEPSEEK_API_KEY` in the shell instead of storing it in `.env`.
The default endpoint is `https://api.deepseek.com`; override
`DEEPSEEK_BASE_URL` only for a compatible endpoint. The public API model ID is
`deepseek-flash`, not the display name `DeepSeek-V4.1-Flash`.

Check the unified entry point:

```bash
docker compose -f compose.ace-rtl.yaml run --rm \
  --entrypoint /opt/ace-rtl-venv/bin/python \
  ace-rtl-agentic scripts/smart_rtl.py --help
```

## Run one CVDP sample

The following command reproduces the release-gate task and writes a unique
timestamped run below `artifacts/experiment/cvdp_smoke/`:

```bash
docker compose -f compose.ace-rtl.yaml run --rm \
  --entrypoint /opt/ace-rtl-venv/bin/python \
  ace-rtl-agentic scripts/smart_rtl.py cvdp \
  --workspace /workspace/ace \
  --cvdp-repo /workspace/ace/external/cvdp_benchmark \
  --dataset /workspace/ace/datasets/cvdp-benchmark-dataset/cvdp_v1.1.0_agentic_code_generation_no_commercial.jsonl \
  --cid cid003 \
  --id cvdp_agentic_binary_to_gray_0003 \
  --outputs-root artifacts/experiment/cvdp_smoke \
  --physical-workers 1 \
  --max-rounds 6 \
  --max-llm-calls 24 \
  --max-verifier-calls 12 \
  --max-total-tokens 250000 \
  --evaluator-seed 1
```

The runner verifies that exactly one row matches `--cid` and `--id`, checks the
required native tools before the first model call, and records the preflight in
the run directory.

## Run one ChipVerilog sample

List the discovered public tasks without making a model call:

```bash
docker compose -f compose.ace-rtl.yaml run --rm \
  --entrypoint /opt/ace-rtl-venv/bin/python \
  ace-rtl-agentic scripts/smart_rtl.py chipverilog \
  --chipverilog-root /workspace/chipverilog \
  --list-tasks
```

Run the validated `mips_IF_stage` sample:

```bash
docker compose -f compose.ace-rtl.yaml run --rm \
  --entrypoint /opt/ace-rtl-venv/bin/python \
  ace-rtl-agentic scripts/smart_rtl.py chipverilog \
  --workspace /workspace/ace \
  --chipverilog-root /workspace/chipverilog \
  --id mips_IF_stage \
  --artifacts-root artifacts/experiment \
  --physical-workers 1 \
  --max-rounds 6 \
  --max-llm-calls 24 \
  --max-verifier-calls 12 \
  --max-total-tokens 250000 \
  --evaluator-depth 16
```

All six model roles default to `DEEPSEEK_MODEL`. Each role also has an explicit
`--<role>-model`, `--<role>-max-tokens`, and
`--<role>-reasoning-effort` override; use `--help` to inspect the full contract.

## Outputs and failure semantics

Each run preserves enough state to audit or resume the reasoning path:

- `metadata.json` and the preflight/tool manifest;
- `result.json`, resource usage, and the model-call ledger;
- plans, Director decisions, candidate lineage, and sanitized evidence;
- selected RTL and isolated native evaluator artifacts;
- `EVALUATION_SUMMARY.json` for the ChipVerilog adapter.

Run directories under `outputs/` and `artifacts/` are ignored by Git. Review
them before sharing: prompts, generated RTL, and evaluator feedback can still
contain benchmark-derived or organization-sensitive information.

A compile-only result, print-only simulation, timeout, malformed evaluator
output, missing metric, or formal `unknown` remains non-passing. Resource caps
are enforced separately for rounds, model calls, evaluator calls, and observed
tokens.

## Test

Run the repository suite in the built image:

```bash
docker compose -f compose.ace-rtl.yaml run --rm \
  --entrypoint /opt/ace-rtl-venv/bin/python \
  ace-rtl-agentic -m pytest -q
```

The unified CLI can also be exercised from the local environment created by
the setup helper:

```bash
.cvdp-native/bin/python -m pytest -q tests/test_smart_rtl_cli.py
```

## Security and external services

Never commit API keys, EDA license values, private testbenches, hidden targets,
or raw confidential logs. Smart-RTL sends task prompts, generated candidates,
role history, and bounded evaluator feedback to the configured model provider.
Run it only when those materials may be shared with that provider and API costs
are acceptable. See [SECURITY.md](SECURITY.md) for the full handling policy.

## Repository layout

- `scripts/smart_rtl.py`: unified CVDP/ChipVerilog command.
- `skills/ace-rtl/scripts/ace_cvdp_native/rtl_search/`: controller, runtime,
  roles, evidence graph, and benchmark adapters.
- `skills/bw-rtl/`: versioned role instructions retained for compatibility.
- `scripts/run_bw_*_campaign.py`: restartable multi-task campaign drivers.
- `tests/`: controller, adapter, evaluator-integration, and CLI tests.
- `docs/`: design notes, validation evidence, and case studies.

## License and provenance

See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). External benchmark
repositories, datasets, tools, and model services are not vendored and retain
their own licenses and terms.

If you use the upstream ACE-RTL foundation, cite:

```bibtex
@article{deng2026ace,
  title={ACE-RTL: When Agentic Context Evolution Meets RTL-Specialized LLMs},
  author={Deng, Chenhui and Yu, Zhongzhi and Liu, Guan-Ting and Pinckney, Nathaniel and Khailany, Brucek and Ren, Haoxing},
  journal={NSF Workshop on Agents for Chip Design Automation (Agent4Chip)},
  year={2026}
}
```
