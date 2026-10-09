# DeepSeek-V4.1-Flash dual-benchmark validation

## Result

On 2026-10-08 UTC, the unified Smart-RTL runtime completed one public sample
from each supported benchmark with DeepSeek-V4.1-Flash. The DeepSeek API model
ID was `deepseek-flash`; every model role used reasoning effort `low`.

| Benchmark | Public sample | Status | Rounds | Candidates | LLM calls | Evaluator calls | Total tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| CVDP | `cid003/cvdp_agentic_binary_to_gray_0003` | passed | 1 | 1 | 3 | 1 | 10,359 |
| ChipVerilog | `mips_IF_stage` | passed | 1 | 2 | 4 | 2 | 23,626 |

The CVDP candidate passed the native Icarus/cocotb evaluation with zero reported
errors. For ChipVerilog, the first candidate produced an evaluator-inconclusive
`$mem_v2` SAT-import result. Smart-RTL preserved the competing reset hypothesis;
the second candidate compiled and passed the official formal-equivalence flow
by temporal induction at depth 16. No role call failed and no candidate was
rejected by the public contract checker.

## Fixed protocol

- Controller: `smart_rtl_adaptive_round_director/v2`
- Physical workers: 1
- Maximum rounds: 6
- Maximum LLM calls: 24 per sample
- Maximum fail-closed evaluator calls: 12 per sample
- Maximum observed total tokens: 250,000 per sample
- CVDP evaluator seed: 1
- ChipVerilog induction depth: 16
- API base URL: `https://api.deepseek.com`

The run resolved `deepseek-flash` through the authenticated `/models` endpoint
before execution. Credentials were supplied only through the runtime environment
and were not written to the repository.

## Historical reference

The samples were selected because earlier DeepSeek-V4-Pro runs had passed the
same benchmark contracts:

| Benchmark | Historical status | Rounds | LLM calls | Evaluator calls | Total tokens |
|---|---:|---:|---:|---:|---:|
| CVDP `cid003` | passed | 2 | 4 | 1 | 14,528 |
| ChipVerilog `mips_IF_stage` | passed | 2 | 4 | 1 | 26,386 |

These figures are provenance references, not a controlled model comparison.
The model changed, and the ChipVerilog Flash run evaluated two hypotheses, so
call and token differences are descriptive only.

## Evidence and claim boundary

Each run preserved its controller metadata, call ledger, selected RTL,
sanitized evaluator feedback, and native evaluator artifacts. Raw run
directories are intentionally ignored by Git because they may contain large
generated outputs and benchmark-derived material. The release gate was accepted
only after the terminal `result.json` and underlying evaluator result agreed on
`passed`.

This result supports a narrow claim: the same Smart-RTL controller and runtime
can execute end to end, fail closed, on these two samples using the official
CVDP and ChipVerilog evaluators. Two samples do not establish aggregate
benchmark quality, model superiority, statistical robustness, or suitability
for unseen private RTL. A broader claim requires a preregistered multi-task
campaign with repeated seeds and cost/quality reporting.
