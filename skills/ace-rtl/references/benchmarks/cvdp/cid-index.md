# CVDP Task Index

Choose guidance by task behavior and target file, not by a hard-coded partial
task-ID list. The requested CVDP category is a runtime filter only; it should
not select a bespoke script path.

## Task Shapes

| Typical target | Task shape | Guidance |
| --- | --- | --- |
| `rtl/*.sv` | Partial or full RTL generation, feature work, or repair | Preserve public module interfaces, timing, reset behavior, and synthesizability. |
| `rtl/*.sv` | Lint, area, power, or performance optimization | Keep functional sanity first, then improve only the requested metric. |
| `verif/*.sv` | Coverage or stimulus generation | Produce finite legal stimulus that finishes cleanly and leaves coverage collection to the harness. |
| `verif/*.sv` | Checker testbench generation | Clean DUT must pass; real spec mismatches must fail injected buggy variants. |
| `rtl/*.sv` or `verif/*.sv` | Assertion insertion | Add observer-only assertions without changing existing design behavior. |
| multi-file repo patch | Agentic multi-file implementation | Touch only generated/allowed project files and keep wrappers, cores, and helpers consistent. |

For any requested CVDP category, inspect:

- `categories`
- `prompt`
- `context` and `output.context` paths
- harness `.env` services
- whether scoring expects coverage, mutation detection, sanity, or synthesis

## Dataset Selection

Use the user's requested dataset when supplied. Otherwise inspect local CVDP
dataset JSONL files under the workspace dataset root and filter rows whose
`categories` contain the requested category. Do not hard-code dataset-to-task
maps in scripts or prompts unless the map is complete for the dataset family.
