# Role Guidance

## Input Filter

Before any role agent or external model call, filter inputs through the evaluator boundary in `agent-workflow.md`. Roles may see public task text, generated targets, candidates, history, and sanitized evaluator summaries only.

Do not give roles private scorer assets, real test files, expected-output code, mutation definitions, harness internals, or raw logs that reveal private test structure. For `verif/` generation tasks, the generated target file is allowed; private scoring tests are not.

## Generator

Input:

- task specification
- generated target file list
- current candidate
- primary failure stage and signature
- reflector focused error and fix guidance
- coordinator context and preserve constraints
- immutable helper context, when allowed

Output complete replacement code only.

- Single target: return the full file.
- Multiple generated targets: return sections headed by `// TARGET_FILE: <filename>`.
- Do not return diffs, shell commands, partial snippets, or explanatory prose.
- Preserve public module names, parameters, ports, reset polarity, clocking, latency, and checked output timing.
- For native simulator compatibility, declare procedural temporaries before executable statements.
- If the latest primary stage is `LOCAL_PREFLIGHT` or `CANDIDATE_COMPILE`, fix source hygiene and Icarus syntax first. Do not continue functional debugging until the target compiles.

## Reflector

Input:

- latest real evaluator report
- local preflight report, when present
- primary failure stage/signature selected by the runner
- current candidate
- specification and task type
- recent history
- per-signal mismatch data when available

Output:

```text
FOCUSED_ERROR:
ROOT_CAUSE:
FIX_GUIDANCE:
PRESERVE_CONSTRAINTS:
REGRESSION_RISKS:
```

Stay on the same focused error until solved unless the failure category changes or progress stalls.
For compile failures, prefer real `sim.log` compiler lines over wrapper pytest tracebacks. For local preflight failures, treat the issue as generated-source hygiene, not CVDP infrastructure.

## Coordinator

Maintain:

- mismatch trend
- best candidate
- recent attempts
- focused error history
- primary failure signature trend
- candidate hash/no-change streak
- implementation changes
- fix effectiveness
- restart count

Decision output:

```text
DECISION: continue | revert_to_best | accept_detour | fresh_start | exhausted
REASON:
GENERATOR_CONTEXT:
BEST_CANDIDATE:
HISTORY_SUMMARY:
```

Regression policy:

- Improve metric: update best.
- Compile, no-cheat, contract, or semantic failure: reject and revert.
- Repeated `LOCAL_PREFLIGHT` or `CANDIDATE_COMPILE` signatures: force fresh start after 3 repeats.
- Functional regression over 10 percent from best: revert.
- After three consecutive functional reverts, accept a detour only if runnable, structurally valid, no-cheat clean, no worse than `max(2 * best, best + 5000)`, and below 100000 mismatches.

Restart policy:

- Force restart after three consecutive no-code-change iterations.
- Otherwise consider restart only after at least 12 failures since last restart.
- Continue if there is any meaningful progress.
