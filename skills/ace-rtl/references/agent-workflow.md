# Agent Workflow

## Shared State

Track:

- task id, category, dataset, and generated target files
- immutable helper/context files
- real evaluator command and working directory
- role-to-model map
- maximum iterations and sample id
- allowed input set for role agents and external models
- iteration records
- primary failure stage/signature for the latest iteration
- candidate hash and no-change streak
- best candidate path and metric
- bounded evaluator report path for the latest iteration

## Default Run Budget

Unless the user overrides the run budget:

- use `max_iterations = 30` for each independent process
- run up to 4 CVDP datapoints in parallel by default
- run 5 independent parallel processes per datapoint
- keep separate working directories, histories, prompts, generated files, and evaluator outputs for each process
- terminate sibling processes for the same datapoint after one process has a confirmed real evaluator pass

## Evaluator Boundary

The evaluator may run private benchmark tests, but the iteration loop must not separately read private scorer assets or inject private files into generator, reflector, coordinator, or external-model prompts.

Allowed into the loop:

- public JSONL prompt text
- public context files that are task inputs, excluding scorer/test internals
- generated target files and current candidate code
- high-confidence local preflight diagnostics from generated source only
- bounded evaluator reports emitted by the real CVDP run
- compile diagnostics, runtime failure summaries, mismatch counts, first failing public signal values, and aggregate coverage or mutation scores that appear in those evaluator reports
- role history and coordinator summaries derived only from allowed inputs

Disallowed during iteration:

- real test files, private harness code, private checker code, hidden scoring scripts, or expected-output functions
- mutation definitions or injected bug internals
- full private wave dumps or logs that reveal private test structure
- any file under a scorer/test directory unless it is explicitly the generated target listed by the datapoint

For tasks that generate a `verif/` target, the generated file itself is allowed. Private tests that score that generated file remain disallowed.

## Iteration Loop

The default implementation should use the local component package in
`skills/ace-rtl/scripts/ace_rtl_agent/`, launched by
`skills/ace-rtl/scripts/ace_agent_runner.py`. Prefer those components over
rewriting role loops, prompt assembly, benchmark adapters, validation, or
failure extraction.

For the lean native runner, reflector feedback must come from
`FocusedDebugger.create_focused_prompt(...)`, and coordinator restart decisions
must come from `FreshStartCoordinator.should_fresh_start(...)`. Do not add a
second native-runner reflector/coordinator prompt or a custom regex feedback
summarizer in front of those components.

```text
state = initialize(task, role_model_map)
candidate = initial_generation_or_existing_candidate(task)
result = real_evaluator(candidate)
error_report = bounded_evaluator_report(result)
state = coordinator.record(iteration=0, candidate, error_report)

for iteration in 1..max_iterations:
    reflection = FocusedDebugger.create_focused_prompt(candidate, error_report, task)
    decision = FreshStartCoordinator.should_fresh_start(state)

    if decision == exhausted:
        break
    if decision == fresh_start:
        candidate_input = empty_or_spec_only
    else:
        candidate_input = candidate

    candidate = generator.generate(task, candidate_input, reflection, decision.context)
    candidate = normalize_generated_source(candidate)
    preflight = run_high_confidence_preflight(candidate)
    inject_only_generated_targets(candidate)
    if preflight.failed:
        result = local_preflight_failure(preflight)
    else:
        result = real_evaluator(candidate)
    error_report = bounded_evaluator_report(result)
    state = coordinator.record(iteration, candidate, error_report, reflection)

    if result.passed:
        return passed

return failed
```

## Parallel Races

Run up to 4 CVDP datapoints in parallel by default. Within each datapoint, run independent processes in separate working directories and separate histories. Default is 5 parallel processes per datapoint. First confirmed real pass wins for that datapoint. Terminate sibling processes for the same datapoint after a real evaluator pass.

## Artifacts

Save:

- iteration candidate code
- prompt/context sent to any external model
- raw external model responses when used
- error reports
- `evaluator_error_report.txt` input passed to reflector/coordinator components
- post-process/preflight reports
- primary failure stage/signature
- mismatch counts and per-signal mismatch data
- coordinator state
- final real evaluator report

## Local Components

Use `references/agent-components.md` when a run needs the reusable
implementation details, prompt locations, supported benchmark adapters, or
extension points for non-CVDP RTL tasks.
