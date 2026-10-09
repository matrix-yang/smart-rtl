# Agent Components

Use these local components when running or extending ACE-RTL. Keep scripts
inside `skills/ace-rtl/scripts/`; do not create a repository-level scripts
directory for agent logic.

## Entry Point

- `scripts/ace_agent_runner.py`: wrapper that sets the workspace, keeps default
  CVDP paths under the current repo, adds the bundled package to `sys.path`,
  and launches `ace_rtl_agent.main`.
- `scripts/ace_cvdp_native_runner.py`: lean CVDP-native evaluator loop with
  SystemVerilog post-processing, local preflight, real evaluator execution,
  primary failure extraction, and stage classification.
- `scripts/ace_chipverilog_runner.py`: one-task or 64-task ChipVerilog runner
  with unique run directories and suite summaries.

## Package Map

- `scripts/ace_rtl_agent/main.py`: CLI and benchmark dispatch.
- `scripts/ace_rtl_agent/agent.py`: top-level run orchestration.
- `scripts/ace_rtl_agent/rtl_generator.py`: initial and iterative RTL prompt
  assembly.
- `scripts/ace_rtl_agent/focused_debugger.py`: targeted fix prompt flow.
- `scripts/ace_rtl_agent/llm_centric_debugger.py`: model-guided debug loop.
- `scripts/ace_rtl_agent/fresh_start_coordinator.py`: restart decisions.
- `scripts/ace_rtl_agent/primary_failure_extractor.py`: bounded failure
  summaries for the next role call.
- `scripts/ace_rtl_agent/rtl_validator.py`: structural and syntax-oriented RTL
  validation before benchmark execution.
- `scripts/ace_rtl_agent/context_builder.py`: compact task-specific guidance.
- `scripts/ace_rtl_agent/utils/cvdp_integration.py`: CVDP adapter.
- `scripts/cvdp_nvidia_inference_factory.py`: CVDP custom model factory for
  NVIDIA Inference API or `ACE_RTL_LLM_SCRIPT` calls.
- `scripts/ace_rtl_agent/llm_backend.py`: NVIDIA Inference/custom backend.
- `scripts/ace_rtl_agent/utils/llm_client.py`: retrying LLM client wrapper used
  by the reusable role components.
- `scripts/ace_rtl_agent/utils/verilogeval_integration.py`: VerilogEval
  adapter.
- `scripts/ace_rtl_agent/utils/chipbench_integration.py`: ChipBench adapter.
- `scripts/ace_rtl_agent/utils/chipverilog_integration.py`: ChipVerilog task
  discovery, allowed-context construction, isolated golden evaluation, and
  bounded feedback adapter.
- `scripts/ace_rtl_agent/utils/ace_model_client.py`: external model client used
  only when the user explicitly assigns external models to roles.

## Prompt Reuse

For CVDP, preserve the CID-specific guidance files under
`references/benchmarks/cvdp/` and route them into the generator, reflector, and
coordinator context. For other RTL benchmarks, reuse the same role split but
swap the benchmark adapter and public task context.

Do not pass hidden tests, private scorer internals, expected-output code, or
mutation definitions to the role prompts. Only public task text, generated
targets, candidate source, bounded diagnostics, and sanitized evaluator
summaries are allowed.

## Extension Guidance

- Add new benchmark adapters under `scripts/ace_rtl_agent/utils/`.
- Add new prompt or task guidance under `references/`, then link it from
  `SKILL.md` or the benchmark-specific runbook.
- Keep wrapper defaults local to the workspace so future runs do not silently
  depend on outside CVDP checkouts, datasets, or environments.
