# Run CVDP

## Checklist

1. Clone the official CVDP benchmark repository into the current workspace, for
   example `external/cvdp_benchmark`, and set `CVDP_REPO_ROOT` to that clone.
2. Read and follow the official setup instructions from the cloned
   `NVlabs/cvdp_benchmark` checkout. If Docker is unavailable, complete
   `setup-native.md` and its backend verification block before launching
   evaluation.
3. If `CVDP_DATASET_ROOT` is unset, follow `setup-datasets.md` to download the
   v1.1.0 JSONL files into local `datasets/cvdp-benchmark-dataset`, then export
   `CVDP_DATASET_ROOT`.
4. Create a unique run directory under `outputs/` before row materialization or
   LLM calls. All logs, prompts, responses, candidates, evaluator outputs, and
   summaries for this run must stay under that directory.
5. Select the dataset JSONL from `CVDP_DATASET_ROOT` and filter rows whose
   `categories` contain the requested CID.
6. Before any datapoint worker or LLM call starts, verify the selected rows'
   harness services have their required native simulator tools available.
   Xcelium/Cadence services require the configured simulator command such as
   `irun` or `xrun` and usually `imc`; if missing, use the `eda_tool_setup`
   skill to configure site-provided EDA tools before stopping.
7. Run the ACE-RTL role-agent loop with real CVDP evaluation. Prefer the local
   component wrapper:
   `python skills/ace-rtl/scripts/ace_agent_runner.py --workspace "$PWD" --cvdp-benchmark-path "$CVDP_REPO_ROOT" ...`
   Use `skills/ace-rtl/scripts/ace_cvdp_native_runner.py` for the lean native
   runner path. Do not translate Docker compose commands manually.
8. Use the default run budget unless the user overrides it: 4 datapoints in
   parallel, 30 max iterations per process, and 5 parallel independent
   processes per datapoint.
9. Stop sibling processes for a datapoint after the first confirmed real
   evaluator pass.
10. Save all iteration artifacts and final reports.
11. Report pass rate from real CVDP summary files only.

## Iteration Failure Classes

- `LOCAL_PREFLIGHT`: generated source failed high-confidence local checks before
  private CVDP evaluation. This is a candidate-source issue, not infra/setup.
- `CANDIDATE_COMPILE`: CVDP reached simulator build and rejected the generated
  source. Prefer bounded simulator diagnostics over wrapper pytest lines.
- `CANDIDATE_FUNCTIONAL`: CVDP compiled and ran, but checks failed.
- `INFRA_SETUP`: role-call/API failures before candidate extraction or
  evaluation, repeated empty external-model responses, missing simulator
  binaries, license checkout/setup failures, coverage database/version
  mismatches, missing `MDV_XLM_HOME`, missing Python modules, pytest
  import/collection failures, missing harness environment variables, malformed
  native path mapping, missing env files, or CVDP checkout/dataset/env outside
  the workspace. Stop the run and fix setup before continuing.

## Integrity Checks

Before reporting, scan generated artifacts for hidden solution text, copied
golden code, hardcoded expected output arrays, datapoint-specific lookup tables,
forbidden file/shell operations, modifications to immutable files, or fake
coverage report generation. Reject suspicious passes and rerun cleanly.
