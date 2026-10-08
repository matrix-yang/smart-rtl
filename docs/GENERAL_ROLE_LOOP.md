# General ACE-compatible role loop

> Status: mechanism and ACE/CVDP live adapter implemented. The role path is
> covered by a no-network integration test, and the native verifier injection
> path has been replayed against persisted ACE candidates. A paid live search
> has not been started automatically.

## Purpose

This phase verifies that the complete multi-role search mechanism is executable
before any instruction is derived from successful or failed benchmark traces.
It intentionally separates two questions:

1. Can the role/search loop execute correct state transitions?
2. Do trace-derived role skills improve hidden RTL correctness on unseen tasks?

Only the first question is addressed here.

## Loop

```text
TaskContract
    |
    v
Planner ----> plan A ----> Builder ----> Verifier ----> Analyst
    |                         ^                           |
    +-------> plan B          |                           v
                              +---- Director decision <---+
                                      |
                         Expand / Refine / Backtrack / Stop
```

- The Planner creates distinct architectural or semantic hypotheses.
- The Builder is the only role allowed to produce RTL.
- The Verifier supplies immutable, structured EDA evidence.
- The Failure Analyst converts evidence into a bounded diagnosis without
  editing RTL.
- The Director is deterministic and owns only search topology.
- Every candidate is immutable; a repair or alternative creates a child.

## General instruction contract

The initial instruction provider is
`rtl_search.instructions.GeneralACEInstructions`. It contains no CVDP task ID,
ChipVerilog module name, historical fix, or task-specific threshold.

The common constraints match the original ACE-RTL responsibility boundary:

- preserve the public module interface, timing, reset, parameter, and protocol
  requirements;
- treat context, tests, harnesses, checkers, build scripts, and evaluators as
  immutable;
- return complete declared target files, not patches;
- never generate a testbench in an RTL-generation task;
- never inspect private tests or golden RTL;
- never self-declare correctness; only fail-closed EDA evidence can produce a
  pass;
- keep diagnosis separate from code generation;
- let hard EDA state override LLM confidence.

The engine depends on the `RoleInstructionProvider` interface rather than this
specific implementation. A future trace-derived skill can replace Planner,
Builder, and Analyst instructions without changing scheduling, tree lineage,
candidate contracts, or verification.

## Immutable contracts

`TaskContract` separates:

- public specification;
- exact generated target paths;
- immutable public context.

`CandidateArtifact` is immutable and is rejected before the Verifier if it:

- omits a target;
- emits an empty target;
- adds an undeclared file.

The same boundary prevents the previously observed testbench-only and golden
substitution classes of evaluator contamination from becoming valid search
states.

## Mechanism smoke

The no-API backend intentionally drives this sequence:

```text
plan streaming
  candidate_001: missing target       -> Refine
  candidate_002: localized sim fail   -> Refine
  candidate_003: structural regression -> Backtrack(candidate_002)
  candidate_004: alternative sim fail -> Expand

plan staged
  candidate_005: fail-closed pass      -> Stop
```

The important property is that `candidate_004` is a child of
`candidate_002`, the selected backtrack destination, rather than a child of the
rejected `candidate_003`. Unlike the earlier historical replay, this smoke test
therefore exercises the changed trajectory directly.

Run it with:

```bash
PYTHONPATH=skills/ace-rtl/scripts \
python3 -m ace_cvdp_native.rtl_search.smoke \
  --output /tmp/rtl_search_loop_smoke.json
```

## Evidence boundary

Passing this smoke proves only:

- role interfaces compose;
- general instructions reach the intended roles;
- pending breadth survives depth-first refinement;
- structural backtracking changes the actual next parent;
- every control action is executable;
- fail-closed pass stops delivery.

It does not prove that an LLM follows the instructions, that the general
instructions are good, that the Director selects the right branch on real RTL,
or that the system improves correctness, tokens, or wall time.

## Live adapter

The adapter was added without changing this engine or the accepted static
`ace_cvdp_native/cli.py`:

- Planner: bounded strict-JSON call through ACE's Coordinator component;
- Builder: ACE Generator call plus current complete-target extraction and SV
  normalization;
- Verifier: a no-LLM prebuilt-candidate model injected into the existing native
  CVDP Copilot processor;
- Analyst: bounded strict-JSON call through ACE's Reflector component;
- Director: unchanged deterministic `TreePolicy`.

`CallLedger` records every attempted role call. Provider usage fields are summed
when present; absent usage remains explicitly marked `missing`, rather than
being estimated or treated as zero.

Before a live row is persisted, all patch/output answer bodies are erased. The
roles receive only the public prompt, public context, generated candidate, and
the bounded evaluator report already used by ACE. The original evaluator and
static ACE CLI remain byte-identical.

Every candidate in one search uses the same configured `COCOTB_RANDOM_SEED`.
This removes seed drift from parent/child comparisons; it does not make the
single-seed oracle strong. Multi-seed verification is a later Verify-escalation
mechanism, not part of this general-loop shakedown.

The native replay exposed why this matters. The persisted historical
`dynamic_equalizer_0001` iteration-10 candidate and new injected candidate were
byte-identical, as were all non-runtime harness sources. It passed historically
but failed under a new random seed with small numerical mismatches, and then
failed consistently in two repeated evaluations under fixed seed 1. In contrast,
the persisted `binary_to_gray_0003` candidate passed through the new adapter.
Therefore the dynamic-equalizer replay is evidence of seed-sensitive oracle
weakness, not adapter corruption, and must not be used as a deterministic adapter
acceptance oracle.

The next gate is one paid `dynamic_equalizer_0001` live search with the general
instruction provider. Only after that loop is reproducible should trace analysis
be used to implement a new instruction provider and test held-out samples.
