# RTL-Director v0: role-aware breadth, depth, and backtracking search

> Status: offline tree/policy/replay and generic role-loop engine implemented;
> real ACE/CVDP role adapter not started  
> Date: 2026-08-29  
> Initial anchor: CVDP cid003 `cvdp_agentic_dynamic_equalizer_0001`  
> Non-goal for v0: online token, dollar, wall-time, or parallel-slot budgeting

Implementation package:
`skills/ace-rtl/scripts/ace_cvdp_native/rtl_search/`. The first accepted-trace
replay loaded 10 nodes and produced the expected `iter6 -> iter5` structural
backtrack, `iter9 -> iter7` duplicate backtrack, and iteration-10 stop with no
unexpected backtrack. This is an observational policy audit, not yet a live
correctness or efficiency result.

The complete replaceable role loop and its general instruction contract are
documented in [`GENERAL_ROLE_LOOP.md`](./GENERAL_ROLE_LOOP.md).

## 1. Research question

Can an RTL multi-agent system preserve ACE-RTL's functional success while
exploring fewer redundant candidates by deciding, after every EDA observation,
whether to widen the search, deepen the current branch, or abandon and
backtrack from it?

The v0 thesis is deliberately narrower than the original budget-aware design:

> The main missing mechanism is not another persistent agent. It is a global,
> EDA-grounded controller that decides when to explore a different plan, when
> to repair the current implementation, and when a trajectory has ceased to be
> worth continuing.

Token use remains telemetry and an outcome metric, but it is not an input to the
v0 policy. A budget-aware policy should be attempted only after the search
mechanism itself improves the breadth/depth/backtrack decisions.

## 2. Scope reduction from the original proposal

| Original concept | v0 decision |
|---|---|
| Branch | Keep as `EXPAND` |
| Refine | Keep |
| Discuss | Optional subroutine for unresolved plan/spec conflicts |
| Verify | Mandatory environment transition after each code-producing action |
| Aggregate | Defer; it introduces multi-parent credit assignment |
| Stop | Keep |
| Heterogeneous budget vector | Defer; record usage only |
| Dynamic parallel width | Defer; initially cap expansion width by configuration |
| LLM value prediction | Optional ablation after deterministic scoring works |

The Director therefore chooses only:

```text
EXPAND | REFINE | BACKTRACK | STOP
```

`VERIFY` is not optional: each newly generated candidate is compiled and
simulated before another control decision. Debate and LLM judging do not sit on
the hot path in the MVP.

## 3. Roles and authority boundaries

### Director / Tree Controller

- Owns the global search tree and frontier.
- Chooses `EXPAND`, `REFINE`, `BACKTRACK`, or `STOP` from structured evidence.
- Does not write RTL and does not invent simulator results.
- Is deterministic in the first implementation.

### Planner

- At the root, proposes two deliberately distinct implementation plans or spec
  interpretations.
- Is called again only after a plan-level backtrack or a diagnosed spec/plan
  conflict.
- Produces explicit hypotheses and obligations, not RTL.

### Builder

- Implements a fresh plan for `EXPAND` or a bounded repair for `REFINE`.
- Replaces the current Generator/Refiner distinction with one code-producing
  role whose prompt is selected by the action.
- Must return a complete target bundle; partial output is a structural failure.

### Failure Analyst

- Converts compile/simulation evidence into a failure class, localization,
  confidence, and a small set of repair hypotheses.
- It may recommend an action, but the Director owns the transition.
- It is skipped for obvious manifest, missing-module, or compile failures that
  deterministic parsing can classify.

### Verifier

- Runs fail-closed structural checks, compile/elaboration, and public simulation.
- Emits structured evidence and stable candidate/failure hashes.
- Has veto authority over any LLM assessment.

### Judge and Debate panel (staged extensions)

- An LLM Judge is used only to rank near-tied leaves or assess plan/spec
  alignment after the deterministic policy is validated.
- Debate is a temporary panel, not a permanent role called every iteration:
  Spec Advocate, Architecture Critic, and Verification Analyst.
- Debate is triggered only when competing semantic hypotheses cannot be
  distinguished by current EDA evidence.

This role design separates *who proposes code*, *who interprets evidence*, and
*who owns the search topology*. ACE-RTL currently mixes the last two through
independent branch-local Coordinator decisions.

## 4. Search-tree state

The root represents the task specification and target manifest. Plan nodes
represent alternative architectural or semantic hypotheses. Candidate nodes
represent executable RTL bundles.

```text
SearchNode:
  node_id
  parent_id
  node_kind: root | plan | candidate
  depth
  plan_id
  code_bundle_ref
  candidate_hash
  action_from_parent: expand | refine
  evidence:
    stage
    failure_class
    failure_signature
    passed_subtests
    total_subtests
    mismatch_count
    first_mismatch_cycle
    complete_bundle
  status: new | active | pruned | passed | backtracked
  tried_actions
```

The tree, rather than an agent's private chat history, is the source of truth.
Each candidate is immutable. A repair creates a child, so regressions and
oscillations remain visible and backtracking is well-defined.

## 5. Deterministic candidate evaluation

Hard EDA progress is ordered lexicographically:

```text
incomplete bundle
  < compile failure
  < elaboration failure
  < simulation failure
  < public-test pass
```

Within the same stage, prefer:

1. more passed subtests and fewer regressions;
2. fewer mismatches or a later first mismatch;
3. a changed, more localized failure signature;
4. a candidate that adds diversity to the active frontier;
5. a shorter lineage without duplicate code/failure states.

A practical initial score is:

```text
priority(n) = stage_score(n)
            + progress_delta(n, parent)
            + diversity_bonus(n, frontier)
            - regression_penalty(n, parent)
            - repetition_penalty(n, lineage)
```

The terms are ordinal configuration values, not calibrated probabilities. Hard
stage ordering must not be reversed by soft terms. An LLM Judge, when added,
may only break ties inside an EDA-equivalent group.

## 6. Breadth, depth, and backtracking policy

### Choose `REFINE` (depth) when

- the failure is localized to a module, interface, timing rule, or obligation;
- the parent-to-child transition made measurable EDA progress;
- the failure signature or candidate changed, so the branch is not repeating;
- the current plan remains consistent with the specification.

### Choose `EXPAND` (breadth) when

- two or more materially different plan/spec interpretations remain plausible;
- a promising ancestor still has an untried repair hypothesis;
- the current active leaves are structurally similar but an alternative
  architecture has not been tested;
- the task is at the initial high-uncertainty stage.

Expansion must be hypothesis-diverse. Asking two Builders to produce generic
fresh answers is not breadth; it is duplicated sampling.

### Choose `BACKTRACK` when

- a simulation-reaching parent produces a structurally incomplete or
  non-compiling child;
- the same `(candidate_hash, failure_signature)` repeats twice;
- the lineage shows an `A -> B -> A -> B` failure-signature oscillation;
- passed subtests regress for two consecutive descendants;
- all descendants of a plan converge to the same unresolved failure;
- the Analyst identifies a spec/plan error rather than a local RTL bug.

Backtrack destination is explicit:

| Symptom | Destination |
|---|---|
| Immediate structural/compile regression | Parent candidate |
| Local stagnation | Nearest ancestor with a different failure signature |
| Repeated semantic oscillation | Owning plan node |
| Plan/spec contradiction across branches | Root, followed by replanning |

`BACKTRACK` does not generate code. It reactivates the destination and marks the
failed repair hypothesis as tried, preventing the same transition from being
selected again.

### Choose `STOP` when

- a candidate passes the fail-closed public evaluator; or
- all plan nodes and legal repair hypotheses are exhausted.

The hidden evaluator remains report-only and cannot drive search decisions.

## 7. Search procedure

The first controller is best-first search with bounded root diversity, not MCTS.
MCTS would introduce learned value calibration and rollout semantics before we
know whether basic RTL-specific backtracking is useful.

```text
root = create_task_root(spec, manifest)
plans = Planner.propose(root, k=2, require_distinct_hypotheses=True)
frontier = [Builder.expand(plan) for plan in plans]

while frontier is not empty:
    verify every new candidate
    if any candidate passes:
        return STOP(best_passing_candidate)

    leaf = rank_and_select(frontier)
    diagnosis = analyze_if_needed(leaf)

    if should_backtrack(leaf, diagnosis):
        destination = select_backtrack_ancestor(leaf, diagnosis)
        mark_failed_transition_tried(leaf)
        reactivate(destination)
    elif is_localized(diagnosis) and has_new_progress(leaf):
        frontier += Builder.refine(leaf, diagnosis.best_hypothesis)
    elif has_untried_distinct_hypothesis(leaf_or_ancestor):
        frontier += Builder.expand(leaf_or_ancestor)
    else:
        prune(leaf)

return STOP(no_legal_transition)
```

The implementation may set safety caps on nodes and depth to avoid an unbounded
run, but those caps are experiment controls, not inputs to a learned budget
policy.

## 8. cid003 replay hypothesis

The accepted `dynamic_equalizer_0001` ACE trace is the first stress case:

- iteration 1: incomplete bundle -> deterministic structural classification;
- iterations 2--5: functional failures -> depth is reasonable while evidence
  changes and progress remains possible;
- iteration 6: incomplete `error_calc.sv`/top-level bundle after functional
  parents -> immediate backtrack to the last simulation-reaching ancestor;
- iterations 8 and 9: identical bundle hash -> repetition gate prevents another
  equivalent refinement;
- iteration 10: the historical branch passes.

The replay does not claim that the tree would reproduce iteration 10 exactly.
It tests whether the policy makes the locally rational decisions: retain good
ancestors, reject structural regression, detect duplicates, and force a new
hypothesis instead of spending another identical loop.

## 9. Evaluation and ablations

Primary criterion:

```text
hidden functional pass is preserved
```

Search-process criteria:

- generated/evaluated candidate nodes before pass;
- maximum depth and maximum/mean active breadth;
- counts of expand, refine, backtrack, prune, and replan actions;
- duplicate candidate/failure ratio;
- structural regression recovery rate;
- EDA-stage and passed-subtest progression;
- role-call counts.

Tokens, dollars, EDA time, and wall time are recorded as secondary outcomes but
do not affect v0 decisions.

Required comparison:

| Variant | Purpose |
|---|---|
| Static ACE `5 x 30` | incumbent behavior |
| Deterministic tree controller | tests breadth/depth/backtrack mechanism |
| Tree + LLM Judge | tests whether semantic tie-breaking adds value |
| Tree + conflict-triggered Debate | tests whether replanning benefits from roles |

Do not add Judge and Debate before the deterministic variant is replay-tested;
otherwise the source of any improvement cannot be identified.

## 10. Minimal implementation plan

Add a new package under the existing CVDP-native runner:

```text
ace_cvdp_native/rtl_search/
  node.py          # immutable nodes and evidence records
  tree.py          # frontier, lineage, pruning, and ancestor selection
  scoring.py       # deterministic EDA ordering and penalties
  policy.py        # expand/refine/backtrack/stop rules
  roles.py         # Planner, Builder, Analyst interfaces
  replay.py        # convert historical traces into policy transitions
```

Refactor `run_attempt()` into a resumable one-step operation before adding the
live controller. Preserve current behavior behind:

```text
--search-policy static   # existing ACE behavior
--search-policy tree     # new v0 controller
```

First tests:

1. node lineage and immutable parents;
2. hard EDA-stage ordering;
3. parent backtrack after structural regression;
4. duplicate and `A-B-A-B` detection;
5. nearest useful ancestor selection;
6. stop on fail-closed pass;
7. offline replay of `dynamic_equalizer_0001`.

## 11. Advancement rule

Advance to live cid003 testing only if replay demonstrates the expected
backtrack and duplicate-suppression decisions. Advance LLM Judge and Debate only
if the deterministic controller exposes genuine EDA-equivalent choices that it
cannot resolve. Advance heterogeneous budget control only after the tree
controller preserves pass while reducing redundant search states.
