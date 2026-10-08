# RTL-Director: design recommendation and implementation plan

> **Scope update (2026-08-29):** this document is the deferred, budget-aware
> extension. The primary MVP is now the simpler role-aware breadth/depth/
> backtracking design in [`RTL_SEARCH_TREE_V0.md`](./RTL_SEARCH_TREE_V0.md).
> Budget estimation, aggregation, always-available debate, and dynamic parallel
> width are not required for v0.

> Status: design only; no controller mechanism has been implemented yet  
> Date: 2026-08-29  
> Initial benchmark anchor: CVDP cid003 `cvdp_agentic_dynamic_equalizer_0001`

## 1. Recommendation in one sentence

Refactor ACE-RTL's fixed `5 attempts × 30 iterations` process farm into a
resource-constrained belief-state controller that owns a global candidate pool
and chooses among `BRANCH`, `REFINE`, `DISCUSS`, `VERIFY`, `AGGREGATE`, and
`STOP`; implement the controller as deterministic EDA-first gates plus a sparse
LLM Director/Judge, not as an LLM call after every iteration.

The sparse design is essential. An always-on Planner can easily consume the
tokens saved by pruning Generator and Reflector calls. Cheap structural and EDA
facts should decide obvious transitions. LLM judgment is reserved for cases in
which semantic uncertainty is high enough that its expected value of
information exceeds its cost.

## 2. Why ACE-RTL is the right starting point, and what changes

ACE-RTL already has a strong grounded loop: Generator, Reflector, Coordinator,
simulation feedback, restart, and parallel scaling. Its paper explicitly frames
five-way parallelism as a speed/compute trade-off rather than a lower-compute
method. In the current local CVDP runner:

- five attempt processes are launched immediately;
- each attempt independently runs up to 30 iterations;
- every functional failure normally proceeds through the same Generator →
  Reflector → Coordinator sequence;
- the local Coordinator's action space is effectively `continue` or
  `fresh_start`;
- the default restart guard does not consider restart before 12 failed
  iterations;
- sibling candidates do not share a global value estimate, evidence ledger, or
  budget state.

RTL-Director retains ACE's role implementations but moves ownership of the loop
one level up. The Coordinator remains a within-branch context manager. The new
Director is a cross-branch scheduler and stopping policy.

| Concern | Current ACE-RTL | RTL-Director |
|---|---|---|
| Parallel width | fixed at task start | selected and revised online |
| Iteration depth | fixed cap per branch | granted one action at a time |
| State | private branch history | global candidate/evidence pool |
| Control actions | continue/restart | branch/refine/discuss/verify/aggregate/stop |
| Progress estimate | LLM/history heuristics | EDA-first residual value plus calibrated judge |
| Budget | iteration limits | hard heterogeneous resource vector |
| Spec/plan repair | implicit in evolving context | explicit versioned SpecLedger and PlanGraph |
| Selection | first observed pass | fail-closed pass or best posterior candidate at backstop |

## 3. Formal problem statement

Hidden functional correctness is not directly observable during generation, so
the clean abstraction is a resource-constrained partially observable sequential
decision process, implemented over a belief state.

At decision step `t`:

```text
s_t = (P_t, E_t, S_t, b_t)
```

- `P_t`: global CandidateDAG, including code bundles, parents, spec/plan
  revisions, hashes, and branch lineage;
- `E_t`: EvidenceLedger containing compile, simulation, synthesis, waveform,
  assertion, and regression observations;
- `S_t`: task state: SpecLedger, PlanGraph, target manifest, immutable context,
  and harness-visible constraints;
- `b_t`: remaining heterogeneous budget vector.

Recommended initial budget vector:

```text
b_t = (
  billed_input_tokens,
  billed_output_tokens,
  dollar_cost,
  EDA_seconds,
  wall_seconds,
  evaluator_calls,
  active_parallel_slots
)
```

Cached and reasoning tokens must remain separately observable even when the
provider includes reasoning in output tokens. The hard budget uses the billed
provider definition; reporting preserves all raw fields.

The action is parameterized, not merely categorical:

```text
a_t = (kind, candidate_ids, role, model, verification_level, max_cost)
kind ∈ {BRANCH, REFINE, DISCUSS, VERIFY, AGGREGATE, STOP}
```

The deployment objective is:

```text
maximize   Pr(HiddenCorrect(selected candidate) = 1 | observed evidence)
subject to cumulative_cost_j <= B_j for every resource dimension j
```

For experiments, hidden correctness is measured only by the benchmark evaluator
after delivery. It must never be exposed to the Director, Judge, or debate roles.

## 4. Candidate graph and action semantics

The conceptual mechanism is a Budget-Aware Value Tree. The implementation
should use a DAG because `AGGREGATE` has multiple parents and verification can
attach new evidence without changing code.

### `BRANCH(parent, k, diversity_contract)`

Create `k` sibling implementations or plan/spec hypotheses. Branching is useful
when candidate failure signatures are diverse or the current plan/spec belief is
multimodal. Do not branch merely because budget remains.

### `REFINE(candidate, obligation)`

Run a targeted local repair on one candidate. Refine is appropriate when the
failure maps to one obligation and EDA evidence shows comparable progress.
Repeated identical candidate and failure hashes make this action illegal until
new evidence or a plan/spec revision arrives.

### `DISCUSS(candidate_set, question)`

Run a bounded evidence-grounded debate. Discussion produces structured claims,
counterclaims, and a recommended next action; it does not directly modify RTL.
It is triggered only for plan/spec disagreement, conflicting root causes, or
near-tied candidates where the expected decision improvement exceeds the debate
cost.

Recommended transient roles:

- Spec/Timing Advocate: enumerates mutually exclusive temporal interpretations;
- RTL/Architecture Critic: identifies architectural and implementation risks;
- Verification Analyst: asks which cheapest observation can distinguish them;
- Judge: adjudicates using public spec plus EDA evidence.

Two rounds should be the hard default cap. Debate must cite evidence IDs and is
invalid if it introduces private-test or golden-reference information.

### `VERIFY(candidate, level)`

Use progressive verification rather than repeating the most expensive oracle:

- `V0`: target manifest, extraction, module completeness, lint/safe compile;
- `V1`: public/current directed simulation;
- `V2`: focused waveform or event/dependency query around the first mismatch;
- `V3`: synthesis/equivalence/assertion or additional directed tests when
  available and soundness-checked.

The Director should select the next verification level by expected information
gain per cost. Structural failures never justify an LLM Reflector before V0 is
repaired.

### `AGGREGATE(candidate_set, contract)`

Create a new child that combines complementary modules, obligations, or fixes.
Never overwrite parent candidates. The aggregate must pass V0 and regression
checks before receiving a value above either parent. Whole-file averaging or
free-form synthesis across semantically incompatible branches is disallowed.

### `STOP(candidate, reason)`

Stop immediately on a fail-closed benchmark pass. Otherwise invoke the budget
backstop before no affordable action remains, reserve enough resources for one
final V0/V1 check, and deliver the candidate with the highest calibrated
posterior. Stop reasons are explicit: `verified_pass`, `dominant_candidate`,
`budget_backstop`, `no_positive_value_action`, or `infrastructure_failure`.

## 5. Budget-aware residual value

Do not ask an LLM to assign an unconstrained absolute score. Use a deterministic
base score and ask the Judge only for uncertain residual progress.

For candidate node `n`:

```text
V(n) = V_eda(n) + w_judge * ΔV_judge(n) + w_div * D(n) - R(n)
```

`V_eda` should use monotone or directly observable features:

- complete target bundle and valid top module;
- compilation/elaboration/synthesis stage reached;
- passed subtests and regression count;
- first mismatch moving later;
- mismatch count decreasing;
- number of satisfied spec/plan obligations;
- novelty of failure signature relative to the pool.

`R(n)` penalizes regressions, duplicate hashes, repeated failure signatures,
unsound generated checks, incomplete bundles, and unverified aggregation.

The Judge receives a blinded parent/child or pairwise comparison and returns
strict JSON:

```json
{
  "delta_value": 0.12,
  "confidence": 0.68,
  "evidence_ids": ["sim:iter4:first_mismatch", "plan:P2:obligation7"],
  "failure_class": "functional_local",
  "recommended_actions": ["REFINE", "VERIFY"],
  "disallowed_actions": ["AGGREGATE"]
}
```

Missing evidence IDs, invalid JSON, or disagreement with a hard EDA fact reduces
the judge contribution to zero. Position should be randomized for pairwise
judging, and self-enhancement/verbosity must not affect value.

### Heterogeneous action choice

For each legal action, estimate residual value, information gain, and cost:

```text
Q(a | s_t) = E[ΔV + beta * information_gain]
             - sum_j lambda_j(t) * E[cost_j(a | s_t)]
             - uncertainty_risk(a)
```

Use resource-specific shadow prices instead of collapsing all resources through
only the minimum remaining ratio:

```text
r_j(t)      = remaining_j / initial_j
lambda_j(t) = lambda_j0 / (r_j(t) + epsilon)^p
```

As any resource becomes scarce, its actions become more expensive. Early in a
run, sample legal actions from a softened `Q` distribution to preserve
exploration. Near exhaustion, select greedily and reserve the delivery backstop.
Action costs begin with role/level priors and are updated online by an EWMA over
actual provider usage and EDA latency.

This extends the scalar/uniform-tool assumption in BAVT to asymmetric RTL
actions: a manifest check, DeepSeek generation, debate, simulation, waveform
query, and synthesis have different cost vectors and information yield.

## 6. Two-tier Director policy

### Tier 0: deterministic hard gates

Run on every event without an LLM call:

1. API/infra failure → retry under an infra budget; do not count as RTL failure.
2. Missing/truncated target file → restore last complete bundle or deterministic
   manifest repair; do not call a semantic Reflector.
3. Exact candidate+failure duplicate twice → prohibit `REFINE` on that state.
4. ABAB candidate oscillation → branch or discuss the disputed assumption.
5. Hard EDA pass → `STOP` and cancel unsent work.
6. No affordable action after reserving final verification → budget backstop.

### Tier 1: sparse Director/Judge

Invoke only at event boundaries:

- initial width/model/role allocation;
- two or more competitive branches after fresh evidence;
- plan/spec conflict or a value tie;
- choice between additional generation and stronger verification;
- potential aggregate;
- pre-backstop delivery when no candidate passes.

The Director does not emit code. It emits one legal action with a maximum cost
reservation. The runtime validates legality and budget before dispatch.

## 7. cid003 anchor: `dynamic_equalizer_0001`

This is the best first pressure test because it passed, but only at the tenth
winner iteration and it mixes genuine fixed-point reasoning with avoidable
structural failures.

Observed winner trace from the accepted ACE cid003 run:

| Iteration | Observation |
|---:|---|
| 1 | incomplete `dynamic_equalizer.sv`; local preflight rejection |
| 2–5 | complete bundle, functional failures while fixed-point/LMS semantics evolve |
| 6 | `error_calc.sv` omitted and top truncated; structural regression |
| 7–9 | functional failures; iterations 8 and 9 have the same bundle hash |
| 10 | complete bundle passes |

Exact provider usage is recoverable for the winner's ten Generator calls from
the nested DeepSeek response payload:

| Metric | Winner branch |
|---|---:|
| Prompt tokens | 169,318 |
| Completion tokens | 156,534 |
| Reasoning tokens (included in completion) | 133,998 |
| Cached prompt tokens | 45,440 |
| Total Generator tokens | **325,852** |

The run also contains nine Reflector calls, but their provider usage was not
persisted in the per-iteration JSON. The four sibling attempts were deleted or
stopped after first pass, so total billed task cost cannot be reconstructed.
This means the first live comparison must rerun the static ACE policy with
complete telemetry; the historical 325,852 is a winner-branch lower bound, not
the task total.

Expected Director behavior on this trace:

- iter1: classify structural; repair/continue without semantic debate;
- iter2–5: permit targeted refinements only while evidence changes;
- iter6: restore the last complete bundle and target the missing file rather
  than reflecting on algorithm semantics;
- iter8/9 duplicate: block another identical refine; request a cheap
  discriminating verification query or branch the unresolved fixed-point plan;
- keep a long-tail reserve because the branch has previously shown functional
  progress; do not globally cap all tasks at four iterations.

## 8. Implementation sequence

### Phase 0 — freeze and measure

1. Keep ACE-RTL read-only; all work occurs in BW-RTL.
2. Hoist provider `usage` from nested adapter payloads for every role.
3. Record sent/completed/cancelled/billed status, retries, timestamps, model,
   branch, action, prompt/cached/reasoning/completion tokens, dollars, and EDA
   seconds in one append-only event log.
4. Preserve sibling traces after success; mark cancellation instead of deleting.

Exit gate: static ACE replay produces a complete cost ledger whose totals equal
the sum of role calls.

### Phase 1 — resumable step API and shadow Director

Refactor the current `run_attempt()` internal loop into a resumable
`run_candidate_step(state, action) -> observation`. Run the existing static
policy while the Director observes and logs the action it would have selected.

Exit gate: shadow mode reproduces current functional behavior, and every state
transition can be replayed offline.

### Phase 2 — global pool plus deterministic controller

Implement CandidateDAG, EvidenceLedger, ResourceLedger, V0/V1 verification,
deduplication, structural routing, adaptive width, and hard STOP. Limit the
initial action set to `BRANCH`, `REFINE`, `VERIFY`, and `STOP`.

Exit gate: mock tests prove no hard-budget overspend; historical trace replay
correctly recognizes structural regressions and duplicates.

### Phase 3 — residual Judge and budget-aware value policy

Add blinded pairwise residual judging, action cost prediction, vector shadow
prices, and exploration-to-exploitation scheduling. Charge Judge cost to the
same ledger.

Exit gate: on replay, Judge calls occur only at configured ambiguity events and
cannot override hard EDA facts.

### Phase 4 — bounded debate and aggregation

Add `DISCUSS` for spec/plan conflicts and `AGGREGATE` for complementary verified
parents. Both are disabled by default and enabled through value-of-information
gates.

Exit gate: aggregation never mutates parents and must pass V0 plus regression;
debate terminates after two rounds.

### Phase 5 — one-sample live experiment, then cid003 slice

Run the paired `dynamic_equalizer_0001` experiment first. Only after it passes
the efficiency gate, expand to a small cid003 panel representing easy pass,
local repair, multi-file planning, and spec ambiguity, then the full 34 tasks.

## 9. Code-level touchpoints

Recommended new package:

```text
skills/ace-rtl/scripts/rtl_director/
  actions.py          # typed action schema and legality
  state.py            # candidate/spec/plan/evidence state
  budget.py           # vector resource ledger and reservations
  value.py            # EDA base value and residual composition
  policy.py           # shadow prices and action selection
  director.py         # event loop
  judge.py            # structured pairwise judge
  debate.py           # bounded evidence-grounded discussion
  aggregate.py        # provenance-preserving code aggregation
  telemetry.py        # provider/EDA cost events
  replay.py           # offline trace replay
```

Required modifications:

- `ace_cvdp_native/cli.py`: replace eager five-process launch with Director-owned
  worker admission and resumable candidate steps;
- `ace_cvdp_native/llm.py`: return role-call metadata and usage, not content only;
- `llm_call.py`: support `planner`, `judge`, `debater`, and `aggregator` roles;
- `deepseek_adapter.py`: expose normalized usage while retaining raw provider
  payload;
- `agent_types.py`: add candidate, evidence, action, and resource dataclasses;
- reporting: emit `events.jsonl`, `candidate_dag.json`, `decisions.jsonl`, and a
  cost-normalized summary.

Do not begin by rewriting the legacy `agent_loop.py`. The CVDP native runner is
the accepted working path and gives the cheapest controlled validation surface.

## 10. First experiment contract

### Offline test (zero paid generation)

Replay the historical `dynamic_equalizer_0001` events and check:

- structural failures route to deterministic repair;
- duplicate bundle/failure states cannot receive another identical REFINE;
- every action respects vector budget reservations;
- the policy keeps a long-tail reserve instead of stopping solely at depth four;
- Judge/debate call count is sparse and fully charged.

### Live paired test

Use the same model endpoint, prompt visibility, adapter, evaluator, temperature,
and target task for both arms:

1. static ACE policy with full telemetry;
2. RTL-Director policy under the static run's measured billed-cost envelope.

Primary success criterion:

```text
Hidden pass = 1 AND total billed tokens <= 50% of the paired static ACE run
```

Because a single stochastic run is only a development gate, follow with at least
three paired repeats. Report success rate and bootstrap uncertainty rather than
selecting the cheapest successful seed.

Secondary metrics:

- tokens/dollars/EDA-seconds/wall-time to first hidden pass;
- role- and action-level cost share;
- peak concurrency and tokens spent by cancelled siblings;
- candidate and failure-signature duplication rate;
- Judge calibration against subsequent EDA progress;
- number of plan/spec revisions and verification escalations.

Anti-win conditions:

- token reduction caused by failure or premature stop;
- use of private tests, golden RTL, or hidden expected values;
- a Judge score improvement without EDA/hidden-correctness improvement;
- excluding retries, failed calls, reasoning tokens, or cancelled billed calls;
- comparing different base models or evaluator protocols.

## 11. Main risks and mitigations

| Risk | Mitigation |
|---|---|
| LLM Judge overconfidence/bias | relative pairwise score, blinded ordering, evidence IDs, EDA veto |
| Planner overhead erases savings | deterministic Tier 0 and sparse event-triggered Tier 1 |
| Debate becomes fixed extra cost | value-of-information gate and two-round cap |
| Aggregation causes regressions | immutable parents, new DAG child, mandatory V0/regression |
| Hidden correctness is unobservable | optimize calibrated belief; measure hidden pass only offline |
| Static ACE baseline cost is incomplete | telemetry-first paired rerun before efficiency claim |
| Parallel cancellation still bills tokens | record request-send and provider completion; charge billed calls |
| Hard task needs long depth | task-level reserve based on observed progress, not uniform depth cap |
| BAVT guarantee does not transfer | make no convergence claim under stochastic RTL/EDA dynamics |

## 12. Related-work boundary

- ACE-RTL supplies grounded Generator/Reflector/Coordinator iteration and static
  parallel scaling; RTL-Director changes the global scheduling problem.
- BAVT supplies step-level residual value and budget-conditioned exploration;
  its paper explicitly leaves heterogeneous tools and asymmetric costs as future
  work. RTL actions make that limitation central.
- Snell et al. establish that the best parallel/sequential compute split varies
  with difficulty; RTL-Director estimates this online from actual EDA progress
  instead of thousands of oracle-labeled samples.
- Tree of Thoughts and multi-agent debate motivate branch search and discussion,
  but neither gives an RTL-specific hard-budget/evidence policy.
- LLM-as-a-Judge work documents position, verbosity, and self-enhancement bias;
  hence the Judge is residual and subordinate to EDA evidence.
- VerilogCoder supplies explicit RTL planning and waveform tracing; SeqFeed
  strengthens the case for event-addressable, dependency-traceable verification
  observations. These should inform `VERIFY(V2)`, not become unconditional cost.
- MAGE establishes specialized generation/test/judge/debug roles, but its role
  workflow is not an online heterogeneous-budget scheduler.

## References

1. Deng et al., [ACE-RTL: When Agentic Context Evolution Meets RTL-Specialized LLMs](https://arxiv.org/abs/2602.10218), 2026.
2. Li et al., [Spend Less, Reason Better: Budget-Aware Value Tree Search for LLM Agents](https://arxiv.org/abs/2603.12634), 2026.
3. Snell et al., [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314), 2024.
4. Yao et al., [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601), 2023.
5. Du et al., [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325), 2023.
6. Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685), 2023.
7. Zhao et al., [MAGE: A Multi-Agent Engine for Automated RTL Code Generation](https://arxiv.org/abs/2412.07822), 2024/2025.
8. Ho et al., [VerilogCoder: Autonomous Verilog Coding Agents with Graph-based Planning and AST-based Waveform Tracing](https://arxiv.org/abs/2408.08927), 2024/2025.
9. Du et al., [SeqFeed: Improving Agentic RTL Code Generation with Sequential Behavior Feedback](https://arxiv.org/abs/2608.16934), 2026.
10. Pinckney et al., [Comprehensive Verilog Design Problems](https://arxiv.org/abs/2506.14074), 2025.
