---
name: bw-rtl
description: Coordinate adaptive multi-agent RTL generation as an evidence-grounded, round-based search. Use when planning, implementing, debugging, verifying, comparing, backtracking, fusing, or stopping RTL candidate branches under a shared public specification and EDA feedback.
---

# BW-RTL

Use a global search state and an LLM Director to decide search breadth, depth,
rollback, and information sharing.  Do not use a fixed branch count or a
benchmark-specific repair recipe.

## Load the role contract

Read the reference matching each active role before invoking it:

- Director: `references/director.md`
- Planner: `references/planner.md`
- Builder: `references/builder.md`
- Analyst: `references/analyst.md`
- Debater: `references/debater.md`
- Verifier boundary: `references/verifier.md`
- Shared claims: `references/knowledge-ledger.md`

The role references are instructions, not task facts.  The public task
specification, immutable public context, candidate deltas, EDA evidence, and
current ledger are supplied separately at runtime.

## Execute one search round

1. Present the Director with the complete active frontier, newly completed
   branch reviews, accepted/proposed/conflicting knowledge, and prior actions.
2. Let the Director return any finite number of legal action cards.  Dispatch
   length is the Director's decision; the runtime worker count is only a
   physical concurrency limit.
3. Execute independent cards concurrently.  Let role judgement determine what
   to investigate or change.  Use deterministic code only to validate schemas,
   references, immutable contracts, state transitions, and EDA results.
   If one card produces fail-closed passing evidence, cancel cards that have not
   started and skip new losing-sibling diagnosis, then return all completed
   evidence to the Director for the final delivery decision.
4. Verify every complete candidate.  The Verifier reports evidence and never
   edits RTL.  The Analyst reviews each branch independently and proposes
   evidence-linked knowledge; it never edits RTL or chooses global scheduling.
5. Merge every branch outcome into one round summary.  Validate evidence
   references, record conflicts explicitly, and call the Director again.
6. Stop only when a candidate has fail-closed passing evidence and the Director
   selects it, or when the Director gives a reasoned exhausted/infrastructure
   stop with no correctness claim.

## Preserve authority boundaries

- The Director controls topology and dispatch, not RTL contents.
- The Planner creates or revises implementation hypotheses, not RTL.
- The Builder alone writes target RTL.
- The Verifier alone turns tool output into pass/fail evidence.
- The Analyst diagnoses branch evidence and proposes shared claims, not code.
- The Debater compares named branches and exposes disagreements; the Director
  remains the sole global judge and scheduler.
- Deterministic runtime code may reject an illegal decision but must not replace
  it with a heuristic semantic choice.

## Guard against overfitting and reward hacking

- Never reveal golden RTL, hidden tests, hidden correctness, or benchmark
  labels to any role.
- Never encode task IDs, known winning patches, or trace-specific fixes in this
  skill.
- Treat public-test pass as verifier evidence, not proof of hidden correctness.
- Attach every promoted fact to observable evidence and retain refutations and
  conflicts rather than silently overwriting them.
- Validate mechanism behavior with deterministic mocks first; make
  effectiveness claims only from held-out tasks under unchanged evaluators.
