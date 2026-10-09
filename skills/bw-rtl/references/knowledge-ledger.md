# Knowledge Ledger

The ledger is shared semantic memory, not a transcript.  Store concise claims
that can affect future decisions.

## Item types

- `spec_constraint`: required public behavior or interface property.
- `observed_fact`: direct compile, elaboration, simulation, or candidate fact.
- `root_cause`: inferred explanation for evidence.
- `fix_pattern`: change supported by an observed improvement.
- `rejected_hypothesis`: explanation contradicted by evidence.
- `conflict`: mutually incompatible supported/proposed claims.
- `reusable_artifact`: plan, assertion, probe, or code property worth reusing.
- `open_question`: unresolved uncertainty with a proposed discriminator.

## Scope and status

Scope is one of `global`, `plan_family`, `module`, or `candidate_lineage`.
Status is one of `proposed`, `supported`, `refuted`, `conflicting`, or
`superseded`.

Every item needs provenance and evidence references.  Deterministic validation
checks that references exist and deduplicates exact normalized claims.  It does
not decide whether a semantic inference is true.  Only a Director decision may
promote, refute, or supersede a proposed inference.  Conflicting claims remain
visible until a discriminating observation resolves them.
