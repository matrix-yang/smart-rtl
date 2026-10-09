# Verifier boundary

The Verifier is an EDA execution boundary, not a generative agent.  It receives
an immutable candidate and the public evaluator configuration, runs the declared
compile/elaboration/simulation checks, and emits normalized evidence.

It must never modify RTL, invent missing results, consult golden RTL, or expose
hidden test contents.  Infrastructure failures are distinct from candidate
failures.  A success claim requires a complete, fail-closed pass under the
configured evaluator.  Additional probes may use only public tests or generated
non-golden checks and must be labeled with their soundness status.
