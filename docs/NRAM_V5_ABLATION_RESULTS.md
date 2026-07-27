# NRAM v5 Ablation Results

## Bounded runtime proofs

The bounded pass produced real RTX 4090 public-API proofs for the following
intermediate effects:

| Proof | Conditions | Direct result | Text-level result | Scope |
|---|---|---|---|---|
| Forced token | same prompt/seed; force disabled/enabled | token 11064 became sole finite sampled candidate; mask count 151,935 | enabled decoded `proof`; disabled differed | one prompt/seed/token |
| Phrase masks | English and Czech phrase schedules; disabled/enabled | completion ID changed to `-inf`; scheduled force rejected | disabled emitted phrase; enabled did not | two tokenizer variants |
| Source blocker | exact first source 8-gram scheduled; disabled/enabled | eighth continuation ID masked before sampling | enabled differed from exact disabled continuation | one source fixture |
| Streaming | streamed phrase schedule | generated-history count advanced per event; final ID masked | SSE chunks completed | one stream fixture |
| Grammar composition | JSON schema + phrase schedule | completion ID masked while grammar active | response remained parseable JSON without phrase | one schema fixture |
| PID entropy | low/medium/high targets | actual entropy moved toward each target; P/I/D/clamp logged | text is not used as causal evidence | one prompt/seed per target + reset |
| Capsule/soft/vector | enabled local controls | exact local logits/ranks/probabilities changed; window/sign/zero controls held | text is not used as causal evidence | one prompt/seed |

## Scientific study status

The canonical full-stack ablation is not complete. Representation-only,
closed-loop-only, tournament-only, DExperts, and all leave-one-out conditions
depending on those mechanisms are `NOT_IMPLEMENTED` or `BLOCKED`. No human
labels exist. The bounded paired harness records only supported conditions and
retains unsupported condition rows without invented outputs.

No confidence intervals or significance tests are claimed for the direct proof
fixtures because their `n` is too small and they are deterministic mechanism
tests, not a population study. Negative results and failed attempts are retained
under the versioned artifact directory.

The bounded paired text run executed 28 rows: two prompts, two seeds, and seven
supported conditions (`n=4` per condition). All seven conditions produced the
same aggregate text metrics: mean 8.5 output words, mean lexical diversity 0.95,
and zero exact source 8-gram overlaps. This is a retained negative text-effect
result. Eight unsupported canonical conditions were emitted as explicit
`NOT_IMPLEMENTED`/`BLOCKED` rows. No CI or hypothesis test was calculated.

Scientific verdict: `PARTIALLY FUNCTIONAL`.

