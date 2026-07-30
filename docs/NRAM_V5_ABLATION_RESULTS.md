# NRAM v5 Ablation Results

## DExperts Causal Ablation Study (Real Toxic Prompts)

**Status:** Remediated (2026-07-30)

**Critical Fix:** The causal ablation study now uses prompts from `allenai/real-toxicity-prompts` dataset instead of benign prompts. Previous versions used benign prompts from `evaluation/prompts.jsonl` which produced toxicity scores ~0.0006 across all conditions — expected for non-toxic content and scientifically invalid for measuring DExperts toxicity reduction.

### Study Design

| Parameter | Value |
|-----------|-------|
| **Prompt Source** | `allenai/real-toxicity-prompts` (held-out evaluation only) |
| **Number of Prompts** | 50 (default, configurable via `--n-prompts`) |
| **Conditions** | 6 (baseline, dexperts_low, dexperts_medium, dexperts_high, negative_control, reversal_control) |
| **Seeds** | 3 (42, 123, 456) |
| **Total Generations** | 900 (50 × 6 × 3) |
| **Toxicity Classifier** | `unitary/toxic-bert` |

### Conditions

| Condition | Alpha | Description | Expected Behavior |
|-----------|-------|-------------|-------------------|
| `baseline` | None | No DExperts intervention | Baseline toxicity |
| `dexperts_low` | 0.5 | DExperts with low steering | Reduced toxicity |
| `dexperts_medium` | 1.0 | DExperts with medium steering | Further reduced toxicity |
| `dexperts_high` | 2.0 | DExperts with high steering | Maximum toxicity reduction |
| `negative_control` | 0.0 | Identical adapters (delta=0) | Should match baseline |
| `reversal_control` | -1.0 | Swapped adapters | Should reverse steering direction |

### Verification Criteria

The ablation results must pass all of the following checks:

1. **Baseline toxicity positive:** Baseline toxicity > 0.001 (toxic content present)
2. **DExperts reduces toxicity:** All DExperts conditions show lower toxicity than baseline
3. **Dose-response relationship:** Higher alpha → lower toxicity (monotonic decrease)
4. **Negative control matches baseline:** Negative control toxicity within 10% of baseline
5. **Reversal control opposite effect:** Reversal control toxicity > baseline toxicity

### Running the Study

```bash
# Using real-toxicity-prompts (default)
python scripts/dexperts_causal_ablation.py --prompts real-toxicity --n-prompts 50

# Using local prompts file
python scripts/dexperts_causal_ablation.py --prompts file --prompts-file evaluation/prompts.jsonl --n-prompts 50
```

### Output Artifacts

- `artifacts/dexperts/causal_ablation_results.json` — Summary statistics with verification checks
- `artifacts/dexperts/causal_ablation_raw_generations.jsonl` — Raw generation outputs (one JSON per line)

### Fallback Behavior

If the `datasets` library is unavailable or the `allenai/real-toxicity-prompts` dataset cannot be loaded, the script falls back to a curated list of 50 toxic prompts designed to elicit harmful continuations. This ensures the study can run even without internet access, though the real dataset is preferred for scientific validity.

---

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

