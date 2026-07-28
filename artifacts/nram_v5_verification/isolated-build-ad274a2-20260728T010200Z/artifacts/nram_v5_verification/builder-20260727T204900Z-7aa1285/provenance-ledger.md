# Provenance ledger

## Starting point

- Branch: `main`.
- Commit: `7aa12854800327ea905e608b9ececc3e3f5fd758`.
- Pre-existing tracked modifications: `Dockerfile`, `api/routes.py`,
  `compose.yaml`, `core/engines/sglang_engine.py`,
  `core/steering/nram_logit_processor.py`,
  `core/steering/serialization.py`, `inference/sglang/Dockerfile`,
  `inference/sglang/entrypoint.sh`, `section_by_section_ab_test.py`, and
  `tests/adversarial/test_req_injection.py`.
- The complete 142-path pre-existing inventory is retained in the Phase-0
  `pre-existing-worktree.txt` artifact.

## Builder edits to pre-existing paths

The Builder modified the NRAM-related pre-existing runtime paths listed above
except `section_by_section_ab_test.py`. The resulting diffs on those paths
therefore combine user/pre-task content with this remediation; line-level
authorship cannot be inferred from the final Git diff alone.

## Builder-created implementation and evidence

- Required five NRAM v5 documents and claim-correction banners.
- Shared processor telemetry/control changes, API/runtime contract changes,
  security-correct tests, GPU tests, paired harness, artifact collector, and
  versioned builder evidence.
- Test/build metadata and minimal Node/Streamlit gates.

## Phase-0-created inputs retained

The canonical specification, Phase-0 intake, and
`phase0-20260727T180644Z-7aa1285/` artifacts were created before this Builder
pass and are retained unchanged as provenance inputs.

## Test or concurrent side effects excluded from commit scope

`ab_test_outputs/01_normal_state.md`, `02_microdose_state.md`, and
`03_threshold_state.md` became modified after the fresh Builder baseline. They
are generated A/B corpora and are excluded. `ab_section_outputs/` and
`section_by_section_ab_test.py` were already dirty and are also excluded.
No reset, clean, stash, or deletion was used.

## Generated dependency/build state

`node_modules/` was created by `npm ci` and is ignored, not committed.
`package-lock.json` was updated by the bounded audit repair from vulnerable
`form-data` to the audited dependency graph. Scientific artifacts under this
builder run directory are intended evidence, not source inputs.
