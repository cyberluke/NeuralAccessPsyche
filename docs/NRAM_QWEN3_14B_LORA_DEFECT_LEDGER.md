# Defect ledger

| ID | Finding | Status |
|---|---|---|
| D-001 | GPU/Docker/SGLang runtime gates were not available in the initial CPU phase | RESOLVED by local Docker RTX evidence; retained as historical provenance |
| D-002 | Training base and private registry digests require a real build/push | OPEN; local image is proven, registry digest remains unavailable |
| D-003 | Gradient evidence was sampled after Trainer cleared gradients | RESOLVED in `nram_training/trainer.py`; callback captures the pre-optimizer-step tensors |
| D-004 | Repeated SGLang LoRA flags overwrote prior values in direct launch | RESOLVED; Compose uses one flag followed by the complete target/path lists |
