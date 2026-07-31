# Qwen3-14B LoRA traceability

| Requirement | Implementation | Evidence |
|---|---|---|
| Shared CLI | `nram_training/cli.py` | doctor, inspect-data, smoke, train, validate, package |
| 14B parent guard | `nram_training/trainer.py` | rejects 0.6B |
| Existing intent | `nram_training/config.py` | r16/alpha32/.1, seven projections, civil-comments thresholds |
| Synthetic fixtures | `nram_training/fixtures.py` | TEST_ONLY markers |
| Runtime overlay | `compose.lora.yaml` | SGLang LoRA contract flags |
