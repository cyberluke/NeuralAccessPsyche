# Qwen3-14B LoRA Lightning runbook

Use `Qwen/Qwen3-14B` as the training parent and never attach 0.6B adapters to it. Run `python -m nram_training doctor`, `inspect-data`, and `smoke` before GPU work. Build with `IMAGE_NAMESPACE=private-user SOURCE_COMMIT=$(git rev-parse HEAD) bash training/buildx.sh`. GPU, Docker, registry, and scientific measurements are **UNAVAILABLE** in this CPU phase. Do not run the causal ablation.
