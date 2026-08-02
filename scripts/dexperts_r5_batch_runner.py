#!/usr/bin/env python3
"""Length-bucketed, bounded-batch runner for the post-gate R5 smoke path.

The runner sends prompt arrays to the OpenAI-compatible SGLang endpoint so
SGLang can continuous-batch generation requests.  Teacher-forced mechanistic
prefill is a separate contract: it must return only target-token deltas and
scalar KL/JS/TV values.  Full-vocabulary logits are rejected and never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def length_buckets(rows: List[Dict[str, Any]], bucket_size: int, token_budget: int) -> Iterable[List[Dict[str, Any]]]:
    """Sort by source token count and yield bounded microbatches."""
    ordered = sorted(rows, key=lambda row: (int(row["source_tokens"]), row["sequence_id"]))
    batch: List[Dict[str, Any]] = []
    tokens = 0
    for row in ordered:
        row_tokens = int(row["source_tokens"])
        if batch and (len(batch) >= bucket_size or tokens + row_tokens > token_budget):
            yield batch
            batch, tokens = [], 0
        batch.append(row)
        tokens += row_tokens
    if batch:
        yield batch


def _request(endpoint: str, api_key: str, batch: List[Dict[str, Any]], seed: int, max_tokens: int) -> Dict[str, Any]:
    import httpx
    prompts = [row["prompt"] for row in batch]
    body = {
        "model": batch[0].get("model", "nram-qwen3-14b-awq"),
        "prompt": prompts,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "seed": seed,
        "nram": {"enabled": True, "profile": "normal"},
    }
    started = time.perf_counter()
    response = httpx.post(endpoint, json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=180.0)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data.get("choices"), list) or len(data["choices"]) != len(batch):
        raise RuntimeError("batch response must contain one choice per input prompt")
    if any("logits" in item or "logprobs" in item for item in data.get("telemetry", {}).values() if isinstance(data.get("telemetry"), Mapping)):
        raise RuntimeError("full-vocabulary logits/logprobs are forbidden in persisted telemetry")
    return {"latency_ms": (time.perf_counter() - started) * 1000, "effective_batch_size": len(batch), "response": data}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = [json.loads(line) for line in Path(args.sequences).read_text().splitlines() if line.strip()]
    required = {"sequence_id", "prompt", "source_tokens"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"sequence file must contain {sorted(required)}")
    retries = 0
    batches = []
    for batch in length_buckets(rows, args.source_microbatch_size, args.token_budget):
        for attempt in range(args.max_retries + 1):
            try:
                result = _request(args.endpoint, args.api_key, batch, args.seed, args.max_tokens)
                batches.append({"sequence_ids": [x["sequence_id"] for x in batch], **{k: v for k, v in result.items() if k != "response"}})
                break
            except Exception:
                if attempt >= args.max_retries:
                    raise
                retries += 1
    total_ms = sum(item["latency_ms"] for item in batches)
    output = {
        "schema_version": "r5-batch-v1",
        "mode": "generation_continuous_batching",
        "source_microbatch_size": args.source_microbatch_size,
        "token_budget": args.token_budget,
        "n_sequences": len(rows),
        "effective_batch_size": max((x["effective_batch_size"] for x in batches), default=0),
        "retry_count": retries,
        "throughput_sequences_per_second": len(rows) / (total_ms / 1000) if total_ms else 0.0,
        "peak_vram_mb": None,
        "peak_vram_status": "server telemetry required",
        "batches": batches,
        "full_vocabulary_logits_persisted": False,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key", default="dev-nram-key")
    parser.add_argument("--source-microbatch-size", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--autotune", action="store_true", help="Probe effective batch sizes 8, 16 and 32")
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.autotune:
        runs = []
        for size in (8, 16, 32):
            args.source_microbatch_size = size
            runs.append(run(args))
        print(json.dumps({"autotune": runs}, indent=2))
    else:
        print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
