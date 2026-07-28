"""Preregistered bounded NRAM v5 paired live-runtime harness.

This runner executes only conditions that exist on the public API. Unsupported
canonical conditions are emitted as status rows and never receive fabricated
outputs or metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


PROMPTS = [
    {"id": "en-source", "language": "en", "category": "source_transformation", "prompt": "Paraphrase without copying: A transparent mechanism turns evidence into a concrete product decision.", "source": "A transparent mechanism turns evidence into a concrete product decision."},
    {"id": "cs-source", "language": "cs", "category": "source_transformation", "prompt": "Přeformuluj bez kopírování: Transparentní mechanismus převádí důkazy na konkrétní produktové rozhodnutí.", "source": "Transparentní mechanismus převádí důkazy na konkrétní produktové rozhodnutí."},
    {"id": "grounded", "language": "en", "category": "grounded_synthesis", "prompt": "State one mechanism supported by the supplied sentence and do not add facts.", "source": "The controller masks a token before sampling."},
    {"id": "product", "language": "en", "category": "product_vision", "prompt": "Propose a concise product mechanism without clichés.", "source": ""},
    {"id": "conflict", "language": "en", "category": "conflicting_evidence", "prompt": "Evidence A says latency fell; evidence B says it rose. Report the conflict only.", "source": "Evidence A says latency fell. Evidence B says latency rose."},
    {"id": "no-intervention", "language": "en", "category": "no_intervention", "prompt": "Return a concise neutral definition of deterministic decoding.", "source": ""},
]

SUPPORTED_CONDITIONS: dict[str, dict[str, Any] | None] = {
    "vanilla": None,
    "developer_prompt_only": {"prompt_steering_enabled": True, "profile_logit_steering_enabled": False},
    "structural_only": {"prompt_steering_enabled": False, "profile_logit_steering_enabled": False, "forbidden_phrases": ["as an AI", "jako AI"]},
    "logit_only": {"prompt_steering_enabled": False, "profile_logit_steering_enabled": False, "entropy_control": True, "entropy_phase_targets": {"extraction": 3.0, "questioning": 3.0, "divergence": 3.0, "synthesis": 3.0, "formulation": 3.0}},
    "full_supported": {"prompt_steering_enabled": True, "profile_logit_steering_enabled": True, "forbidden_phrases": ["as an AI", "jako AI"], "entropy_control": True},
    "full_minus_structural": {"prompt_steering_enabled": True, "profile_logit_steering_enabled": True, "entropy_control": True},
    "full_minus_logit": {"prompt_steering_enabled": True, "profile_logit_steering_enabled": False, "forbidden_phrases": ["as an AI", "jako AI"]},
}

UNSUPPORTED_CONDITIONS = {
    "representation_only": "NOT_IMPLEMENTED",
    "closed_loop_only": "NOT_IMPLEMENTED",
    "tournament_only": "NOT_IMPLEMENTED",
    "dexperts_only": "BLOCKED_NO_EXPERT_ARTIFACTS",
    "full_canonical_stack": "BLOCKED_UNIMPLEMENTED_COMPONENTS",
    "full_minus_representation": "BLOCKED_UNIMPLEMENTED_COMPONENTS",
    "full_minus_closed_loop": "BLOCKED_UNIMPLEMENTED_COMPONENTS",
    "full_minus_tournament": "BLOCKED_UNIMPLEMENTED_COMPONENTS",
}


def _words(text: str) -> list[str]:
    return [word.strip(".,:;!?()[]{}\"'").lower() for word in text.split() if word.strip()]


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i+n]) for i in range(max(0, len(words) - n + 1))}


def metrics(output: str, source: str) -> dict[str, Any]:
    words = _words(output)
    source_words = _words(source)
    overlap = _ngrams(words, 8) & _ngrams(source_words, 8)
    return {
        "output_words": len(words),
        "lexical_diversity": (len(set(words)) / len(words)) if words else 0.0,
        "exact_source_8gram_overlap": len(overlap),
        "forbidden_frame_count": sum(output.lower().count(x) for x in ("as an ai", "jako ai")),
        "adjacent_repetition_count": sum(1 for a, b in zip(words, words[1:]) if a == b),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--seeds", default="101,202")
    parser.add_argument("--max-tokens", type=int, default=24)
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir)
    raw_dir = artifact_dir / "raw-generations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]
    prompts = PROMPTS[: args.max_prompts]

    preregistration = {
        "created_before_runs": True,
        "model": "nram-qwen3-14b-awq",
        "baseline_model": "qwen3-14b-awq-baseline",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "seeds": seeds,
        "prompt_suite": PROMPTS,
        "executed_prompt_ids": [item["id"] for item in prompts],
        "supported_conditions": SUPPORTED_CONDITIONS,
        "unsupported_conditions": UNSUPPORTED_CONDITIONS,
        "statistics": "descriptive only; bounded n is insufficient for inferential claims",
    }
    (artifact_dir / "preregistration.json").write_text(json.dumps(preregistration, indent=2, ensure_ascii=False), encoding="utf-8")

    api_url = os.getenv("NRAM_TEST_API_URL", "http://127.0.0.1:8000/v1/chat/completions")
    headers = {"Authorization": f"Bearer {os.getenv('NRAM_TEST_API_KEY', 'dev-nram-key')}"}
    rows = []
    with httpx.Client(timeout=180.0) as client:
        for prompt in prompts:
            for seed in seeds:
                for condition, nram_delta in SUPPORTED_CONDITIONS.items():
                    request_id = f"paired-{prompt['id']}-{seed}-{condition}"
                    body: dict[str, Any] = {
                        "model": "qwen3-14b-awq-baseline" if nram_delta is None else "nram-qwen3-14b-awq",
                        "messages": [{"role": "user", "content": prompt["prompt"]}],
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": args.max_tokens,
                        "seed": seed,
                        "tools": [],
                    }
                    if nram_delta is not None:
                        nram = {"enabled": True, "profile": "normal", "request_id": request_id, "include_telemetry": True, **nram_delta}
                        if condition in {"structural_only", "full_supported", "full_minus_logit"} and prompt["source"]:
                            nram.update({"source_text": prompt["source"], "source_ngram_size": 8})
                        body["nram"] = nram
                    started = time.perf_counter()
                    response = client.post(api_url, headers=headers, json=body)
                    latency_ms = (time.perf_counter() - started) * 1000
                    response.raise_for_status()
                    payload = response.json()
                    output = payload["choices"][0]["message"]["content"]
                    raw = {
                        "prompt": prompt,
                        "seed": seed,
                        "condition": condition,
                        "request": body,
                        "response": payload,
                        "latency_ms": latency_ms,
                    }
                    raw_path = raw_dir / f"{request_id}.json"
                    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
                    config_hash = (payload.get("nram_correlation") or {}).get("config_hash", "baseline")
                    rows.append({
                        "status": "EXECUTED",
                        "prompt_id": prompt["id"],
                        "language": prompt["language"],
                        "category": prompt["category"],
                        "seed": seed,
                        "condition": condition,
                        "request_id": request_id,
                        "config_hash": config_hash,
                        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                        "raw_path": raw_path.as_posix(),
                        "latency_ms": round(latency_ms, 3),
                        **metrics(output, prompt["source"]),
                    })

    for condition, status in UNSUPPORTED_CONDITIONS.items():
        rows.append({"status": status, "prompt_id": "", "language": "", "category": "", "seed": "", "condition": condition, "request_id": "", "config_hash": "", "output_sha256": "", "raw_path": "", "latency_ms": "", "output_words": "", "lexical_diversity": "", "exact_source_8gram_overlap": "", "forbidden_frame_count": "", "adjacent_repetition_count": ""})

    with (artifact_dir / "generation-metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"executed": sum(r["status"] == "EXECUTED" for r in rows), "unsupported": len(UNSUPPORTED_CONDITIONS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
