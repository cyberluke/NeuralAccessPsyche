"""A/B evaluation harness: baseline model vs NRAM-steered model.

For every prompt this script queries an OpenAI-compatible endpoint twice —
once with the baseline model and once with the NRAM model — using identical
sampling parameters and seed, but with randomized A/B order to avoid
order/position bias. Heuristic metrics from ``metrics.py`` are computed on
both outputs and written to a results JSONL file.

The module imports only the standard library at import time; the ``openai``
client is created lazily inside ``main()`` so the module can be imported and
tested without a live server or the dependency installed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, Optional

from metrics import compute_all

BASELINE_MODEL = "deepseek-r1-qwen-7b-baseline"
NRAM_MODEL = "nram-deepseek-r1-qwen-7b"


def load_prompts(path: str) -> list:
    """Load prompts from a JSONL file (one JSON object per line)."""
    prompts = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] skipping malformed line {lineno}: {exc}", file=sys.stderr)
                continue
            if "prompt" not in obj:
                print(f"[warn] skipping line {lineno}: missing 'prompt'", file=sys.stderr)
                continue
            prompts.append(obj)
    return prompts


def _call_model(client: Any, model: str, prompt: str, seed: int,
                temperature: float, top_p: float, max_tokens: int) -> Dict[str, Any]:
    """Call the endpoint once and return {'output': str, 'raw': dict}."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    output = ""
    try:
        output = response.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        output = ""
    raw = None
    try:
        raw = response.model_dump()
    except Exception:  # pragma: no cover - defensive
        raw = None
    return {"output": output, "raw": raw}


def run_pair(client: Any, entry: Dict[str, Any], seed: int, temperature: float,
             top_p: float, max_tokens: int, rng: random.Random) -> Dict[str, Any]:
    """Run one baseline/NRAM pair with randomized order. Never raises."""
    prompt = entry.get("prompt", "")
    result: Dict[str, Any] = {
        "id": entry.get("id"),
        "category": entry.get("category"),
        "prompt": prompt,
        "seed": seed,
        "order": None,
        "baseline_output": None,
        "nram_output": None,
        "baseline_metrics": None,
        "nram_metrics": None,
        "baseline_error": None,
        "nram_error": None,
    }

    # Randomize A/B order to avoid position bias, but record which was which.
    order = ["baseline", "nram"]
    if rng.random() < 0.5:
        order = ["nram", "baseline"]
    result["order"] = order

    for label in order:
        model = BASELINE_MODEL if label == "baseline" else NRAM_MODEL
        try:
            call = _call_model(client, model, prompt, seed, temperature, top_p, max_tokens)
            result[f"{label}_output"] = call["output"]
            result[f"{label}_metrics"] = compute_all(call["output"])
        except Exception as exc:  # record error, continue with the other arm
            result[f"{label}_error"] = f"{type(exc).__name__}: {exc}"

    return result


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Baseline vs NRAM A/B evaluation harness.")
    parser.add_argument("--input", default="evaluation/prompts.jsonl",
                        help="Path to prompts JSONL file.")
    parser.add_argument("--output", default="evaluation/results.jsonl",
                        help="Path to write results JSONL file.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampling seed passed to the endpoint.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling top_p.")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max output tokens per completion.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N prompts.")
    args = parser.parse_args(argv)

    # Lazy import: only needed when actually hitting a server.
    try:
        from openai import OpenAI
    except ImportError:
        print("error: the 'openai' package is required to run evaluations. "
              "Install it with: pip install openai", file=sys.stderr)
        return 2

    base_url = os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("EVAL_API_KEY", "dev-nram-key")
    client = OpenAI(base_url=base_url, api_key=api_key)

    prompts = load_prompts(args.input)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    if not prompts:
        print(f"error: no prompts loaded from {args.input}", file=sys.stderr)
        return 1

    print(f"Evaluating {len(prompts)} prompt(s) "
          f"(baseline={BASELINE_MODEL}, nram={NRAM_MODEL}) "
          f"seed={args.seed} temp={args.temperature} top_p={args.top_p} "
          f"max_tokens={args.max_tokens}")

    rng = random.Random(args.seed)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as fh:
        for idx, entry in enumerate(prompts, start=1):
            result = run_pair(client, entry, args.seed, args.temperature,
                              args.top_p, args.max_tokens, rng)
            fh.write(json.dumps(result) + "\n")
            fh.flush()
            status = []
            if result["baseline_error"]:
                status.append("baseline_err")
            if result["nram_error"]:
                status.append("nram_err")
            tag = " [" + ",".join(status) + "]" if status else ""
            print(f"  ({idx}/{len(prompts)}) {result['id']} order={result['order']}{tag}")

    print(f"Done. Results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
