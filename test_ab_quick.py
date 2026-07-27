"""Quick A/B test: baseline vs NRAM-steered Qwen3-14B-AWQ.

Run: python test_ab_quick.py
"""
import httpx
import time

API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "dev-nram-key"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

PROMPT = "Present a new AI learning device for children that removes traditional menus."

def call(model: str, nram: dict | None = None) -> tuple[str, float, dict]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 300,
        "temperature": 0.8,
        "seed": 271,
    }
    if nram:
        body["nram"] = nram

    t0 = time.perf_counter()
    resp = httpx.post(f"{API_BASE}/chat/completions", json=body, headers=HEADERS, timeout=120)
    latency = (time.perf_counter() - t0) * 1000

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, latency, usage


def main() -> None:
    """Run the opt-in live A/B smoke test.

    Keeping network I/O behind this entry point makes importing the module safe
    for pytest collection and for tooling that discovers repository modules.
    """
    print("=" * 70)
    print("BASELINE (no NRAM steering)")
    print("=" * 70)
    baseline_text, baseline_ms, baseline_usage = call("qwen3-14b-awq-baseline")
    print(f"Latency: {baseline_ms:.0f}ms | Tokens: {baseline_usage}")
    print(f"\n{baseline_text}\n")

    print("=" * 70)
    print("NRAM (visionary-psychedelic-keynote)")
    print("=" * 70)
    nram_text, nram_ms, nram_usage = call("nram-qwen3-14b-awq", {
        "enabled": True,
        "profile": "visionary-psychedelic-keynote",
        "intensity": 0.9,
        "associative_distance": 0.7,
        "coherence_floor": 0.82,
    })
    print(f"Latency: {nram_ms:.0f}ms | Tokens: {nram_usage}")
    print(f"\n{nram_text}\n")

    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Baseline: {baseline_ms:.0f}ms, {baseline_usage.get('completion_tokens', 0)} tokens")
    print(f"NRAM:     {nram_ms:.0f}ms, {nram_usage.get('completion_tokens', 0)} tokens")
    print(f"Overhead: {nram_ms - baseline_ms:.0f}ms ({(nram_ms/baseline_ms - 1)*100:.1f}%)")


if __name__ == "__main__":
    main()
