"""Aggregate A/B results and report a PASS/FAIL verdict.

Reads ``results.jsonl`` (produced by ``run_ab.py``), computes mean metrics
across all prompts for the baseline and NRAM arms, evaluates the success
criteria, and prints an honest summary. It never fudges — failures are
reported directly.

Success criteria:
  1. At least 25% reduction in corporate jargon (mean corporate_jargon_count).
  2. No more than ~5% loss in relevance, approximated here by the stability of
     output_length and product_noun_count (see APPROXIMATION NOTE below).
  3. No significant rise in repeated phrases (repeated_phrase_ratio).

APPROXIMATION NOTE:
  True semantic relevance is not measurable with offline heuristics. As a proxy
  we assume an answer that stays roughly the same length and keeps a similar
  density of product-related nouns has not drifted off-topic. This is a coarse
  signal and is labeled as such; it should not be mistaken for a quality score.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional

# Tunable thresholds.
JARGON_REDUCTION_THRESHOLD = 0.25       # >= 25% reduction required
RELEVANCE_MAX_LOSS = 0.05               # <= 5% loss tolerated (approximation)
REPEATED_RISE_TOLERANCE = 0.02          # allow up to +2 percentage points

# Metrics to aggregate (means over prompts).
AGG_KEYS = [
    "corporate_jargon_count",
    "repeated_phrase_ratio",
    "short_declarative_ratio",
    "product_noun_count",
    "forbidden_cliche_count",
    "output_length",
]


def load_results(path: str) -> List[dict]:
    """Load results, keeping only rows where both arms produced metrics."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("baseline_metrics") and obj.get("nram_metrics"):
                rows.append(obj)
    return rows


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: List[dict]) -> Dict[str, Dict[str, float]]:
    """Return {'baseline': {metric: mean}, 'nram': {metric: mean}}."""
    agg = {"baseline": {}, "nram": {}}
    for arm in ("baseline", "nram"):
        for key in AGG_KEYS:
            values = []
            for row in rows:
                metrics = row.get(f"{arm}_metrics") or {}
                if key in metrics:
                    values.append(metrics[key])
            agg[arm][key] = _mean(values)
    return agg


def _pct_change(base: float, new: float) -> float:
    """Positive value means new is higher than base (an increase)."""
    if base == 0:
        return 0.0 if new == 0 else float("inf")
    return (new - base) / base


def evaluate(agg: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    """Evaluate each success criterion. Returns a list of check dicts."""
    b = agg["baseline"]
    n = agg["nram"]
    checks: List[Dict[str, object]] = []

    # 1. Corporate jargon reduction >= 25%.
    jargon_change = _pct_change(b["corporate_jargon_count"], n["corporate_jargon_count"])
    reduction = -jargon_change  # fraction reduced
    checks.append({
        "name": "Corporate jargon reduction >= 25%",
        "detail": (f"baseline={b['corporate_jargon_count']:.3f} "
                   f"nram={n['corporate_jargon_count']:.3f} "
                   f"(change={jargon_change * 100:+.1f}%)"),
        "passed": jargon_change <= -JARGON_REDUCTION_THRESHOLD,
    })

    # 2. Relevance proxy: output_length and product_noun stability (<= 5% loss).
    len_change = _pct_change(b["output_length"], n["output_length"])
    noun_change = _pct_change(b["product_noun_count"], n["product_noun_count"])
    relevance_ok = (len_change >= -RELEVANCE_MAX_LOSS and
                    noun_change >= -RELEVANCE_MAX_LOSS)
    checks.append({
        "name": "Relevance proxy loss <= 5% (approx: length & product-noun stability)",
        "detail": (f"output_length change={len_change * 100:+.1f}%, "
                   f"product_noun_count change={noun_change * 100:+.1f}%"),
        "passed": relevance_ok,
    })

    # 3. No significant rise in repeated phrases.
    repeated_rise = n["repeated_phrase_ratio"] - b["repeated_phrase_ratio"]
    checks.append({
        "name": "No significant rise in repeated phrases",
        "detail": (f"baseline={b['repeated_phrase_ratio']:.4f} "
                   f"nram={n['repeated_phrase_ratio']:.4f} "
                   f"(rise={repeated_rise:+.4f})"),
        "passed": repeated_rise <= REPEATED_RISE_TOLERANCE,
    })

    return checks


def _print_table(title: str, agg: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{title}")
    print("-" * 72)
    header = f"{'metric':<26}{'baseline':>14}{'nram':>14}{'change':>18}"
    print(header)
    print("-" * 72)
    for key in AGG_KEYS:
        b = agg["baseline"][key]
        n = agg["nram"][key]
        change = _pct_change(b, n)
        change_str = f"{change * 100:+.1f}%" if change != float("inf") else "+inf%"
        print(f"{key:<26}{b:>14.4f}{n:>14.4f}{change_str:>18}")
    print("-" * 72)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate A/B results and report verdict.")
    parser.add_argument("--input", default="evaluation/results.jsonl",
                        help="Path to results JSONL file.")
    args = parser.parse_args(argv)

    rows = load_results(args.input)
    if not rows:
        print(f"error: no usable results in {args.input} "
              f"(need rows with both baseline_metrics and nram_metrics).",
              file=sys.stderr)
        return 1

    agg = aggregate(rows)
    _print_table(f"Aggregated metrics over {len(rows)} prompt(s)", agg)

    checks = evaluate(agg)

    print("\nSuccess criteria")
    print("-" * 72)
    all_passed = True
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        if not check["passed"]:
            all_passed = False
        print(f"[{status}] {check['name']}")
        print(f"         {check['detail']}")
    print("-" * 72)

    verdict = "OVERALL: PASS" if all_passed else "OVERALL: FAIL"
    print(f"\n{verdict}")
    print("Note: relevance is approximated via output_length / product_noun "
          "stability, not semantic scoring.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
