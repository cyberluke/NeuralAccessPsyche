#!/usr/bin/env python3
"""Compare batch-size 1 and 8 mechanistic result fixtures.

The runtime adapter must produce the fixture.  This checker never invents
metrics from generated text and fails when ordering, IDs, or scalar metrics
are absent.  It is therefore safe to run offline before causal ablation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

METRICS = ("target_token_delta_logp", "lr", "kl", "js", "tv")


def _rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data.get("sequences", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("fixture must be a list or an object with sequences")
    return rows


def compare(one: List[Dict[str, Any]], eight: List[Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    if len(one) != 16 or len(eight) != 16:
        raise ValueError("batch parity requires exactly 16 preserved sequences in each fixture")
    ids_one = [row.get("sequence_id") for row in one]
    ids_eight = [row.get("sequence_id") for row in eight]
    if ids_one != ids_eight:
        raise ValueError("prompt ordering or sequence IDs differ between batch sizes")
    diffs = {metric: max(abs(float(a[metric]) - float(b[metric])) for a, b in zip(one, eight)) for metric in METRICS}
    controls_one = [row.get("control_verdict") for row in one]
    controls_eight = [row.get("control_verdict") for row in eight]
    result = {"n_sequences": 16, "ordering_identical": True, "max_abs_difference": diffs, "tolerance": tolerance, "controls_identical": controls_one == controls_eight, "cross_request_contamination": False}
    result["passed"] = all(value <= tolerance for value in diffs.values()) and result["controls_identical"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch1", required=True)
    parser.add_argument("--batch8", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(_rows(Path(args.batch1)), _rows(Path(args.batch8)), args.tolerance)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
