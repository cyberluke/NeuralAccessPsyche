#!/usr/bin/env python3
"""Fail-closed readiness checks for NRAM v5 R5 causal ablation.

This module deliberately does not run the confirmatory ablation.  Runtime
checks are explicit, artifact-producing gates.  A missing endpoint, missing
adapter weight, unverifiable model revision, or unsupported telemetry schema
is reported as ``BLOCKED`` rather than silently treated as a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R5_DIR = ROOT / "artifacts" / "dexperts" / "r5"
R6_DIR = ROOT / "artifacts" / "dexperts" / "r6"
ADAPTER_ROOT = ROOT / "artifacts" / "dexperts" / "adapters"

REGIMES: Dict[str, Dict[str, Any]] = {
    "06b": {
        "model": "Qwen/Qwen3-0.6B-Base",
        "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "endpoint": "http://127.0.0.1:30001/v1/completions",
        "format": "raw_completions",
        "tokenizer_revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    },
    "14b": {
        "model": "nram-qwen3-14b-awq",
        "revision": "31c69efc29464b6bb0aee1398b5a7b50a99340c3",
        "endpoint": "http://127.0.0.1:30000/v1/completions",
        "format": "raw_completions",
        "tokenizer_revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    },
}


def sha256_tree(path: Path) -> str | None:
    """Hash all files under an adapter directory in deterministic order."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    files = [p for p in path.rglob("*") if p.is_file() and "checkpoint-" not in p.parts]
    for file_path in sorted(files):
        digest.update(str(file_path.relative_to(path)).replace("\\", "/").encode())
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _check(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "BLOCKED", "detail": detail}


def _http_json(url: str, api_key: str, timeout: float = 5.0) -> tuple[int, Any, str | None]:
    try:
        import httpx
        response = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
        try:
            body = response.json()
        except Exception:
            body = response.text[:500]
        return response.status_code, body, None
    except Exception as exc:  # pragma: no cover - depends on local runtime
        return 0, None, str(exc)


def preflight(regime: str, api_url: str | None, api_key: str) -> Dict[str, Any]:
    """Validate runtime identity, adapters, request/response contracts and controls."""
    expected = REGIMES[regime]
    endpoint = api_url or expected["endpoint"]
    base = endpoint.rsplit("/v1/", 1)[0]
    checks: List[Dict[str, Any]] = []

    status, health, error = _http_json(base + "/health", api_key)
    checks.append(_check("server_health", status == 200, f"HTTP {status}; {error or health}"))

    status, models, error = _http_json(base + "/v1/models", api_key)
    ids: List[str] = []
    if isinstance(models, Mapping):
        ids = [str(x.get("id")) for x in models.get("data", []) if isinstance(x, Mapping)]
    checks.append(_check("reported_model_identity", expected["model"] in ids or (regime == "14b" and "nram-qwen3-14b-awq" in ids), f"reported={ids}, expected={expected['model']}"))
    checks.append(_check("base_model_revision", False, "Revision is not exposed by the OpenAI models contract; server-side metadata is required"))

    for name, adapter_role in (("expert", "nontoxic"), ("anti_expert", "toxic")):
        adapter_path = ADAPTER_ROOT / adapter_role
        config_path = adapter_path / "adapter_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        weight_files = [p for p in adapter_path.iterdir() if p.name.startswith("adapter_model") and p.is_file()] if adapter_path.exists() else []
        digest = sha256_tree(adapter_path)
        checks.append(_check(f"{name}_adapter_hash", digest is not None, f"sha256={digest}; weight_files={[p.name for p in weight_files]}"))
        checks.append(_check(f"{name}_role_assignment", (adapter_role == "nontoxic") == (name == "expert"), f"role={adapter_role}"))
        checks.append(_check(f"{name}_base_compatibility", config.get("base_model_name_or_path", "").endswith(expected["revision"]), f"base={config.get('base_model_name_or_path')}"))

    expected_path = "/v1/completions" if expected["format"] == "raw_completions" else "/v1/chat/completions"
    checks.append(_check("endpoint_compatibility", endpoint.endswith(expected_path), f"endpoint={endpoint}; expected suffix={expected_path}"))
    checks.append(_check("tokenizer_compatibility", False, "Tokenizer/model fingerprint comparison requires the serving instance to report tokenizer revision and vocabulary hash"))
    checks.append(_check("output_schema_compatibility", False, "A successful generation probe is required; preflight never substitutes a synthetic response"))
    checks.append(_check("deterministic_configuration", True, "Seeds [42, 123, 456], temperature=0.7, max_tokens=100 are captured by the experiment contract"))

    # These controls require deterministic runtime responses and are not inferred
    # from HTTP health. They are intentionally blocked until a probe is supplied.
    for control in ("activation", "deactivation", "swapped_adapter"):
        checks.append(_check(f"{control}_behavior", False, "Requires deterministic runtime control probe; health alone is insufficient"))

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "regime": regime,
        "expected": expected,
        "endpoint": endpoint,
        "checks": checks,
        "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "BLOCKED",
    }
    R5_DIR.mkdir(parents=True, exist_ok=True)
    (R5_DIR / f"preflight_{regime}.json").write_text(json.dumps(result, indent=2))
    return result


def _records(phase4: Mapping[str, Any], condition: str) -> List[Mapping[str, Any]]:
    group = phase4["all_results"][condition]
    return list(group.get("toxic_sequences", [])) + list(group.get("nontoxic_sequences", []))


def offline_analysis() -> Dict[str, Any]:
    """Recompute power, leakage, and existing adapter/evaluator gates offline."""
    phase4_path = R6_DIR / "mechanistic_proof_phase4.json"
    phase4 = json.loads(phase4_path.read_text())
    normal = phase4["all_results"]["normal"]["signed_gap"]
    effect = float(normal["effect_size"])
    n = int(phase4["config"]["n_per_class"])
    # Normal approximation is reported alongside the exact t approximation in
    # the existing reconciliation script; no SGLang/model invocation occurs.
    z_alpha = 1.959963984540054
    z_beta = 0.8416212335729143
    required_n = int(math.ceil(2 * ((z_alpha + z_beta) / abs(effect)) ** 2)) if effect else None
    power = None
    try:
        from scipy import stats
        df = 2 * n - 2
        ncp = effect * math.sqrt(n / 2)
        crit = stats.t.ppf(0.975, df=df)
        power = float(1 - stats.nct.cdf(crit, df=df, nc=ncp) + stats.nct.cdf(-crit, df=df, nc=ncp))
    except Exception:
        pass

    evaluator = json.loads((R6_DIR / "evaluator_calibration.json").read_text())
    adapter = json.loads((R6_DIR / "adapter_discrimination_corrected.json").read_text())
    evaluator_gate = bool(evaluator.get("metrics", {}).get("roc_auc", 0) >= 0.70 and evaluator.get("class_distributions", {}).get("toxic", {}).get("mean", 0) > evaluator.get("class_distributions", {}).get("nontoxic", {}).get("mean", 0))
    adapter_gate = bool(adapter.get("discrimination_metrics", {}).get("roc_auc", 0) >= 0.70 and adapter.get("discrimination_metrics", {}).get("ci_lower", 0) > 0.50 and adapter.get("sign_checks", {}).get("expert_prefers_nontoxic") and adapter.get("sign_checks", {}).get("anti_prefers_toxic"))

    # Phase 4 contains IDs/metrics but no source text, so exact and near-
    # duplicate leakage cannot be certified from this artifact alone.
    ids = []
    for row in _records(phase4, "normal"):
        ids.append((row.get("class_label"), row.get("prompt_id")))
    leakage = {"status": "BLOCKED", "detail": "Phase 4 sequence artifact contains prompt_id and metrics, not source text; exact/near-duplicate audit requires the preserved source manifest", "duplicate_sequence_keys": len(ids) != len(set(ids))}

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(phase4_path),
        "power": {"effect_size": effect, "n_per_class": n, "achieved_two_sided_power": power, "required_n_per_class_power_80": required_n},
        "evaluator_gate": {"status": "PASS" if evaluator_gate and evaluator.get("acceptance_gates", {}).get("all_pass", False) else "BLOCKED", "source": str(R6_DIR / "evaluator_calibration.json"), "note": "ECE gate from prior artifact is false; calibration is not accepted as fully passed"},
        "adapter_gate": {"status": "PASS" if adapter_gate else "BLOCKED", "source": str(R6_DIR / "adapter_discrimination_corrected.json")},
        "leakage_gate": leakage,
        "causal_ablation_authorized": False,
    }
    (R5_DIR / "offline_readiness.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Recompute gates/power without invoking SGLang")
    parser.add_argument("--regime", choices=sorted(REGIMES), default="14b")
    parser.add_argument("--api-url")
    parser.add_argument("--api-key", default="dev-nram-key")
    args = parser.parse_args()
    if not args.preflight and not args.offline:
        parser.error("select --preflight or --offline")
    result = preflight(args.regime, args.api_url, args.api_key) if args.preflight else offline_analysis()
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
