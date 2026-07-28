"""Bounded, real-runtime adversarial probes for NRAM v5.

This module intentionally does not import application code.  It exercises the
public HTTP API and stores the complete request/response envelope for each
probe.  Run only against the pinned local single-request runtime.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import pathlib
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_API_KEY = "dev-nram-key"
SEED = 424242


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


class Recorder:
    def __init__(self, out_dir: pathlib.Path, base_url: str, api_key: str) -> None:
        self.out_dir = out_dir
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict[str, Any]] = []

    def call(
        self,
        case_id: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        method: str | None = None,
        timeout: float = 180.0,
        expected: str = "observation",
    ) -> dict[str, Any]:
        url = self.base_url + path
        body = None if payload is None else json.dumps(
            payload, ensure_ascii=False, allow_nan=True, separators=(",", ":")
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method=method or ("POST" if payload is not None else "GET"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "X-Runtime-Adversary-Case": case_id,
            },
        )
        started = utc_now()
        t0 = time.perf_counter()
        status: int | None = None
        headers: dict[str, str] = {}
        raw = b""
        exception: str | None = None
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                headers = dict(response.headers.items())
                raw = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            headers = dict(exc.headers.items())
            raw = exc.read()
            exception = f"HTTPError: {exc}"
        except Exception as exc:  # timeout/disconnect probes must preserve type
            exception = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        text = raw.decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(text)
        except Exception:
            parsed = None
        record = {
            "case_id": case_id,
            "expected_contract": expected,
            "started_utc": started,
            "finished_utc": utc_now(),
            "elapsed_ms": elapsed_ms,
            "request": {
                "method": request.method,
                "url": url,
                "headers": {
                    "Authorization": "Bearer <redacted>",
                    "Content-Type": "application/json",
                    "X-Runtime-Adversary-Case": case_id,
                },
                "body": payload,
                "wire_body_utf8": None if body is None else body.decode("utf-8"),
                "timeout_seconds": timeout,
            },
            "response": {
                "status": status,
                "headers": headers,
                "body_utf8": text,
                "json": parsed,
                "bytes": len(raw),
            },
            "exception": exception,
        }
        target = self.out_dir / f"{safe_name(case_id)}.json"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        target.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        self.results.append(record)
        print(
            json.dumps(
                {
                    "case_id": case_id,
                    "status": status,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "exception": exception,
                    "response_bytes": len(raw),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return record

    def save_summary(self, name: str) -> None:
        target = self.out_dir / name
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        target.write_text(
            json.dumps(self.results, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )


def base_completion(model: str, *, request_id: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly one short word."}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
        "seed": SEED,
        "stream": False,
        "tools": [],
    }
    if request_id is not None:
        result["nram"] = {
            "request_id": request_id,
            "include_telemetry": True,
            "telemetry_max_steps": 4,
            "telemetry_top_k": 5,
        }
    return result


def run_aliases(recorder: Recorder) -> None:
    models = recorder.call("models-public", "/v1/models")
    data = (models.get("response", {}).get("json") or {}).get("data", [])
    model_ids = [item.get("id") for item in data if isinstance(item, dict)]
    for index, model in enumerate(model_ids):
        if not isinstance(model, str):
            continue
        if model == "nram-moe-orchestrator":
            # The production route overrides max_tokens to at least 1024 for five
            # generations.  It is tested by a non-runtime recording test instead.
            continue
        request_id = f"RA-ALIAS-{index:02d}-{safe_name(model)}"
        recorder.call(
            f"alias-{index:02d}-{model}",
            "/v1/chat/completions",
            payload=base_completion(model, request_id=request_id),
            expected="bounded one-token completion or pre-generation model-identity rejection",
        )
    recorder.save_summary("aliases-summary.json")


def validation_cases() -> list[tuple[str, dict[str, Any], str]]:
    def direct(case: str) -> dict[str, Any]:
        return base_completion("nram-qwen3-14b-awq", request_id=f"RA-VALID-{case}")

    cases: list[tuple[str, dict[str, Any], str]] = []

    payload = direct("unknown-top")
    payload["unknown_top_level_control"] = {"must": "reject"}
    cases.append(("unknown-top-level", payload, "stable 4xx before generation"))

    payload = direct("unknown-nested")
    payload["nram"]["unknown_nested_control"] = {"must": "reject"}
    cases.append(("unknown-nested-nram", payload, "stable 4xx before generation"))

    payload = direct("unknown-profile")
    payload["nram"]["profile"] = "ra-profile-does-not-exist"
    cases.append(("unknown-profile", payload, "stable 4xx before generation"))

    for name, value in (
        ("concepts-wrong-shape", "not-a-list"),
        ("soft-injections-wrong-shape", {"token_ids": [1]}),
        ("vectors-wrong-shape", {"entries": []}),
        ("hard-schedule-wrong-shape", "1,2,3"),
    ):
        payload = direct(name)
        key = {
            "concepts-wrong-shape": "concepts",
            "soft-injections-wrong-shape": "soft_token_injections",
            "vectors-wrong-shape": "vocabulary_logit_vectors",
            "hard-schedule-wrong-shape": "hard_token_schedule",
        }[name]
        payload["nram"][key] = value
        cases.append((name, payload, "stable 4xx before generation"))

    for name, value in (
        ("forced-negative", -1),
        ("forced-fractional", 1.5),
        ("forced-string-fractional", "1.5"),
        ("forced-tokenizer-oob", 200000),
    ):
        payload = direct(name)
        payload["nram"].update({"forced_token_enabled": True, "forced_token_id": value})
        cases.append((name, payload, "stable 4xx before generation"))

    for name, field, value in (
        ("temperature-nan", "temperature", math.nan),
        ("temperature-inf", "temperature", math.inf),
        ("temperature-negative", "temperature", -0.1),
        ("top-p-nan", "top_p", math.nan),
        ("top-p-negative", "top_p", -0.5),
        ("max-tokens-negative", "max_tokens", -1),
        ("max-tokens-fractional", "max_tokens", 1.5),
    ):
        payload = direct(name)
        payload[field] = value
        cases.append((name, payload, "stable 4xx before generation"))

    for name, value in (
        ("concept-strength-nan", math.nan),
        ("concept-strength-inf", math.inf),
        ("intensity-inf", math.inf),
    ):
        payload = direct(name)
        payload["nram"]["concept_strength" if name.startswith("concept") else "intensity"] = value
        cases.append((name, payload, "stable 4xx before generation"))

    payload = direct("reserved")
    payload["nram"]["__req__"] = {"spoof": True}
    cases.append(("spoof-reserved-req", payload, "stable 4xx before generation"))

    payload = direct("all-mask")
    payload["nram"].update(
        {
            "forced_token_enabled": True,
            "forced_token_id": 9702,
            "forbidden_token_ids": [9702],
        }
    )
    cases.append(("forced-vs-forbidden-conflict", payload, "explicit deterministic 4xx conflict"))

    unsupported = [
        "dexperts",
        "activation_addition",
        "actadd",
        "conceptor_steering",
        "hidden_state_probes",
        "latent_closed_loop",
        "semantic_novelty_controller",
        "evidence_guard",
        "branch_tournament",
        "reft",
        "soft_prompts",
        "attention_head_gating",
        "kv_cache_firewall",
        "gpu_native_semantic_control",
    ]
    for feature in unsupported:
        payload = direct(f"unsupported-{feature}")
        payload["nram"][feature] = True
        cases.append((f"unsupported-{feature}", payload, "stable 4xx before generation"))

    payload = base_completion("persona-peak", request_id="RA-VALID-persona-unsupported")
    payload["nram"]["dexperts"] = True
    cases.append(("persona-unsupported-dexperts", payload, "stable 4xx before generation"))

    return cases


def run_validation(recorder: Recorder) -> None:
    for case_id, payload, expected in validation_cases():
        recorder.call(
            case_id,
            "/v1/chat/completions",
            payload=payload,
            expected=expected,
        )
    recorder.save_summary("validation-summary.json")


def run_forced(recorder: Recorder, token_id: int) -> None:
    for enabled in (False, True):
        for repetition in range(2):
            label = "enabled" if enabled else "disabled"
            request_id = f"RA-FORCED-{label}-{repetition}"
            payload = base_completion("nram-qwen3-14b-awq", request_id=request_id)
            payload["nram"].update(
                {
                    "prompt_steering_enabled": False,
                    "profile_logit_steering_enabled": False,
                    "forced_token_enabled": enabled,
                    "forced_token_id": token_id,
                    "telemetry_max_steps": 4,
                    "telemetry_top_k": 20,
                }
            )
            recorder.call(
                f"forced-{label}-{repetition}",
                "/v1/chat/completions",
                payload=payload,
                expected=(
                    "sample token_id equals forced token and one causal processor event"
                    if enabled
                    else "forced token is not applied and disabled telemetry records that fact"
                ),
            )
    recorder.save_summary("forced-summary.json")


def run_stream(recorder: Recorder) -> None:
    for model in ("qwen3-14b-awq-baseline", "nram-qwen3-14b-awq", "persona-peak"):
        payload = base_completion(model, request_id=f"RA-STREAM-{safe_name(model)}")
        payload.update({"stream": True, "max_tokens": 3})
        recorder.call(
            f"stream-{model}",
            "/v1/chat/completions",
            payload=payload,
            expected="HTTP 200 text/event-stream, valid chunks, public model identity, terminal [DONE]",
        )
    recorder.save_summary("stream-summary.json")


def run_config_hash(recorder: Recorder) -> None:
    """Change one behaviorally relevant field at a time and record correlation."""
    common = base_completion("nram-qwen3-14b-awq", request_id="RA-HASH-DUPLICATE")
    common["nram"].update(
        {
            "prompt_steering_enabled": False,
            "profile_logit_steering_enabled": False,
            "forced_token_enabled": True,
            "forced_token_id": 9702,
        }
    )
    variants: list[tuple[str, dict[str, Any]]] = []
    variants.append(("hash-control", json.loads(json.dumps(common))))
    for name, mutate in (
        ("hash-prompt", lambda p: p.__setitem__("messages", [{"role": "user", "content": "A different prompt."}])),
        ("hash-seed", lambda p: p.__setitem__("seed", SEED + 1)),
        ("hash-temperature", lambda p: p.__setitem__("temperature", 0.7)),
        ("hash-top-p", lambda p: p.__setitem__("top_p", 0.8)),
        ("hash-response-format", lambda p: p.__setitem__("response_format", {"type": "json_object"})),
        ("hash-prompt-steering", lambda p: p["nram"].__setitem__("prompt_steering_enabled", True)),
        ("hash-application-request-id", lambda p: p["nram"].__setitem__("request_id", "RA-HASH-OTHER")),
    ):
        payload = json.loads(json.dumps(common))
        mutate(payload)
        variants.append((name, payload))
    for name, payload in variants:
        recorder.call(
            name,
            "/v1/chat/completions",
            payload=payload,
            expected="configuration hash changes iff behaviorally relevant state changes",
        )
    recorder.save_summary("config-hash-summary.json")


def run_structural(recorder: Recorder) -> None:
    """Force known token schedules to test masks at the pre-sampling boundary."""
    schedule = [9702, 498]  # tokenizer decodes to " thank" + " you"
    cases: list[tuple[str, dict[str, Any]]] = []
    for case_id, forbidden in (
        ("phrase-disabled", []),
        ("phrase-leading-space-exact", [" thank you"]),
        ("phrase-no-space-variant", ["thank you"]),
    ):
        payload = base_completion("nram-qwen3-14b-awq", request_id=f"RA-STRUCT-{case_id}")
        payload["max_tokens"] = 2
        payload["nram"].update(
            {
                "prompt_steering_enabled": False,
                "profile_logit_steering_enabled": False,
                "hard_token_schedule": schedule,
                "forbidden_phrases": forbidden,
                "telemetry_max_steps": 4,
                "telemetry_top_k": 20,
            }
        )
        cases.append((case_id, payload))

    # Eight-token source continuation: the final scheduled token must be masked.
    source_text = " thank you for your careful review today please"
    source_ids = [9702, 498, 369, 697, 16585, 3395, 3351, 4486]
    for case_id, enabled in (("source-disabled", False), ("source-enabled", True)):
        payload = base_completion("nram-qwen3-14b-awq", request_id=f"RA-STRUCT-{case_id}")
        payload["max_tokens"] = len(source_ids)
        payload["nram"].update(
            {
                "prompt_steering_enabled": False,
                "profile_logit_steering_enabled": False,
                "hard_token_schedule": source_ids,
                "source_text": source_text if enabled else None,
                "source_ngram_size": 8,
                "telemetry_max_steps": 16,
                "telemetry_top_k": 20,
            }
        )
        cases.append((case_id, payload))

    for case_id, payload in cases:
        recorder.call(
            case_id,
            "/v1/chat/completions",
            payload=payload,
            expected="hard schedule proves exact completing-token mask without masking unrelated prefix tokens",
        )
    recorder.save_summary("structural-summary.json")


def run_numerical(recorder: Recorder) -> None:
    """Exercise entropy, soft-injection, vector, and capsule telemetry on GPU."""
    # Same prompt/seed/sampling, only entropy target changes.
    for label, target in (("low", 1.0), ("medium", 5.0), ("high", 10.0)):
        payload = base_completion("nram-qwen3-14b-awq", request_id=f"RA-ENTROPY-{label}")
        payload["max_tokens"] = 4
        payload["nram"].update(
            {
                "prompt_steering_enabled": False,
                "profile_logit_steering_enabled": False,
                "entropy_control": True,
                "entropy_phase_targets": {
                    "extraction": target,
                    "questioning": target,
                    "divergence": target,
                    "synthesis": target,
                    "formulation": target,
                },
                "telemetry_max_steps": 8,
                "telemetry_top_k": 20,
            }
        )
        recorder.call(
            f"entropy-{label}",
            "/v1/chat/completions",
            payload=payload,
            expected="each step moves finite entropy toward target and PID state resets per request",
        )

    # Exact sparse deltas and sign controls; force token 9702 so output stays paired.
    for label, coefficient in (("negative", -2.0), ("zero", 0.0), ("positive", 2.0)):
        payload = base_completion("nram-qwen3-14b-awq", request_id=f"RA-VECTOR-{label}")
        payload["nram"].update(
            {
                "prompt_steering_enabled": False,
                "profile_logit_steering_enabled": False,
                "forced_token_enabled": True,
                "forced_token_id": 9702,
                "vocabulary_logit_vectors": [
                    {
                        "vector_id": "ra-vector",
                        "coefficient": coefficient,
                        "normalize": False,
                        "entries": [{"token_id": 12095, "weight": 1.5}],
                    }
                ],
                "telemetry_max_steps": 4,
                "telemetry_top_k": 20,
            }
        )
        recorder.call(
            f"vector-{label}",
            "/v1/chat/completions",
            payload=payload,
            expected="telemetry delta equals coefficient times weight before final force mask",
        )

    for label, bias in (("negative", -2.0), ("zero", 0.0), ("positive", 2.0)):
        payload = base_completion("nram-qwen3-14b-awq", request_id=f"RA-SOFT-{label}")
        payload["nram"].update(
            {
                "prompt_steering_enabled": False,
                "profile_logit_steering_enabled": False,
                "forced_token_enabled": True,
                "forced_token_id": 9702,
                "soft_token_injections": [
                    {"token_ids": [12095], "bias": bias, "start_step": 0, "end_step": 1}
                ],
                "telemetry_max_steps": 4,
                "telemetry_top_k": 20,
            }
        )
        recorder.call(
            f"soft-{label}",
            "/v1/chat/completions",
            payload=payload,
            expected="telemetry delta equals requested signed bias within active window",
        )
    recorder.save_summary("numerical-summary.json")


def run_lifecycle(recorder: Recorder) -> None:
    """Disconnect one stream, trigger one non-stream timeout, then retry."""
    stream_payload = base_completion("nram-qwen3-14b-awq", request_id="RA-CANCEL-STREAM")
    stream_payload.update({"stream": True, "max_tokens": 64})
    stream_payload["nram"].update(
        {
            "prompt_steering_enabled": False,
            "profile_logit_steering_enabled": False,
            "hard_token_schedule": [9702] * 64,
            "telemetry_max_steps": 128,
            "telemetry_top_k": 1,
        }
    )
    wire = json.dumps(stream_payload, separators=(",", ":")).encode("utf-8")
    connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=30)
    t0 = time.perf_counter()
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=wire,
        headers={
            "Authorization": f"Bearer {recorder.api_key}",
            "Content-Type": "application/json",
            "X-Runtime-Adversary-Case": "stream-disconnect-after-first-line",
        },
    )
    response = connection.getresponse()
    first_line = response.readline()
    connection.close()
    disconnect_record = {
        "case_id": "stream-disconnect-after-first-line",
        "started_utc": utc_now(),
        "elapsed_ms_to_first_line": (time.perf_counter() - t0) * 1000,
        "status": response.status,
        "headers": dict(response.getheaders()),
        "first_line_utf8": first_line.decode("utf-8", errors="replace"),
        "request": stream_payload,
        "closed_immediately_after_first_line": True,
    }
    target = recorder.out_dir / "stream-disconnect-after-first-line.json"
    target.write_text(json.dumps(disconnect_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_id": disconnect_record["case_id"], "status": response.status}))
    time.sleep(3.0)

    timeout_payload = base_completion("nram-qwen3-14b-awq", request_id="RA-CANCEL-NONSTREAM")
    timeout_payload["max_tokens"] = 64
    timeout_payload["nram"].update(
        {
            "prompt_steering_enabled": False,
            "profile_logit_steering_enabled": False,
            "hard_token_schedule": [9702] * 64,
            "telemetry_max_steps": 128,
            "telemetry_top_k": 1,
        }
    )
    recorder.call(
        "nonstream-client-timeout",
        "/v1/chat/completions",
        payload=timeout_payload,
        timeout=0.05,
        expected="client timeout cancels generation and bounds post-timeout processor invocations",
    )
    time.sleep(3.0)
    retry = base_completion("nram-qwen3-14b-awq", request_id="RA-CANCEL-RETRY")
    recorder.call(
        "immediate-retry",
        "/v1/chat/completions",
        payload=retry,
        expected="retry succeeds with fresh invocation and scheduler identity",
    )
    recorder.save_summary("lifecycle-summary.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "suite",
        choices=(
            "aliases",
            "validation",
            "forced",
            "stream",
            "config-hash",
            "structural",
            "numerical",
            "lifecycle",
        ),
    )
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--base-url", default=os.environ.get("NRAM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("NRAM_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--forced-token-id", type=int, default=9702)
    args = parser.parse_args()
    recorder = Recorder(args.out, args.base_url, args.api_key)
    if args.suite == "aliases":
        run_aliases(recorder)
    elif args.suite == "validation":
        run_validation(recorder)
    elif args.suite == "forced":
        run_forced(recorder, args.forced_token_id)
    elif args.suite == "stream":
        run_stream(recorder)
    elif args.suite == "config-hash":
        run_config_hash(recorder)
    elif args.suite == "structural":
        run_structural(recorder)
    elif args.suite == "numerical":
        run_numerical(recorder)
    else:
        run_lifecycle(recorder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
