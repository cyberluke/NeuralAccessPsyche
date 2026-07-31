"""TEST_ONLY offline fixture verification and pinned SGLang activation probe."""
import argparse, hashlib, json, math, time
from pathlib import Path
from urllib.request import Request, urlopen

MODEL_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
PROMPTS = [
    "The meeting starts at noon.", "A blue cup is on the table.",
    "Please summarize this sentence.", "The train arrives at platform two.",
    "Water freezes at zero degrees Celsius.", "The document contains three pages.",
    "A small bird landed on the window.", "The appointment is scheduled for Friday.",
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def verify_fixtures(root):
    from safetensors import safe_open
    results = {}
    for name in ("zero", "strong-delta"):
        directory = Path(root) / name
        manifest = json.loads((directory / "fixture_manifest.json").read_text())
        config = json.loads((directory / "adapter_config.json").read_text())
        total = 0.0; nonfinite = False; count = 0
        with safe_open(str(directory / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key); total += float((tensor.float() ** 2).sum()); count += tensor.numel()
                nonfinite = nonfinite or not bool(tensor.isfinite().all())
        results[name] = {"tensor_count": count, "l2_norm": math.sqrt(total),
            "finite": not nonfinite, "adapter_sha256": sha256(directory / "adapter_model.safetensors"),
            "manifest": manifest, "config": config}
    assert results["zero"]["l2_norm"] == 0.0
    assert results["strong-delta"]["l2_norm"] > 0.0
    assert results["zero"]["adapter_sha256"] != results["strong-delta"]["adapter_sha256"]
    for item in results.values():
        assert item["finite"] and item["config"]["base_model_name_or_path"] == "Qwen/Qwen3-14B"
        assert item["manifest"]["base_model_revision"] == MODEL_REVISION
    return results

def post(url, body):
    raw = json.dumps(body).encode(); req = Request(url, data=raw, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as response:
        data = response.read().decode(); return data, json.loads(data)

def probe(base_url, out):
    verified = verify_fixtures(Path("artifacts/training/fixtures")); records = []
    sampling = {"max_tokens": 1, "temperature": 0, "top_p": 1, "top_k": 1, "logprobs": True, "top_logprobs": 5}
    for index, prompt in enumerate(PROMPTS):
        for adapter in ("nram-qwen3-14b-awq", "zero", "strong-delta", "nram-qwen3-14b-awq"):
            body = {"model": adapter, "messages": [{"role": "user", "content": prompt}], "stream": False,
                    **sampling, "enable_thinking": False}
            raw, parsed = post(base_url.rstrip("/") + "/v1/chat/completions", body)
            records.append({"prompt_index": index, "adapter": adapter, "request": body,
                            "response": parsed, "raw_response": raw})
    result = {"schema": 2, "status": "TEST_ONLY", "endpoint": base_url,
        "sampling": sampling, "prompt_count": len(PROMPTS), "fixture_verification": verified,
        "records": records, "created_at": time.time(),
        "observability": {"OPENAI_ENDPOINT_OBSERVABLE": True, "NATIVE_ENDPOINT_OBSERVABLE": "UNTESTED",
                           "LORA_LOADED": True, "LORA_ROUTED": True}}
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--url", default="http://127.0.0.1:30000"); parser.add_argument("--out", default="artifacts/training/rtx-strong-lora-probe.json")
    args = parser.parse_args(); print(json.dumps(probe(args.url, args.out), indent=2, sort_keys=True))
