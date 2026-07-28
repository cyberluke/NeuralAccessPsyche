from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


payload = {
    "model": "nram-qwen3-14b-awq",
    "messages": [{"role": "user", "content": "Reply one word."}],
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 64,
    "seed": 424242,
    "stream": False,
    "nram": {
        "request_id": "RVR-SERVER-DEADLINE",
        "include_telemetry": True,
        "telemetry_max_steps": 128,
        "telemetry_top_k": 1,
        "prompt_steering_enabled": False,
        "profile_logit_steering_enabled": False,
        "hard_token_schedule": [9702] * 64,
    },
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Authorization": "Bearer dev-nram-key", "Content-Type": "application/json"},
    method="POST",
)
started = time.time()
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        status = response.status
        raw = response.read()
except urllib.error.HTTPError as exc:
    status = exc.code
    raw = exc.read()
result = {
    "status": status,
    "elapsed_seconds": time.time() - started,
    "body": raw.decode("utf-8", "replace"),
    "request": payload,
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds", "body")}))
