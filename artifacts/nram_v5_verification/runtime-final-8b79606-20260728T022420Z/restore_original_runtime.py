from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import urllib.request

OUT = Path(__file__).resolve().parent


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=120)


def health(name: str) -> str:
    result = run(
        "docker", "inspect", "--format",
        "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}",
        name,
    )
    return result.stdout.strip()


def wait_health(name: str, seconds: int) -> list[dict[str, object]]:
    deadline = time.monotonic() + seconds
    rows = []
    while time.monotonic() < deadline:
        value = health(name)
        rows.append({"monotonic": time.monotonic(), "health": value})
        if value == "healthy":
            return rows
        time.sleep(5)
    raise RuntimeError(f"{name} did not become healthy: {rows[-5:]}")


exact = run(
    "docker", "inspect", "--format",
    "{{.Name}}|{{.Id}}|{{.Image}}|{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}",
    "nram-api", "nram-sglang",
).stdout
(OUT / "isolated-final-runtime-state.txt").write_text(exact, encoding="utf-8")

removed = run("docker", "rm", "-f", "nram-api", "nram-sglang").stdout
(OUT / "isolated-stop.txt").write_text(removed, encoding="utf-8")
run("docker", "rename", "nram-original-api-20260728t022420z", "nram-api")
run("docker", "rename", "nram-original-sglang-20260728t022420z", "nram-sglang")

run("docker", "start", "nram-sglang")
sglang_poll = wait_health("nram-sglang", 900)
run("docker", "start", "nram-api")
api_poll = wait_health("nram-api", 300)
with urllib.request.urlopen("http://127.0.0.1:8000/ready", timeout=15) as response:
    ready_status = response.status
    ready_body = response.read().decode("utf-8", errors="replace")

identity = run(
    "docker", "inspect", "--format",
    "{{.Name}}|{{.Id}}|{{.Image}}|{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}",
    "nram-api", "nram-sglang",
).stdout
result = {
    "sglang_poll": sglang_poll,
    "api_poll": api_poll,
    "ready_status": ready_status,
    "ready_body": ready_body,
    "identity": identity.splitlines(),
    "restored": ready_status == 200 and health("nram-api") == "healthy" and health("nram-sglang") == "healthy",
}
(OUT / "original-restore-health.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: result[key] for key in ("ready_status", "ready_body", "identity", "restored")}, indent=2))
raise SystemExit(0 if result["restored"] else 2)
