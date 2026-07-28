#!/usr/bin/env python3
"""Healthcheck for the SGLang server.

Hits the local SGLang /health endpoint and exits 0 on success, 1 on failure.
Used both by the Dockerfile HEALTHCHECK and by docker compose healthchecks.
Only depends on the Python standard library so it works inside the SGLang
image without extra dependencies.
"""
import sys
import urllib.error
import urllib.request

HEALTH_URL = "http://127.0.0.1:30000/health"
TIMEOUT_SECONDS = 8


def main() -> int:
    try:
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            status = getattr(resp, "status", resp.getcode())
            if 200 <= status < 400:
                return 0
            print(f"[nram-healthcheck] unhealthy: HTTP {status}", file=sys.stderr)
            return 1
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f"[nram-healthcheck] unhealthy: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - healthcheck must never crash
        print(f"[nram-healthcheck] unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
