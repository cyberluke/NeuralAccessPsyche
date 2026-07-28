from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


API_IMAGE = "nram-rvr-ad274a2-api:20260728t010200z"
SGLANG_IMAGE = "nram-rvr-ad274a2-sglang:20260728t010200z"


def run(*args: str) -> bytes:
    return subprocess.check_output(args)


def image_files(image: str, python: str, roots: list[str]) -> dict[str, dict[str, object]]:
    code = r'''import hashlib,json,pathlib,sys
out={}
for root_s in sys.argv[1:]:
 root=pathlib.Path(root_s)
 paths=[root] if root.is_file() else root.rglob("*")
 for p in paths:
  if p.is_file():
   b=p.read_bytes(); out[p.as_posix()]={"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)}
print(json.dumps(out,sort_keys=True))'''
    raw = run(
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        python,
        image,
        "-c",
        code,
        *roots,
    )
    return json.loads(raw)


def compare(
    actual: dict[str, dict[str, object]],
    source_root: Path,
    mappings: list[tuple[str, str]],
) -> dict[str, object]:
    ignored_generated = sorted(
        path
        for path in actual
        if "/__pycache__/" in path and path.endswith(".pyc")
    )
    actual = {
        path: value for path, value in actual.items() if path not in ignored_generated
    }
    expected: dict[str, dict[str, object]] = {}
    for source_name, image_name in mappings:
        source = source_root / source_name
        paths = [source] if source.is_file() else source.rglob("*")
        for path in paths:
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix() if source.is_dir() else ""
            image_path = image_name.rstrip("/") + ("/" + relative if relative else "")
            data = path.read_bytes()
            expected[image_path] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "source_path": path.relative_to(source_root).as_posix(),
            }
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatches = []
    for path in sorted(set(expected) & set(actual)):
        if expected[path]["sha256"] != actual[path]["sha256"]:
            mismatches.append(
                {"image_path": path, "expected": expected[path], "actual": actual[path]}
            )
    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing": missing,
        "extra": extra,
        "mismatches": mismatches,
        "ignored_generated": ignored_generated,
        "authenticated": not missing and not extra and not mismatches,
        "expected": expected,
        "actual": actual,
    }


def main() -> int:
    source_root = Path(sys.argv[1]).resolve()
    evidence = Path(sys.argv[2]).resolve()
    api_actual = image_files(
        API_IMAGE,
        "python",
        [
            "/app/pyproject.toml",
            "/app/main.py",
            "/app/api",
            "/app/core",
            "/app/nram_sglang",
            "/app/utils",
            "/app/evaluation",
            "/app/templates",
            "/app/static",
        ],
    )
    sglang_actual = image_files(
        SGLANG_IMAGE,
        "python3",
        [
            "/opt/nram/entrypoint.sh",
            "/opt/nram/healthcheck.py",
            "/opt/nram_python/nram_sglang",
        ],
    )
    api = compare(
        api_actual,
        source_root,
        [
            ("pyproject.toml", "/app/pyproject.toml"),
            ("main.py", "/app/main.py"),
            ("api", "/app/api"),
            ("core", "/app/core"),
            ("nram_sglang", "/app/nram_sglang"),
            ("utils", "/app/utils"),
            ("evaluation", "/app/evaluation"),
            ("templates", "/app/templates"),
            ("static", "/app/static"),
        ],
    )
    sglang = compare(
        sglang_actual,
        source_root,
        [
            ("inference/sglang/entrypoint.sh", "/opt/nram/entrypoint.sh"),
            ("inference/sglang/healthcheck.py", "/opt/nram/healthcheck.py"),
            ("nram_sglang", "/opt/nram_python/nram_sglang"),
        ],
    )
    image_inspect = json.loads(
        run("docker", "image", "inspect", API_IMAGE, SGLANG_IMAGE)
    )
    result = {
        "api_image": API_IMAGE,
        "sglang_image": SGLANG_IMAGE,
        "api": api,
        "sglang": sglang,
        "image_inspect": image_inspect,
        "authenticated": bool(api["authenticated"] and sglang["authenticated"]),
    }
    (evidence / "image-source-authentication.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    summary = {
        "api_image_id": image_inspect[0]["Id"],
        "sglang_image_id": image_inspect[1]["Id"],
        "api_authenticated": api["authenticated"],
        "sglang_authenticated": sglang["authenticated"],
        "api_expected_count": api["expected_count"],
        "sglang_expected_count": sglang["expected_count"],
        "authenticated": result["authenticated"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if result["authenticated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
