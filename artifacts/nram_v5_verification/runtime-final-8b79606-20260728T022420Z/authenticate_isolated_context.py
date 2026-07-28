from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    evidence = Path(sys.argv[2]).resolve()
    target = sys.argv[3]
    raw = git("ls-tree", "-rz", "--full-tree", target)
    entries: list[tuple[str, str, str, str]] = []
    expected: set[str] = set()
    failures: list[dict[str, object]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        header, path_bytes = record.split(b"\t", 1)
        mode_bytes, type_bytes, oid_bytes = header.split(b" ", 2)
        mode = mode_bytes.decode()
        object_type = type_bytes.decode()
        oid = oid_bytes.decode()
        path = path_bytes.decode("utf-8", "surrogateescape")
        entries.append((mode, object_type, oid, path))
        if object_type == "blob":
            expected.add(path)

    manifest: list[dict[str, object]] = []
    for mode, object_type, oid, path in entries:
        if object_type != "blob":
            manifest.append(
                {
                    "path": path,
                    "mode": mode,
                    "type": object_type,
                    "git_oid": oid,
                    "status": "non_blob",
                }
            )
            continue
        extracted_path = root.joinpath(*path.split("/"))
        if not extracted_path.exists() and not extracted_path.is_symlink():
            failures.append({"path": path, "error": "missing"})
            continue
        if mode == "120000" and extracted_path.is_symlink():
            data = os.readlink(extracted_path).encode()
        else:
            data = extracted_path.read_bytes()
        actual_oid = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        row: dict[str, object] = {
            "path": path,
            "mode": mode,
            "type": object_type,
            "git_oid": oid,
            "extracted_git_oid": actual_oid,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "match": actual_oid == oid,
        }
        manifest.append(row)
        if actual_oid != oid:
            failures.append(
                {
                    "path": path,
                    "error": "oid_mismatch",
                    "expected": oid,
                    "actual": actual_oid,
                }
            )

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extra:
        failures.append({"error": "extra_paths", "paths": extra})
    if missing:
        failures.append({"error": "missing_paths", "paths": missing})

    canonical = hashlib.sha256()
    blob_rows = (row for row in manifest if row.get("type") == "blob")
    for row in sorted(blob_rows, key=lambda item: str(item["path"])):
        canonical.update(
            str(row["mode"]).encode()
            + b"\0"
            + str(row["path"]).encode()
            + b"\0"
            + str(row["sha256"]).encode()
            + b"\n"
        )
    summary = {
        "target_commit": target,
        "target_tree": git("rev-parse", f"{target}^{{tree}}").decode().strip(),
        "entry_count": len(entries),
        "blob_count": len(expected),
        "actual_file_count": len(actual),
        "extra_paths": extra,
        "missing_paths": missing,
        "failure_count": len(failures),
        "authenticated": not failures,
        "canonical_path_mode_content_sha256": canonical.hexdigest(),
        "method": (
            "direct Git-object extraction from target git ls-tree plus git cat-file; "
            "every extracted byte re-hashed as a Git blob and compared with the "
            "target tree object ID; the separately retained git archive is provenance "
            "evidence but was not used after Windows tar changed byte/name domains"
        ),
    }
    (evidence / "isolated-context-files.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (evidence / "isolated-context-authentication.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (evidence / "isolated-context-authentication-failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
