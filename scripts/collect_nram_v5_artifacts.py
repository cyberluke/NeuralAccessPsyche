"""Collect an exact, fail-loud NRAM v5 provenance manifest.

The collector intentionally has no repository, image, model, or run identity
constants. Identity is measured from Git, Docker, artifact bytes, and explicit
run inputs at collection time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Sequence


class CollectionError(RuntimeError):
    """A required exact-run identity could not be established."""


def run(
    *args: str,
    cwd: Path | None = None,
    check: bool = True,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        command = subprocess.list2cmdline(args)
        raise CollectionError(
            f"command failed ({result.returncode}): {command}\n{result.stderr.strip()}"
        )
    return result


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _git(repo: Path, *args: str, check: bool = True) -> str:
    return run("git", *args, cwd=repo, check=check).stdout.strip()


def _status_path(line: str) -> str:
    value = line[3:] if len(line) > 3 else ""
    if " -> " in value:
        value = value.split(" -> ", 1)[1]
    return value.strip('"').replace("\\", "/")


def _matches_path(path: str, selectors: Sequence[str]) -> bool:
    return any(path == item or path.startswith(item.rstrip("/") + "/") for item in selectors)


def collect_git_identity(
    repo: Path,
    *,
    implementation_commit: str | None = None,
    excluded_paths: Sequence[str] = (),
) -> dict[str, Any]:
    """Measure commit/tree/branch and partition porcelain dirty state."""
    head = _git(repo, "rev-parse", "HEAD")
    implementation = _git(repo, "rev-parse", implementation_commit or head)
    implementation_tree = _git(repo, "rev-parse", f"{implementation}^{{tree}}")
    head_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    branch_result = run(
        "git", "symbolic-ref", "--quiet", "--short", "HEAD", cwd=repo, check=False
    )
    detached = branch_result.returncode != 0
    branch = None if detached else branch_result.stdout.strip()

    porcelain_lines = _git(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    normalized_exclusions = tuple(item.replace("\\", "/").rstrip("/") for item in excluded_paths)
    included_status: list[str] = []
    excluded_status: list[str] = []
    for line in porcelain_lines:
        target = excluded_status if _matches_path(_status_path(line), normalized_exclusions) else included_status
        target.append(line)

    return {
        "implementation_commit": implementation,
        "implementation_tree_oid": implementation_tree,
        "checkout_head_commit": head,
        "checkout_head_tree_oid": head_tree,
        "branch": branch,
        "detached": detached,
        "dirty": bool(porcelain_lines),
        "porcelain_v1": porcelain_lines,
        "included_dirty_status": included_status,
        "excluded_dirty_status": excluded_status,
        "excluded_path_selectors": list(normalized_exclusions),
    }


def git_blob_bytes(repo: Path, commit: str, path: str) -> tuple[str, bytes]:
    oid = _git(repo, "rev-parse", f"{commit}:{path}")
    result = subprocess.run(
        ("git", "cat-file", "blob", oid),
        cwd=repo,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise CollectionError(
            f"cannot read Git blob {commit}:{path}: {result.stderr.decode(errors='replace')}"
        )
    return oid, result.stdout


def collect_byte_domains(repo: Path, commit: str, path: str) -> dict[str, Any]:
    """Record immutable Git bytes, checkout bytes, and explicit LF normalization."""
    oid, blob = git_blob_bytes(repo, commit, path)
    checkout_path = repo / path
    if not checkout_path.is_file():
        raise CollectionError(f"required checkout file is missing: {path}")
    checkout = checkout_path.read_bytes()
    normalized = checkout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return {
        "path": path.replace("\\", "/"),
        "git_blob_oid_sha1": oid,
        "git_blob_sha256": sha256_bytes(blob),
        "git_blob_bytes": len(blob),
        "checkout_bytes_sha256": sha256_bytes(checkout),
        "checkout_bytes": len(checkout),
        "lf_normalized_checkout_sha256": sha256_bytes(normalized),
        "lf_normalized_checkout_bytes": len(normalized),
        "byte_domain_definitions": {
            "git_blob_sha256": "SHA-256 over bytes returned by git cat-file blob",
            "checkout_bytes_sha256": "SHA-256 over unmodified filesystem bytes",
            "lf_normalized_checkout_sha256": (
                "SHA-256 after CRLF and lone CR bytes are normalized to LF"
            ),
        },
    }


def collect_spec_identity(repo: Path, path: str) -> dict[str, Any]:
    commit = _git(repo, "log", "-1", "--format=%H", "--", path)
    if not commit:
        raise CollectionError(f"cannot derive canonical specification commit for {path}")
    return {
        "canonical_specification_commit": commit,
        **collect_byte_domains(repo, commit, path),
    }


def _parse_compose_ps(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        raise CollectionError("docker compose ps returned no running containers")
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        records = []
        for line in raw.splitlines():
            records.append(json.loads(line))
        return records


def collect_container_identity(repo: Path) -> list[dict[str, Any]]:
    records = _parse_compose_ps(
        run("docker", "compose", "ps", "--format", "json", cwd=repo).stdout
    )
    output = []
    for record in records:
        container_id = record.get("ID") or record.get("Id")
        if not container_id:
            raise CollectionError("compose container record has no container ID")
        inspect = json.loads(run("docker", "inspect", container_id).stdout)[0]
        image_id = inspect.get("Image")
        image_ref = (inspect.get("Config") or {}).get("Image")
        image_inspect = json.loads(run("docker", "image", "inspect", image_id).stdout)[0]
        output.append(
            {
                "service": record.get("Service"),
                "container_name": inspect.get("Name", "").lstrip("/"),
                "container_id": inspect.get("Id"),
                "image_reference": image_ref,
                "image_id": image_id,
                "repo_digests": image_inspect.get("RepoDigests") or [],
                "created": inspect.get("Created"),
            }
        )
    return output


def _parse_mapping(values: Iterable[str], option: str) -> list[tuple[str, str]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise CollectionError(f"{option} requires SERVICE=PATH, got {value!r}")
        service, path = value.split("=", 1)
        if not service or not path:
            raise CollectionError(f"{option} requires non-empty SERVICE=PATH")
        parsed.append((service, path))
    return parsed


def _parse_source_mapping(value: str) -> tuple[str, str, str]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise CollectionError(
            "--container-source requires SERVICE=CONTAINER_FILE=REPOSITORY_FILE"
        )
    return parts[0], parts[1], parts[2].replace("\\", "/")


_REMOTE_TREE_HASH = r"""
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
if not root.exists(): raise SystemExit('missing artifact path: '+str(root))
files=[]
paths=[root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file())
for path in paths:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 rel=path.name if root.is_file() else path.relative_to(root).as_posix()
 files.append({'path':rel,'bytes':path.stat().st_size,'sha256':h.hexdigest()})
manifest=json.dumps(files,sort_keys=True,separators=(',',':')).encode()
print(json.dumps({'root':str(root),'file_count':len(files),'manifest_sha256':hashlib.sha256(manifest).hexdigest(),'files':files},sort_keys=True))
"""


def collect_remote_hash(repo: Path, service: str, path: str, timeout: int = 3600) -> dict[str, Any]:
    raw = run(
        "docker", "compose", "exec", "-T", service,
        "python", "-c", _REMOTE_TREE_HASH, path,
        cwd=repo,
        timeout=timeout,
    ).stdout
    try:
        return {"service": service, **json.loads(raw)}
    except json.JSONDecodeError as exc:
        raise CollectionError(f"invalid hash output from service {service}: {raw[:500]}") from exc


def collect_authenticated_source(
    repo: Path,
    implementation_commit: str,
    mapping: tuple[str, str, str],
) -> dict[str, Any]:
    """Authenticate one container source file against implementation Git bytes."""
    service, container_path, repository_path = mapping
    remote = collect_remote_hash(repo, service, container_path)
    files = remote.get("files") or []
    if remote.get("file_count") != 1 or len(files) != 1:
        raise CollectionError(
            f"container source mapping must identify one file: {service}={container_path}"
        )
    git_domain = collect_byte_domains(repo, implementation_commit, repository_path)
    if files[0].get("sha256") != git_domain["git_blob_sha256"]:
        raise CollectionError(
            "container source does not match implementation Git blob: "
            f"{service}:{container_path} != {implementation_commit}:{repository_path}"
        )
    return {
        "service": service,
        "container_path": container_path,
        "repository_path": repository_path,
        "container_file_sha256": files[0]["sha256"],
        "container_file_bytes": files[0]["bytes"],
        "implementation_git_blob_oid_sha1": git_domain["git_blob_oid_sha1"],
        "implementation_git_blob_sha256": git_domain["git_blob_sha256"],
        "authenticated_equal": True,
    }


_RUNTIME_IDENTITY = r"""
import importlib.metadata,json,os,pathlib,platform,subprocess
import torch
import sglang
root=pathlib.Path(sglang.__file__).resolve().parents[2]
git=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
print(json.dumps({
 'python':platform.python_version(),
 'sglang_version':getattr(sglang,'__version__',importlib.metadata.version('sglang')),
 'sglang_source_root':str(root),
 'sglang_source_commit':git.stdout.strip() if git.returncode==0 else None,
 'dill_version':importlib.metadata.version('dill'),
 'pytorch_version':torch.__version__,
 'torch_cuda_version':torch.version.cuda,
 'cuda_available':torch.cuda.is_available(),
},sort_keys=True))
"""


def collect_runtime_identity(repo: Path, sglang_service: str) -> dict[str, Any]:
    raw = run(
        "docker", "compose", "exec", "-T", sglang_service,
        "python", "-c", _RUNTIME_IDENTITY,
        cwd=repo,
    ).stdout
    value = json.loads(raw)
    required = ("python", "sglang_version", "dill_version", "pytorch_version", "torch_cuda_version")
    missing = [name for name in required if not value.get(name)]
    if missing:
        raise CollectionError("runtime identity is incomplete: " + ", ".join(missing))
    if not value.get("sglang_source_commit"):
        raise CollectionError("SGLang source commit cannot be established")
    return value


def collect_launch_command(repo: Path, service: str) -> str:
    script = "import pathlib;print(pathlib.Path('/proc/1/cmdline').read_bytes().replace(b'\\0',b' ').decode())"
    launch = run(
        "docker", "compose", "exec", "-T", service, "python", "-c", script, cwd=repo
    ).stdout.strip()
    if not launch:
        raise CollectionError("SGLang launch command is empty")
    return launch


def load_required_json(path: Path, description: str) -> Any:
    if not path.is_file():
        raise CollectionError(f"missing required {description}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CollectionError(f"invalid {description}: {path}: {exc}") from exc


def collect_junit(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths:
        raise CollectionError("at least one --junit result is required")
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    nodes: list[str] = []
    files = []
    for path in paths:
        if not path.is_file():
            raise CollectionError(f"missing required JUnit file: {path}")
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        for suite in suites:
            for key in totals:
                totals[key] += int(suite.attrib.get(key, 0))
        for case in root.iter("testcase"):
            nodes.append(f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}")
        files.append({"path": str(path), "sha256": sha256(path)})
    return {"counts": totals, "testcase_nodes": nodes, "junit_files": files}


def validate_manifest_schema(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version", "run_id", "implementation_commit", "implementation_tree_oid",
        "evidence_commit", "git", "specification", "environment", "containers",
        "source_hashes", "model_artifact", "tokenizer_artifact", "runtime",
        "launch_command", "configuration_sha256", "run_inputs", "commands",
        "tests", "artifacts",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise CollectionError("manifest schema missing fields: " + ", ".join(missing))
    for name in ("implementation_commit", "implementation_tree_oid"):
        value = manifest[name]
        if not isinstance(value, str) or len(value) != 40:
            raise CollectionError(f"manifest {name} is not a full Git OID")
    evidence = manifest["evidence_commit"]
    if evidence is not None and (not isinstance(evidence, str) or len(evidence) != 40):
        raise CollectionError("manifest evidence_commit is not null or a full Git OID")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--implementation-commit")
    parser.add_argument("--evidence-commit")
    parser.add_argument("--spec-path", required=True)
    parser.add_argument("--exclude-dirty-path", action="append", default=[])
    parser.add_argument("--sglang-service", required=True)
    parser.add_argument(
        "--container-source",
        action="append",
        required=True,
        help="SERVICE=CONTAINER_FILE=REPOSITORY_FILE (repeat for every executed source)",
    )
    parser.add_argument("--model-artifact", required=True, help="SERVICE=PATH")
    parser.add_argument("--tokenizer-artifact", required=True, help="SERVICE=PATH")
    parser.add_argument("--run-metadata", required=True, type=Path)
    parser.add_argument("--command-results", required=True, type=Path)
    parser.add_argument("--test-node-list", required=True, type=Path)
    parser.add_argument("--junit", action="append", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo.resolve()
    root = args.artifact_dir.resolve()
    if not (repo / ".git").exists():
        raise CollectionError(f"repository has no .git directory: {repo}")

    run_metadata = load_required_json(args.run_metadata, "run metadata")
    required_run_keys = {"prompt_ids", "seeds", "intervention_ids", "configuration"}
    missing_run_keys = sorted(required_run_keys - set(run_metadata))
    if missing_run_keys:
        raise CollectionError("run metadata missing fields: " + ", ".join(missing_run_keys))
    commands = load_required_json(args.command_results, "command results")
    if not isinstance(commands, list) or any(
        not isinstance(item, dict) or "command" not in item or "exit_code" not in item
        for item in commands
    ):
        raise CollectionError("command results must be a list of command/exit_code objects")
    if not args.test_node_list.is_file():
        raise CollectionError(f"missing exact tracked test node list: {args.test_node_list}")

    git = collect_git_identity(
        repo,
        implementation_commit=args.implementation_commit,
        excluded_paths=args.exclude_dirty_path,
    )
    evidence_commit = None
    if args.evidence_commit:
        evidence_commit = _git(repo, "rev-parse", args.evidence_commit)

    compose_config = run("docker", "compose", "config", cwd=repo).stdout
    if not compose_config:
        raise CollectionError("resolved Docker Compose configuration is empty")
    containers = collect_container_identity(repo)
    source_mappings = [_parse_source_mapping(value) for value in args.container_source]
    model_mapping = _parse_mapping([args.model_artifact], "--model-artifact")[0]
    tokenizer_mapping = _parse_mapping([args.tokenizer_artifact], "--tokenizer-artifact")[0]

    source_hashes = [
        collect_authenticated_source(repo, git["implementation_commit"], mapping)
        for mapping in source_mappings
    ]
    model_artifact = collect_remote_hash(repo, *model_mapping)
    tokenizer_artifact = collect_remote_hash(repo, *tokenizer_mapping)
    runtime = collect_runtime_identity(repo, args.sglang_service)
    launch_command = collect_launch_command(repo, args.sglang_service)
    tests = collect_junit(args.junit)
    node_list_bytes = args.test_node_list.read_bytes()
    tests["tracked_node_list"] = {
        "path": str(args.test_node_list),
        "sha256": sha256_bytes(node_list_bytes),
        "nodes": [line for line in node_list_bytes.decode("utf-8").splitlines() if "::" in line],
    }

    root.mkdir(parents=True, exist_ok=False)
    environment = {
        "host_os": platform.platform(),
        "host_python": sys.version,
        "git": git,
        "specification": collect_spec_identity(repo, args.spec_path),
        "containers": containers,
        "runtime": runtime,
    }
    (root / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    (root / "server-launch.txt").write_text(launch_command + "\n", encoding="utf-8", newline="\n")
    (root / "compose-config.txt").write_text(compose_config, encoding="utf-8", newline="\n")
    (root / "command-results.json").write_text(
        json.dumps(commands, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    (root / "test-node-list.txt").write_bytes(node_list_bytes)

    files = [path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"]
    specification = collect_spec_identity(repo, args.spec_path)
    manifest = {
        "schema_version": "nram.v5.evidence-manifest.v2",
        "run_id": root.name,
        "implementation_commit": git["implementation_commit"],
        "implementation_tree_oid": git["implementation_tree_oid"],
        "evidence_commit": evidence_commit,
        "git": git,
        "specification": specification,
        "environment": {"host_os": platform.platform(), "host_python": sys.version},
        "containers": containers,
        "source_hashes": source_hashes,
        "model_artifact": model_artifact,
        "tokenizer_artifact": tokenizer_artifact,
        "runtime": runtime,
        "launch_command": launch_command,
        "configuration_sha256": sha256_bytes(compose_config.encode("utf-8")),
        "run_inputs": {
            **run_metadata,
            "configuration_sha256": canonical_json_sha256(run_metadata["configuration"]),
        },
        "commands": commands,
        "tests": tests,
        "artifacts": {
            path.relative_to(root).as_posix(): sha256(path) for path in sorted(files)
        },
    }
    validate_manifest_schema(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"artifact_dir": str(root), "manifest": "manifest.json"}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CollectionError as exc:
        print(f"artifact collection failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
