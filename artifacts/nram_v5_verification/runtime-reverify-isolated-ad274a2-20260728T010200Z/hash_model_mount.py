from __future__ import annotations

import hashlib
import json
from pathlib import Path


root = Path("/models")
rows = []
aggregate = hashlib.sha256()
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    relative = path.relative_to(root).as_posix()
    hexdigest = digest.hexdigest()
    rows.append({"path": relative, "bytes": size, "sha256": hexdigest})
    aggregate.update(
        relative.encode()
        + b"\0"
        + str(size).encode()
        + b"\0"
        + hexdigest.encode()
        + b"\n"
    )

tokenizer_names = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
}
tokenizer_rows = [row for row in rows if row["path"] in tokenizer_names]
tokenizer_aggregate = hashlib.sha256()
for row in tokenizer_rows:
    tokenizer_aggregate.update(
        str(row["path"]).encode()
        + b"\0"
        + str(row["bytes"]).encode()
        + b"\0"
        + str(row["sha256"]).encode()
        + b"\n"
    )

print(
    json.dumps(
        {
            "snapshot": "31c69efc29464b6bb0aee1398b5a7b50a99340c3",
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "canonical_manifest_sha256": aggregate.hexdigest(),
            "tokenizer_canonical_manifest_sha256": tokenizer_aggregate.hexdigest(),
            "tokenizer_files": tokenizer_rows,
            "files": rows,
        },
        indent=2,
    )
)
