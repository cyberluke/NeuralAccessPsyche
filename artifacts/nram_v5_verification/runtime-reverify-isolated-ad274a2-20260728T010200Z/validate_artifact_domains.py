from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


base = Path(sys.argv[1])
output = Path(sys.argv[2])
manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
rows = []
for relative, declared in manifest["artifacts"].items():
    data = (base / relative).read_bytes()
    git_blob_domain = hashlib.sha256(data).hexdigest()
    windows_data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    windows_crlf_domain = hashlib.sha256(windows_data).hexdigest()
    domain = (
        "git_blob"
        if declared == git_blob_domain
        else "windows_crlf"
        if declared == windows_crlf_domain
        else "neither"
    )
    rows.append(
        {
            "path": relative,
            "declared": declared,
            "git_blob_domain": git_blob_domain,
            "windows_crlf_domain": windows_crlf_domain,
            "domain": domain,
        }
    )
result = {"counts": dict(Counter(row["domain"] for row in rows)), "rows": rows}
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"counts": result["counts"], "neither": [row["path"] for row in rows if row["domain"] == "neither"]}, indent=2))
