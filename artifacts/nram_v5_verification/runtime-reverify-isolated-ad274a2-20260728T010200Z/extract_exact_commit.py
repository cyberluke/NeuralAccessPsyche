from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def main() -> int:
    target = sys.argv[1]
    destination = Path(sys.argv[2]).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    raw = git("ls-tree", "-rz", "--full-tree", target)
    count = 0
    for record in raw.split(b"\0"):
        if not record:
            continue
        header, path_bytes = record.split(b"\t", 1)
        mode_bytes, type_bytes, oid_bytes = header.split(b" ", 2)
        mode = mode_bytes.decode()
        object_type = type_bytes.decode()
        oid = oid_bytes.decode()
        if object_type != "blob":
            raise RuntimeError(f"Unsupported tree object {object_type}: {path_bytes!r}")
        relative = path_bytes.decode("utf-8", "strict")
        output = destination.joinpath(*relative.split("/"))
        output.parent.mkdir(parents=True, exist_ok=True)
        data = git("cat-file", "blob", oid)
        if mode == "120000":
            os.symlink(data.decode("utf-8"), output)
        else:
            output.write_bytes(data)
            if mode == "100755":
                output.chmod(output.stat().st_mode | 0o111)
        count += 1
    print(f"EXTRACTED_BLOBS={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
