import os
path = os.path.join("docs", "PHASE12_RELEASE_GATE_TRUE.md")
with open(path, "w", encoding="utf-8") as out:
    out.write(open("_report_content.txt", encoding="utf-8").read())
print("Written", os.path.getsize(path), "bytes")
