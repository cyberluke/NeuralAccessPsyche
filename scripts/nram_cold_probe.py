"""Single-request cold-server TEST_ONLY probe for pinned SGLang LoRA."""
import argparse, json
from pathlib import Path
from urllib.request import Request, urlopen

PROMPT = "The meeting starts at noon."

def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--out", required=True)
    args = p.parse_args()
    body = {"model": args.model, "messages": [{"role": "user", "content": PROMPT}], "stream": False,
            "max_tokens": 1, "temperature": 0, "top_p": 1, "top_k": 1,
            "logprobs": True, "top_logprobs": 5, "enable_thinking": False}
    request = Request("http://127.0.0.1:30000/v1/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        raw = response.read().decode()
    result = {"schema": 1, "status": "TEST_ONLY", "model": args.model, "request": body,
              "raw_response": raw, "response": json.loads(raw)}
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__": main()
