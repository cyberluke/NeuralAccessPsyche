#!/usr/bin/env python3
"""
Test if NRAM hooks are actually invoked during inference.
This script makes a request and checks if the hooks are being called
by looking for telemetry events or log messages.
"""
import requests
import json
import time

print("=" * 70)
print("Testing Hook Invocation During Inference")
print("=" * 70)

# Make a request to the SGLang server
url = "http://localhost:30000/v1/chat/completions"
payload = {
    "model": "nram-qwen3-14b-awq",
    "messages": [
        {"role": "user", "content": "Say hello"}
    ],
    "max_tokens": 10,
    "temperature": 0.7
}

print("\nMaking inference request...")
start_time = time.time()
response = requests.post(url, json=payload, timeout=30)
elapsed = time.time() - start_time

print(f"Response received in {elapsed:.2f}s")
print(f"Status code: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"Response: {result['choices'][0]['message']['content']}")
    print(f"Tokens: {result['usage']['total_tokens']}")
else:
    print(f"Error: {response.text}")

print("\n" + "=" * 70)
print("Checking for hook invocation evidence...")
print("=" * 70)

# Check container logs for hook invocation messages
import subprocess
result = subprocess.run(
    ["docker", "logs", "nram-sglang", "--tail", "100"],
    capture_output=True,
    text=True
)

logs = result.stdout + result.stderr
hook_lines = [line for line in logs.split('\n') if 'hook' in line.lower() or 'nram' in line.lower()]

if hook_lines:
    print(f"\nFound {len(hook_lines)} hook-related log lines:")
    for line in hook_lines[-10:]:  # Show last 10 lines
        print(f"  {line}")
else:
    print("\nNo hook-related log lines found in recent logs.")
    print("This could mean:")
    print("  1. Hooks are not being invoked")
    print("  2. Hooks don't log at INFO level")
    print("  3. Hooks are invoked but don't produce log output")

print("\n" + "=" * 70)
print("Test complete")
print("=" * 70)
