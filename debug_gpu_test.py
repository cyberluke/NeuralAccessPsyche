#!/usr/bin/env python3
"""Debug script to trace GPU test event emission."""
import asyncio
import json
import subprocess
import time
import uuid

import httpx


async def main():
    # Generate unique request ID
    request_id = f"debug-{uuid.uuid4().hex[:12]}"
    print(f"Test request_id: {request_id}")
    
    # Send request
    body = {
        "model": "nram-qwen3-14b-awq",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 2,
        "temperature": 0.7,
        "seed": 42,
        "nram": {
            "enabled": True,
            "profile": "normal",
            "request_id": request_id,
            "include_telemetry": True,
            "telemetry_max_steps": 2,
        },
    }
    
    print("Sending request...")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json=body,
            headers={"Authorization": "Bearer dev-nram-key"},
        )
    
    print(f"Response status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return
    
    result = response.json()
    print(f"Generated text: {result['choices'][0]['message']['content']}")
    
    # Wait for logs to flush
    print("Waiting 3 seconds for logs to flush...")
    time.sleep(3)
    
    # Query docker logs
    print("Querying docker logs...")
    output = subprocess.run(
        ["docker", "logs", "nram-sglang", "--tail", "500"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    # Find all NRAM_PROCESSOR_EVENT lines
    all_events = []
    for line in (output.stdout + output.stderr).splitlines():
        if "NRAM_PROCESSOR_EVENT " in line:
            try:
                event = json.loads(line.split("NRAM_PROCESSOR_EVENT ", 1)[1])
                all_events.append(event)
            except json.JSONDecodeError as e:
                print(f"Failed to parse event: {e}")
                print(f"Line: {line[:200]}")
    
    print(f"\nTotal events found: {len(all_events)}")
    
    # Find events matching our request_id
    matching_events = [e for e in all_events if e.get("request_id") == request_id]
    print(f"Events matching request_id '{request_id}': {len(matching_events)}")
    
    if matching_events:
        print("\nFirst matching event:")
        print(json.dumps(matching_events[0], indent=2))
    else:
        print("\nNo matching events found!")
        print("\nLast 5 events in logs:")
        for event in all_events[-5:]:
            print(f"  request_id: {event.get('request_id')}, step: {event.get('step')}")


if __name__ == "__main__":
    asyncio.run(main())
