"""Generate A/B test outputs from all NRAM consciousness states and MoE orchestrator."""
import requests
import json
import os

API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer test-key"
}

# Same prompt as the ChatGPT strategic document
PROMPT = "Analyzuj budoucnost AI v healthcare pro rok 2030. Zahrň konkrétní příklady, data, trendy, doporučení a strategickou syntézu."

OUTPUT_DIR = "d:/_SATIN_AI/NeuralAccessPsyche/ab_test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_output(model_name, filename, max_tokens=2048, temperature=0.7):
    """Generate output and save to file."""
    print(f"\n{'='*60}")
    print(f"Generating: {model_name}")
    print(f"{'='*60}")
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        
        # Save to file
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Model: {model_name}\n")
            f.write(f"# Prompt: {PROMPT}\n")
            f.write(f"# Max tokens: {max_tokens}\n")
            f.write(f"# Temperature: {temperature}\n")
            f.write(f"# Usage: {json.dumps(usage)}\n\n")
            f.write(content)
        
        print(f"✓ Saved to {filename}")
        print(f"  Length: {len(content)} chars")
        print(f"  Tokens: {usage.get('completion_tokens', 'N/A')}")
        print(f"  First 200 chars: {content[:200]}...")
        
        return content
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

# Generate outputs from all consciousness states
states = [
    ("persona-normal", "01_normal_state.md"),
    ("persona-microdose", "02_microdose_state.md"),
    ("persona-threshold", "03_threshold_state.md"),
    ("persona-psychedelic", "04_psychedelic_state.md"),
    ("persona-peak", "05_peak_state.md"),
    ("persona-dissociative", "06_dissociative_state.md"),
]

print("Generating consciousness state outputs...")
for model, filename in states:
    generate_output(model, filename, max_tokens=2048, temperature=0.7)

# Generate MoE orchestrator output
print("\n" + "="*60)
print("Generating MoE orchestrator output...")
print("="*60)
generate_output("nram-moe-orchestrator", "07_moe_orchestrator.md", max_tokens=4096, temperature=0.7)

print(f"\n{'='*60}")
print("All outputs saved to:")
print(f"  {OUTPUT_DIR}")
print(f"{'='*60}")
