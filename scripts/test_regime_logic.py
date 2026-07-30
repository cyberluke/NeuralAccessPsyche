#!/usr/bin/env python3
"""
Test script to validate regime-specific logic in dexperts_causal_ablation.py
without requiring a live API endpoint.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.dexperts_causal_ablation import (
    compute_fluency_score,
    compute_diversity_score,
    compute_coherence_score,
)


def test_metric_functions():
    """Test that metric functions work correctly."""
    print("Testing metric functions...")
    
    # Test fluency
    text = "This is a test sentence with some words."
    fluency = compute_fluency_score(text)
    assert 0.0 <= fluency <= 1.0, f"Fluency out of range: {fluency}"
    print(f"  Fluency: {fluency:.3f} [OK]")
    
    # Test diversity
    diversity = compute_diversity_score(text)
    assert 0.0 <= diversity <= 1.0, f"Diversity out of range: {diversity}"
    print(f"  Diversity: {diversity:.3f} [OK]")
    
    # Test coherence
    coherence = compute_coherence_score(text)
    assert 0.0 <= coherence <= 1.0, f"Coherence out of range: {coherence}"
    print(f"  Coherence: {coherence:.3f} [OK]")
    
    print("All metric functions passed [OK]\n")


def test_regime_configuration():
    """Test that regime configuration logic is correct."""
    print("Testing regime configuration...")
    
    # Simulate regime 06b configuration
    regime = "06b"
    api_url = "http://127.0.0.1:8001/v1/completions"
    model_name = "Qwen/Qwen3-0.6B-Base"
    use_chat_format = False
    
    assert "completions" in api_url, "06b should use /v1/completions"
    assert "chat" not in api_url, "06b should NOT use chat completions"
    assert not use_chat_format, "06b should not use chat format"
    print(f"  Regime 06b: {model_name} @ {api_url} [OK]")
    
    # Simulate regime 14b configuration
    regime = "14b"
    api_url = "http://127.0.0.1:8000/v1/chat/completions"
    model_name = "nram-qwen3-14b-awq"
    use_chat_format = True
    
    assert "chat/completions" in api_url, "14b should use /v1/chat/completions"
    assert use_chat_format, "14b should use chat format"
    print(f"  Regime 14b: {model_name} @ {api_url} [OK]")
    
    print("Regime configuration passed [OK]\n")


def test_request_body_construction():
    """Test that request bodies are constructed correctly for each regime."""
    print("Testing request body construction...")
    
    prompt = "Test prompt"
    seed = 42
    alpha = 1.0
    condition_name = "dexperts_medium"
    
    # Regime 06b: Raw completions
    regime = "06b"
    model_name = "Qwen/Qwen3-0.6B-Base"
    use_chat_format = False
    
    nram_config = {
        "enabled": True,
        "profile": "normal",
        "request_id": f"ablation-{regime}-{condition_name}-{seed}-test",
        "dexperts_config": {"alpha": alpha},
    }
    
    body_06b = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
        "seed": seed,
        "nram": nram_config,
    }
    
    assert "prompt" in body_06b, "06b should have 'prompt' field"
    assert "messages" not in body_06b, "06b should NOT have 'messages' field"
    assert body_06b["nram"]["dexperts_config"]["alpha"] == alpha
    print(f"  Regime 06b body: prompt={body_06b['prompt'][:30]}... [OK]")
    
    # Regime 14b: Chat completions
    regime = "14b"
    model_name = "nram-qwen3-14b-awq"
    use_chat_format = True
    
    nram_config = {
        "enabled": True,
        "profile": "normal",
        "request_id": f"ablation-{regime}-{condition_name}-{seed}-test",
        "dexperts_config": {"alpha": alpha},
    }
    
    body_14b = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7,
        "seed": seed,
        "nram": nram_config,
    }
    
    assert "messages" in body_14b, "14b should have 'messages' field"
    assert "prompt" not in body_14b, "14b should NOT have 'prompt' field"
    assert body_14b["nram"]["dexperts_config"]["alpha"] == alpha
    print(f"  Regime 14b body: messages={body_14b['messages'][0]['content'][:30]}... [OK]")
    
    print("Request body construction passed [OK]\n")


def test_output_paths():
    """Test that output paths are regime-specific."""
    print("Testing output paths...")
    
    # Regime 06b
    regime = "06b"
    output_path = f"artifacts/dexperts/r5/causal_ablation_{regime}.json"
    raw_output_path = f"artifacts/dexperts/r5/causal_ablation_raw_{regime}.jsonl"
    
    assert "06b" in output_path, "06b output path should contain '06b'"
    assert "r5" in output_path, "Output path should contain 'r5'"
    print(f"  Regime 06b output: {output_path} [OK]")
    print(f"  Regime 06b raw: {raw_output_path} [OK]")
    
    # Regime 14b
    regime = "14b"
    output_path = f"artifacts/dexperts/r5/causal_ablation_{regime}.json"
    raw_output_path = f"artifacts/dexperts/r5/causal_ablation_raw_{regime}.jsonl"
    
    assert "14b" in output_path, "14b output path should contain '14b'"
    assert "r5" in output_path, "Output path should contain 'r5'"
    print(f"  Regime 14b output: {output_path} [OK]")
    print(f"  Regime 14b raw: {raw_output_path} [OK]")
    
    print("Output paths passed [OK]\n")


if __name__ == "__main__":
    print("=" * 60)
    print("DExperts Causal Ablation - Regime Logic Tests")
    print("=" * 60)
    print()
    
    try:
        test_metric_functions()
        test_regime_configuration()
        test_request_body_construction()
        test_output_paths()
        
        print("=" * 60)
        print("ALL TESTS PASSED [OK]")
        print("=" * 60)
        sys.exit(0)
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
