#!/usr/bin/env python3
"""
Diagnostic: Check if adapters are actually different
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_adapter_differences():
    print("=" * 80)
    print("Adapter Diagnostic Test")
    print("=" * 80)
    
    # Load base model
    model_path = "E:/_MODELS/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    print(f"\n[1] Loading base model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load adapters
    expert_path = "artifacts/dexperts/adapters/nontoxic"
    anti_expert_path = "artifacts/dexperts/adapters/toxic"
    
    print(f"\n[2] Loading expert adapter from {expert_path}")
    expert_model = PeftModel.from_pretrained(base_model, expert_path)
    
    print(f"\n[3] Loading anti-expert adapter from {anti_expert_path}")
    # Need to reload base model for second adapter
    base_model2 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    anti_expert_model = PeftModel.from_pretrained(base_model2, anti_expert_path)
    
    # Test with a simple prompt
    test_text = "The government should"
    print(f"\n[4] Testing with text: '{test_text}'")
    
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(expert_model.device)
    
    # Get base model output
    print("\n[5] Base model forward pass")
    with torch.no_grad():
        base_outputs = base_model(input_ids=input_ids)
        base_logits = base_outputs.logits[0, -1, :].cpu().numpy()
        print(f"  Base logits shape: {base_logits.shape}")
        print(f"  Base logits mean: {base_logits.mean():.6f}")
        print(f"  Base logits std: {base_logits.std():.6f}")
    
    # Get expert model output
    print("\n[6] Expert model forward pass")
    with torch.no_grad():
        expert_outputs = expert_model(input_ids=input_ids)
        expert_logits = expert_outputs.logits[0, -1, :].cpu().numpy()
        print(f"  Expert logits shape: {expert_logits.shape}")
        print(f"  Expert logits mean: {expert_logits.mean():.6f}")
        print(f"  Expert logits std: {expert_logits.std():.6f}")
    
    # Get anti-expert model output
    print("\n[7] Anti-expert model forward pass")
    with torch.no_grad():
        anti_expert_outputs = anti_expert_model(input_ids=input_ids)
        anti_expert_logits = anti_expert_outputs.logits[0, -1, :].cpu().numpy()
        print(f"  Anti-expert logits shape: {anti_expert_logits.shape}")
        print(f"  Anti-expert logits mean: {anti_expert_logits.mean():.6f}")
        print(f"  Anti-expert logits std: {anti_expert_logits.std():.6f}")
    
    # Compare
    print("\n[8] Comparison")
    base_expert_diff = (base_logits - expert_logits).mean()
    base_anti_diff = (base_logits - anti_expert_logits).mean()
    expert_anti_diff = (expert_logits - anti_expert_logits).mean()
    
    print(f"  Base vs Expert mean diff: {base_expert_diff:.6f}")
    print(f"  Base vs Anti-expert mean diff: {base_anti_diff:.6f}")
    print(f"  Expert vs Anti-expert mean diff: {expert_anti_diff:.6f}")
    
    # Check if adapters are actually loaded
    print("\n[9] Adapter status")
    print(f"  Expert model has adapters: {hasattr(expert_model, 'peft_config')}")
    print(f"  Anti-expert model has adapters: {hasattr(anti_expert_model, 'peft_config')}")
    
    if hasattr(expert_model, 'peft_config'):
        print(f"  Expert adapter names: {list(expert_model.peft_config.keys())}")
    if hasattr(anti_expert_model, 'peft_config'):
        print(f"  Anti-expert adapter names: {list(anti_expert_model.peft_config.keys())}")
    
    # Check adapter weights
    print("\n[10] Adapter weight check")
    expert_trainable = sum(p.numel() for p in expert_model.parameters() if p.requires_grad)
    anti_expert_trainable = sum(p.numel() for p in anti_expert_model.parameters() if p.requires_grad)
    
    print(f"  Expert trainable parameters: {expert_trainable:,}")
    print(f"  Anti-expert trainable parameters: {anti_expert_trainable:,}")
    
    print("\n[11] Conclusion")
    if abs(expert_anti_diff) < 0.001:
        print("  ❌ FAIL: Expert and anti-expert produce nearly identical outputs")
        print("  Possible causes:")
        print("    - Adapters not properly loaded")
        print("    - Adapters trained on same data")
        print("    - Adapter weights are identical")
        print("    - LoRA rank too small to capture differences")
    else:
        print("  ✓ PASS: Adapters produce different outputs")
        print(f"  Mean difference: {abs(expert_anti_diff):.6f}")

if __name__ == "__main__":
    test_adapter_differences()
