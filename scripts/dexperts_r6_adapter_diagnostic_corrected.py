#!/usr/bin/env python3
"""
Corrected diagnostic: Properly compare base vs adapter outputs
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_adapter_differences_corrected():
    print("=" * 80)
    print("Corrected Adapter Diagnostic Test")
    print("=" * 80)
    
    model_path = "E:/_MODELS/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    expert_path = "artifacts/dexperts/adapters/nontoxic"
    anti_expert_path = "artifacts/dexperts/adapters/toxic"
    
    test_text = "The government should"
    print(f"\nTest text: '{test_text}'")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    
    # Test 1: Load base model, get outputs, then load expert adapter
    print("\n[TEST 1] Base model -> Load expert adapter")
    print("-" * 80)
    
    base_model_1 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Getting base model outputs...")
    with torch.no_grad():
        base_outputs_1 = base_model_1(input_ids=input_ids)
        base_logits_1 = base_outputs_1.logits[0, -1, :].cpu()
        print(f"  Base logits mean: {base_logits_1.mean().item():.6f}")
    
    print("Loading expert adapter...")
    expert_model_1 = PeftModel.from_pretrained(base_model_1, expert_path)
    
    print("Getting expert model outputs...")
    with torch.no_grad():
        expert_outputs_1 = expert_model_1(input_ids=input_ids)
        expert_logits_1 = expert_outputs_1.logits[0, -1, :].cpu()
        print(f"  Expert logits mean: {expert_logits_1.mean().item():.6f}")
    
    diff_1 = (base_logits_1 - expert_logits_1).abs().mean().item()
    print(f"  Mean absolute difference: {diff_1:.6f}")
    
    # Test 2: Load base model, get outputs, then load anti-expert adapter
    print("\n[TEST 2] Base model -> Load anti-expert adapter")
    print("-" * 80)
    
    base_model_2 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Getting base model outputs...")
    with torch.no_grad():
        base_outputs_2 = base_model_2(input_ids=input_ids)
        base_logits_2 = base_outputs_2.logits[0, -1, :].cpu()
        print(f"  Base logits mean: {base_logits_2.mean().item():.6f}")
    
    print("Loading anti-expert adapter...")
    anti_expert_model_2 = PeftModel.from_pretrained(base_model_2, anti_expert_path)
    
    print("Getting anti-expert model outputs...")
    with torch.no_grad():
        anti_expert_outputs_2 = anti_expert_model_2(input_ids=input_ids)
        anti_expert_logits_2 = anti_expert_outputs_2.logits[0, -1, :].cpu()
        print(f"  Anti-expert logits mean: {anti_expert_logits_2.mean().item():.6f}")
    
    diff_2 = (base_logits_2 - anti_expert_logits_2).abs().mean().item()
    print(f"  Mean absolute difference: {diff_2:.6f}")
    
    # Test 3: Compare expert vs anti-expert
    print("\n[TEST 3] Expert vs Anti-expert comparison")
    print("-" * 80)
    
    # Load both adapters on the same base model
    base_model_3 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading expert adapter...")
    expert_model_3 = PeftModel.from_pretrained(base_model_3, expert_path, adapter_name="expert")
    
    print("Loading anti-expert adapter...")
    expert_model_3.load_adapter(anti_expert_path, adapter_name="anti_expert")
    
    print("Getting expert outputs...")
    expert_model_3.set_adapter("expert")
    with torch.no_grad():
        expert_outputs_3 = expert_model_3(input_ids=input_ids)
        expert_logits_3 = expert_outputs_3.logits[0, -1, :].cpu()
        print(f"  Expert logits mean: {expert_logits_3.mean().item():.6f}")
    
    print("Getting anti-expert outputs...")
    expert_model_3.set_adapter("anti_expert")
    with torch.no_grad():
        anti_expert_outputs_3 = expert_model_3(input_ids=input_ids)
        anti_expert_logits_3 = anti_expert_outputs_3.logits[0, -1, :].cpu()
        print(f"  Anti-expert logits mean: {anti_expert_logits_3.mean().item():.6f}")
    
    diff_3 = (expert_logits_3 - anti_expert_logits_3).abs().mean().item()
    print(f"  Mean absolute difference: {diff_3:.6f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Expert adapter effect (base vs expert): {diff_1:.6f}")
    print(f"Anti-expert adapter effect (base vs anti-expert): {diff_2:.6f}")
    print(f"Expert vs Anti-expert difference: {diff_3:.6f}")
    
    if diff_1 < 1e-6:
        print("\n[FAIL] Expert adapter has NO EFFECT")
    else:
        print(f"\n[PASS] Expert adapter produces different outputs (diff={diff_1:.6f})")
    
    if diff_2 < 1e-6:
        print("[FAIL] Anti-expert adapter has NO EFFECT")
    else:
        print(f"[PASS] Anti-expert adapter produces different outputs (diff={diff_2:.6f})")
    
    if diff_3 < 1e-6:
        print("[FAIL] Expert and Anti-expert produce IDENTICAL outputs")
    else:
        print(f"[PASS] Expert and Anti-expert produce DIFFERENT outputs (diff={diff_3:.6f})")

if __name__ == "__main__":
    test_adapter_differences_corrected()
