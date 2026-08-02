#!/usr/bin/env python3
"""
Deep diagnostic: Check adapter weight loading and application
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def check_adapter_weights():
    print("=" * 80)
    print("Deep Adapter Weight Diagnostic")
    print("=" * 80)
    
    # Load base model
    model_path = "E:/_MODELS/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    print(f"\n[1] Loading base model")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load expert adapter
    expert_path = "artifacts/dexperts/adapters/nontoxic"
    print(f"\n[2] Loading expert adapter from {expert_path}")
    
    # Check adapter config
    config_path = f"{expert_path}/adapter_config.json"
    with open(config_path) as f:
        adapter_config = json.load(f)
    print(f"  Adapter config: {json.dumps(adapter_config, indent=2)}")
    
    # Load the adapter
    expert_model = PeftModel.from_pretrained(base_model, expert_path)
    
    # Check if adapter is actually active
    print(f"\n[3] Checking adapter status")
    print(f"  Active adapter: {expert_model.active_adapter}")
    print(f"  Available adapters: {expert_model.peft_config.keys()}")
    
    # Check adapter weights
    print(f"\n[4] Checking adapter weight tensors")
    adapter_found = False
    for name, param in expert_model.named_parameters():
        if 'lora' in name.lower():
            adapter_found = True
            print(f"  Found LoRA parameter: {name}")
            print(f"    Shape: {param.shape}")
            print(f"    Requires grad: {param.requires_grad}")
            print(f"    Mean: {param.mean().item():.6f}")
            print(f"    Std: {param.std().item():.6f}")
            print(f"    Min: {param.min().item():.6f}")
            print(f"    Max: {param.max().item():.6f}")
            
            # Check if weights are all zeros
            if param.abs().sum().item() < 1e-6:
                print(f"    WARNING: Weights appear to be all zeros!")
            break
    
    if not adapter_found:
        print(f"  ERROR: No LoRA parameters found!")
    
    # Try to manually activate the adapter
    print(f"\n[5] Manually activating adapter")
    try:
        expert_model.set_adapter('default')
        print(f"  Successfully set adapter to 'default'")
    except Exception as e:
        print(f"  Error setting adapter: {e}")
    
    # Test forward pass with adapter explicitly enabled
    print(f"\n[6] Testing forward pass with adapter enabled")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_text = "The government should"
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(expert_model.device)
    
    # Get base model output (disable adapter temporarily)
    print(f"  Getting base model output...")
    with expert_model.disable_adapter():
        with torch.no_grad():
            base_outputs = expert_model(input_ids=input_ids)
            base_logits = base_outputs.logits[0, -1, :].cpu()
    
    # Get adapter output
    print(f"  Getting adapter output...")
    with torch.no_grad():
        adapter_outputs = expert_model(input_ids=input_ids)
        adapter_logits = adapter_outputs.logits[0, -1, :].cpu()
    
    # Compare
    diff = (base_logits - adapter_logits).abs().mean().item()
    print(f"\n[7] Comparison")
    print(f"  Mean absolute difference: {diff:.6f}")
    
    if diff < 1e-6:
        print(f"  ERROR: Adapter has NO EFFECT on output!")
        print(f"  This indicates the adapter is not being applied.")
    else:
        print(f"  SUCCESS: Adapter is producing different outputs")
    
    # Check if we're in inference mode
    print(f"\n[8] Checking model mode")
    print(f"  Training mode: {expert_model.training}")
    print(f"  Model device: {expert_model.device}")
    
    # Check PEFT configuration
    print(f"\n[9] PEFT configuration details")
    peft_config = expert_model.peft_config['default']
    print(f"  Task type: {peft_config.task_type}")
    print(f"  Inference mode: {peft_config.inference_mode}")
    print(f"  LoRA rank (r): {peft_config.r}")
    print(f"  LoRA alpha: {peft_config.lora_alpha}")
    print(f"  Target modules: {peft_config.target_modules}")

if __name__ == "__main__":
    check_adapter_weights()
