#!/usr/bin/env python3
"""
Diagnostic script to verify NRAM hook registration in SGLang container.
"""
import sys
import torch
from pathlib import Path

# Add nram_sglang to path
sys.path.insert(0, '/opt/nram_python')

def test_hook_factory():
    """Test that hook factory can be imported and called."""
    print("=" * 60)
    print("TEST 1: Hook Factory Import")
    print("=" * 60)
    
    try:
        from nram_sglang.hooks.factory import make_nram_hook
        print("✓ Hook factory imported successfully")
        
        # Test creating a hook
        config = {'enabled': True}
        hook = make_nram_hook(config)
        print(f"✓ Hook created: {type(hook).__name__}")
        print(f"  - Hook ID: {hook.hook_id}")
        print(f"  - Layer index: {hook.layer_index}")
        print(f"  - Module name: {hook.module_name}")
        print(f"  - Enabled: {hook._enabled}")
        return True
    except Exception as e:
        print(f"✗ Hook factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pattern_matching():
    """Test that pattern matching works for Qwen model layers."""
    print("\n" + "=" * 60)
    print("TEST 2: Pattern Matching")
    print("=" * 60)
    
    import fnmatch
    import re
    
    # Test fnmatch pattern
    pattern = 'model.layers.*'
    test_names = [
        'model.layers.0',
        'model.layers.1',
        'model.layers.20',
        'model.layers.0.self_attn',
        'model.layers.0.mlp',
        'model',
        'model.layers',
    ]
    
    print(f"Pattern: {pattern}")
    for name in test_names:
        match = fnmatch.fnmatch(name, pattern)
        print(f"  {name:40s} -> {match}")
    
    # Test regex pattern (used in factory.py)
    print("\nRegex pattern: ^model\\.layers\\.\\d+$")
    regex = re.compile(r'^model\.layers\.(\d+)$')
    for name in test_names:
        match = regex.match(name)
        print(f"  {name:40s} -> {match is not None}")
    
    return True

def test_model_hook_registration():
    """Test that hooks can be registered on actual model layers."""
    print("\n" + "=" * 60)
    print("TEST 3: Model Hook Registration")
    print("=" * 60)
    
    try:
        from transformers import AutoModelForCausalLM
        from nram_sglang.hooks.factory import NRAMHookFactory
        
        print("Loading Qwen model (CPU, float16)...")
        model = AutoModelForCausalLM.from_pretrained(
            '/models',
            torch_dtype=torch.float16,
            device_map='cpu',
            low_cpu_mem_usage=True
        )
        print("✓ Model loaded")
        
        # Count decoder layers
        decoder_layers = []
        for name, module in model.named_modules():
            if name.startswith('model.layers.') and name.count('.') == 2:
                decoder_layers.append((name, module))
        
        print(f"✓ Found {len(decoder_layers)} decoder layers")
        if decoder_layers:
            print(f"  - First layer: {decoder_layers[0][0]}")
            print(f"  - Last layer: {decoder_layers[-1][0]}")
        
        # Test hook registration
        print("\nRegistering hooks via NRAMHookFactory...")
        factory = NRAMHookFactory()
        count = factory.register_hooks_on_model(model)
        print(f"✓ Registered {count} hooks")
        
        # Verify hooks are attached
        print("\nVerifying hook attachment...")
        hooked_layers = []
        for name, module in model.named_modules():
            if name.startswith('model.layers.') and name.count('.') == 2:
                if hasattr(module, '_forward_hooks') and len(module._forward_hooks) > 0:
                    hooked_layers.append(name)
        
        print(f"✓ {len(hooked_layers)} layers have hooks attached")
        if hooked_layers:
            print(f"  - First hooked: {hooked_layers[0]}")
            print(f"  - Last hooked: {hooked_layers[-1]}")
        
        # Test hook invocation
        print("\nTesting hook invocation with dummy input...")
        if hooked_layers:
            first_layer_name = hooked_layers[0]
            first_layer = dict(model.named_modules())[first_layer_name]
            
            # Create dummy input
            hidden_states = torch.randn(1, 1, model.config.hidden_size, dtype=torch.float16)
            
            # Call forward pass
            try:
                with torch.no_grad():
                    output = first_layer(hidden_states)
                print(f"✓ Forward pass succeeded on {first_layer_name}")
                print(f"  - Output type: {type(output)}")
                if isinstance(output, tuple):
                    print(f"  - Output shape: {output[0].shape}")
            except Exception as e:
                print(f"✗ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
        
        return count > 0
        
    except Exception as e:
        print(f"✗ Model hook registration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 60)
    print("NRAM Hook Registration Diagnostic")
    print("=" * 60)
    
    results = []
    
    # Test 1: Hook factory
    results.append(("Hook Factory", test_hook_factory()))
    
    # Test 2: Pattern matching
    results.append(("Pattern Matching", test_pattern_matching()))
    
    # Test 3: Model hook registration
    results.append(("Model Hook Registration", test_model_hook_registration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
