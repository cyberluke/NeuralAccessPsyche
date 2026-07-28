#!/usr/bin/env python3
"""
Diagnostic script to verify NRAM hook registration in SGLang.
This script tests the hook registration mechanism and checks if hooks
are actually being attached to model layers.
"""
import sys
import logging

# Configure logging to see all messages
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

sys.path.insert(0, '/opt/nram_python')

print("=" * 70)
print("NRAM Hook Registration Diagnostic")
print("=" * 70)

# Test 1: Import hook factory
print("\n[TEST 1] Importing hook factory...")
try:
    from nram_sglang.hooks.factory import make_nram_hook
    print("✓ Hook factory imported successfully")
except Exception as e:
    print(f"✗ Failed to import hook factory: {e}")
    sys.exit(1)

# Test 2: Create a hook instance
print("\n[TEST 2] Creating hook instance...")
try:
    config = {'enabled': True}
    hook = make_nram_hook(config)
    print(f"✓ Hook created: {type(hook).__name__}")
    print(f"  - Hook ID: {hook.hook_id}")
    print(f"  - Layer index: {hook.layer_index}")
    print(f"  - Module name: {hook.module_name}")
    print(f"  - Enabled: {hook._enabled}")
except Exception as e:
    print(f"✗ Failed to create hook: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test pattern matching
print("\n[TEST 3] Testing pattern matching...")
import fnmatch
pattern = 'model.layers.*'
test_names = [
    'model.layers.0',
    'model.layers.1',
    'model.layers.20',
    'model.layers.0.self_attn',
    'model',
]
print(f"Pattern: {pattern}")
for name in test_names:
    match = fnmatch.fnmatch(name, pattern)
    print(f"  {name:40s} -> {match}")

# Test 4: Test hook registration on a mock model
print("\n[TEST 4] Testing hook registration on mock model...")
try:
    from sglang.srt.model_executor.hook_manager import register_forward_hooks
    import torch.nn as nn
    
    class MockLayer(nn.Module):
        def forward(self, x):
            return x
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([MockLayer() for _ in range(40)])
    
    model = MockModel()
    hook_specs = [{
        'name': 'nram_actadd',
        'target_modules': ['model.layers.*'],
        'hook_factory': 'nram_sglang.hooks.factory:make_nram_hook',
        'config': {'enabled': True}
    }]
    
    print("Calling register_forward_hooks...")
    register_forward_hooks(model, hook_specs)
    
    # Count registered hooks
    hook_count = 0
    for name, module in model.named_modules():
        if hasattr(module, '_forward_hooks') and len(module._forward_hooks) > 0:
            hook_count += 1
            if hook_count <= 3:
                print(f"  ✓ Hook registered on: {name}")
    
    print(f"\nTotal hooks registered: {hook_count}")
    print(f"Expected: 40 (one per decoder layer)")
    
    if hook_count == 40:
        print("✓ Hook registration mechanism is working correctly")
    else:
        print(f"✗ Hook registration failed: expected 40, got {hook_count}")
        
except Exception as e:
    print(f"✗ Hook registration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check if SGLang's hook_manager logs are visible
print("\n[TEST 5] Checking SGLang hook_manager logging...")
try:
    from sglang.srt.model_executor import hook_manager
    logger = logging.getLogger('sglang.srt.model_executor.hook_manager')
    print(f"Logger level: {logger.level}")
    print(f"Logger handlers: {logger.handlers}")
    print(f"Logger effective level: {logger.getEffectiveLevel()}")
except Exception as e:
    print(f"Could not check logger: {e}")

print("\n" + "=" * 70)
print("Diagnostic complete")
print("=" * 70)
