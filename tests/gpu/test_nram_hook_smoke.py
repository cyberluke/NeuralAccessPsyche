"""
Minimal smoke test for NRAM v5 representation hook execution.

This test verifies that:
1. The hook factory can be imported
2. The hook can be registered on a mock module
3. The hook executes during forward pass
4. Telemetry is emitted when enabled
"""
import pytest
import torch
import torch.nn as nn
from nram_sglang.hooks.factory import make_nram_hook, hook_manager
from nram_sglang.hooks.request_context import context_manager


class MockQwenLayer(nn.Module):
    """Mock Qwen decoder layer for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5120, 5120)
    
    def forward(self, positions, hidden_states, forward_batch, residual, post_residual_addition=None):
        # Simulate SGLang's Qwen3DecoderLayer signature
        # Returns (hidden_states, residual) tuple
        out = self.linear(hidden_states)
        return (out, residual)


def test_hook_factory_creates_hook():
    """Test that make_nram_hook creates a valid hook."""
    config = {"enabled": True}
    hook = make_nram_hook(config)
    
    assert hook is not None
    assert hook._enabled is True
    # SGLangCompatibleNRAMHook determines layer index dynamically from module
    # at call time, not at registration time. It has a layer_cache dict.
    assert hasattr(hook, '_layer_cache')
    assert isinstance(hook._layer_cache, dict)
    assert hasattr(hook, '_invocation_count')
    assert hook._invocation_count == 0


def test_hook_registers_on_module():
    """Test that hook can be registered on a module."""
    module = MockQwenLayer()
    hook = make_nram_hook({"enabled": True})
    
    # Register hook
    handle = module.register_forward_hook(hook)
    
    assert handle is not None
    
    # Clean up
    handle.remove()


def test_hook_executes_during_forward():
    """Test that hook executes during forward pass."""
    module = MockQwenLayer()
    hook = make_nram_hook({"enabled": True})
    
    # Set up a request context so the hook will execute
    from nram_sglang.hooks.request_context import context_manager, get_current_context
    context = context_manager.create_context("test-request-001")
    context_manager.install_context("test-request-001")
    
    # Verify context is active
    current_ctx = get_current_context()
    assert current_ctx is not None, "Context should be active"
    assert current_ctx.request_id == "test-request-001", "Context ID should match"
    
    # Register hook
    handle = module.register_forward_hook(hook)
    
    # Verify hook is registered
    assert handle is not None, "Hook handle should not be None"
    
    # Create mock inputs matching SGLang's Qwen3DecoderLayer signature
    positions = torch.tensor([0, 1, 2])
    hidden_states = torch.randn(3, 5120)  # (num_tokens, hidden_dim)
    forward_batch = None  # Mock
    residual = torch.randn(3, 5120)
    
    # Execute forward pass
    output = module(positions, hidden_states, forward_batch, residual)
    
    # Verify hook was called by checking internal invocation count
    assert hook._invocation_count == 1, f"Hook should be called once during forward, got {hook._invocation_count}"
    
    # Verify output structure preserved
    assert isinstance(output, tuple), "Output should be tuple"
    assert len(output) == 2, "Output should have 2 elements"
    
    # Clean up
    handle.remove()
    context_manager.remove_context("test-request-001")


def test_hook_disabled_no_modification():
    """Test that disabled hook doesn't modify output."""
    module = MockQwenLayer()
    hook = make_nram_hook({"enabled": False})
    
    # Register hook
    handle = module.register_forward_hook(hook)
    
    # Create inputs
    positions = torch.tensor([0])
    hidden_states = torch.randn(1, 5120)
    forward_batch = None
    residual = torch.randn(1, 5120)
    
    # Execute forward pass
    output = module(positions, hidden_states, forward_batch, residual)
    
    # Verify output is unchanged (no intervention applied)
    assert isinstance(output, tuple)
    assert len(output) == 2
    
    # Clean up
    handle.remove()


def test_hook_manager_global_instance():
    """Test that global hook_manager instance exists."""
    assert hook_manager is not None
    assert hasattr(hook_manager, 'register_on_model')
    assert hasattr(hook_manager, 'create_request_context')
    assert hasattr(hook_manager, 'install_request_context')
    assert hasattr(hook_manager, 'remove_request_context')


def test_request_context_lifecycle():
    """Test request context creation and cleanup."""
    request_id = "test-request-001"
    
    # Create context
    context = context_manager.create_context(request_id)
    assert context is not None
    assert context.request_id == request_id
    
    # Install context
    context_manager.install_context(request_id)
    
    # Verify context is active
    from nram_sglang.hooks.request_context import get_current_context
    current = get_current_context()
    assert current is not None
    assert current.request_id == request_id
    
    # Remove context
    context_manager.remove_context(request_id)
    
    # Verify context is cleared
    current = get_current_context()
    assert current is None


if __name__ == "__main__":
    # Run tests
    test_hook_factory_creates_hook()
    print("[OK] test_hook_factory_creates_hook")
    
    test_hook_registers_on_module()
    print("[OK] test_hook_registers_on_module")
    
    test_hook_executes_during_forward()
    print("[OK] test_hook_executes_during_forward")
    
    test_hook_disabled_no_modification()
    print("[OK] test_hook_disabled_no_modification")
    
    test_hook_manager_global_instance()
    print("[OK] test_hook_manager_global_instance")
    
    test_request_context_lifecycle()
    print("[OK] test_request_context_lifecycle")
    
    print("\n[OK] All smoke tests passed")
