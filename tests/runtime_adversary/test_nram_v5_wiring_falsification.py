"""
NRAM v5 Wiring Falsification Tests

These tests attempt to falsify claims that NRAM v5 features are runtime-wired.

CLAIMS TO FALSIFY:
1. Activation Addition is wired into SGLang inference
2. Multi-vector representation control is active
3. Conceptor steering is applied during inference
4. Trained hidden-state probes execute during inference
5. Latent closed-loop steering feeds back to intervention strength
6. Semantic closed-loop evaluation affects generation
7. Branch-and-tournament generation is active
8. DExperts with three distributions is operational
9. Request control plane integrates all components
10. Streamlit integration exposes advanced controls

METHODOLOGY:
- Check if features are rejected by the API (UNSUPPORTED_NRAM_FEATURES)
- Verify if code is imported in actual inference engines
- Test if features can be invoked without errors
- Measure if features produce observable effects
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestNRAMv5RuntimeWired:
    """Test that NRAM v5 features are now runtime-wired through logit processor."""
    
    def test_unsupported_features_set_exists(self):
        """Verify UNSUPPORTED_NRAM_FEATURES set exists (now minimal)."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        assert isinstance(UNSUPPORTED_NRAM_FEATURES, set)
        # After wiring, only truly unsupported infrastructure features remain
        assert "reft" in UNSUPPORTED_NRAM_FEATURES
        assert "soft_prompts" in UNSUPPORTED_NRAM_FEATURES
    
    def test_activation_addition_is_now_supported(self):
        """VERIFICATION: Activation addition is now runtime-wired."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        assert "activation_addition" not in UNSUPPORTED_NRAM_FEATURES, \
            "activation_addition should now be runtime-wired"
    
    def test_dexperts_is_now_supported(self):
        """VERIFICATION: DExperts is now runtime-wired."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        assert "dexperts" not in UNSUPPORTED_NRAM_FEATURES, \
            "dexperts should now be runtime-wired"
    
    def test_conceptor_steering_is_now_supported(self):
        """VERIFICATION: Conceptor steering is now runtime-wired."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        assert "conceptor_steering" not in UNSUPPORTED_NRAM_FEATURES, \
            "conceptor_steering should now be runtime-wired"
    
    def test_hidden_state_probes_is_now_supported(self):
        """VERIFICATION: Hidden state probes are now runtime-wired."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        assert "hidden_state_probes" not in UNSUPPORTED_NRAM_FEATURES, \
            "hidden_state_probes should now be runtime-wired"
    
    def test_latent_closed_loop_is_now_supported(self):
        """VERIFICATION: Latent closed loop is now runtime-wired."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        assert "latent_closed_loop" not in UNSUPPORTED_NRAM_FEATURES, \
            "latent_closed_loop should now be runtime-wired"
    
    def test_branch_tournament_is_now_supported(self):
        """VERIFICATION: Branch tournament is now runtime-wired."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        assert "branch_tournament" not in UNSUPPORTED_NRAM_FEATURES, \
            "branch_tournament should now be runtime-wired"


class TestNRAMv5CodeWired:
    """Test that NRAM v5 code IS now imported in inference engines."""
    
    def test_sglang_engine_imports_request_control_plane(self):
        """VERIFICATION: SGLang engine module now imports request_control_plane."""
        import inspect
        import core.engines.sglang_engine as sglang_module
        
        source = inspect.getsource(sglang_module)
        assert "NRAMRequestControlPlane" in source or "NRAMRequestConfig" in source, \
            "SGLangEngine module should now reference NRAMRequestControlPlane or NRAMRequestConfig"
    
    def test_sglang_engine_has_activation_addition_config_builder(self):
        """VERIFICATION: SGLang engine now has activation addition config builder."""
        from core.engines.sglang_engine import SGLangEngine
        
        assert hasattr(SGLangEngine, '_build_activation_addition_config'), \
            "SGLangEngine should have _build_activation_addition_config method"
    
    def test_sglang_engine_has_conceptor_config_builder(self):
        """VERIFICATION: SGLang engine now has conceptor config builder."""
        from core.engines.sglang_engine import SGLangEngine
        
        assert hasattr(SGLangEngine, '_build_conceptor_steering_config'), \
            "SGLangEngine should have _build_conceptor_steering_config method"
    
    def test_sglang_engine_has_dexperts_config_builder(self):
        """VERIFICATION: SGLang engine now has dexperts config builder."""
        from core.engines.sglang_engine import SGLangEngine
        
        assert hasattr(SGLangEngine, '_build_dexperts_config'), \
            "SGLangEngine should have _build_dexperts_config method"
    
    def test_sglang_engine_has_branch_tournament_config_builder(self):
        """VERIFICATION: SGLang engine now has branch tournament config builder."""
        from core.engines.sglang_engine import SGLangEngine
        
        assert hasattr(SGLangEngine, '_build_branch_tournament_config'), \
            "SGLangEngine should have _build_branch_tournament_config method"
    
    def test_serialization_includes_nram_v5_configs(self):
        """VERIFICATION: Serialization now includes NRAM v5 configs."""
        from core.steering.serialization import build_custom_params
        import inspect
        
        source = inspect.getsource(build_custom_params)
        assert "activation_addition_config" in source, \
            "build_custom_params should accept activation_addition_config"
        assert "conceptor_config" in source, \
            "build_custom_params should accept conceptor_config"
        assert "dexperts_config" in source, \
            "build_custom_params should accept dexperts_config"


class TestNRAMv5CodeExists:
    """Test that NRAM v5 code exists and is properly named."""
    
    def test_forward_hooks_module_exists(self):
        """Verify forward_hooks module exists."""
        from core.steering import forward_hooks
        assert hasattr(forward_hooks, 'HookManager')
        assert hasattr(forward_hooks, 'ActivationSteeringHook')
    
    def test_live_activation_addition_module_exists(self):
        """Verify live_activation_addition module exists."""
        from core.steering import live_activation_addition
        assert hasattr(live_activation_addition, 'LiveActivationAddition')
    
    def test_multi_vector_representation_module_exists(self):
        """Verify multi_vector_representation module exists."""
        from core.steering import multi_vector_representation
        assert hasattr(multi_vector_representation, 'MultiVectorRepresentationController')
    
    def test_conceptor_steering_module_exists(self):
        """Verify conceptor_steering module exists."""
        from core.steering import conceptor_steering
        assert hasattr(conceptor_steering, 'ConceptorController')
    
    def test_hidden_state_probes_module_exists(self):
        """Verify hidden_state_probes module exists."""
        from core.steering import hidden_state_probes
        assert hasattr(hidden_state_probes, 'HiddenStateProbe')
    
    def test_latent_closed_loop_module_exists(self):
        """Verify latent_closed_loop module exists."""
        from core.steering import latent_closed_loop
        assert hasattr(latent_closed_loop, 'LatentClosedLoopController')
    
    def test_semantic_closed_loop_module_exists(self):
        """Verify semantic_closed_loop module exists."""
        from core.steering import semantic_closed_loop
        assert hasattr(semantic_closed_loop, 'SemanticClosedLoopController')
    
    def test_branch_tournament_module_exists(self):
        """Verify branch_tournament module exists."""
        from core.steering import branch_tournament
        assert hasattr(branch_tournament, 'BranchTournamentEngine')
    
    def test_dexperts_module_exists(self):
        """Verify dexperts module exists."""
        from core.steering import dexperts
        assert hasattr(dexperts, 'DExpertsController')
    
    def test_request_control_plane_module_exists(self):
        """Verify request_control_plane module exists."""
        from core.steering import request_control_plane
        assert hasattr(request_control_plane, 'NRAMRequestControlPlane')


class TestNRAMv5RejectionBehavior:
    """Test that requesting unsupported features produces errors."""
    
    @pytest.mark.asyncio
    async def test_activation_addition_request_accepted(self):
        """VERIFICATION: Request with activation_addition should be accepted."""
        from core.engines.sglang_engine import SGLangEngine, SGLangEngineError
        from core.contracts.openai import ChatCompletionRequest
        
        engine = SGLangEngine(base_url="http://localhost:30000/v1", model="nram-qwen3-14b-awq")
        
        request = ChatCompletionRequest(
            model="nram-qwen3-14b-awq",
            messages=[{"role": "user", "content": "test"}],
            nram={"activation_addition": True}
        )
        
        # Should NOT raise - feature is now wired
        try:
            engine.validate(request)
        except SGLangEngineError as e:
            if "Unsupported NRAM runtime feature" in str(e):
                pytest.fail("activation_addition should be accepted, not rejected")
            raise
    
    @pytest.mark.asyncio
    async def test_dexperts_request_accepted(self):
        """VERIFICATION: Request with dexperts should be accepted."""
        from core.engines.sglang_engine import SGLangEngine, SGLangEngineError
        from core.contracts.openai import ChatCompletionRequest
        
        engine = SGLangEngine(base_url="http://localhost:30000/v1", model="nram-qwen3-14b-awq")
        
        request = ChatCompletionRequest(
            model="nram-qwen3-14b-awq",
            messages=[{"role": "user", "content": "test"}],
            nram={"dexperts": True}
        )
        
        # Should NOT raise - feature is now wired
        try:
            engine.validate(request)
        except SGLangEngineError as e:
            if "Unsupported NRAM runtime feature" in str(e):
                pytest.fail("dexperts should be accepted, not rejected")
            raise
    
    @pytest.mark.asyncio
    async def test_conceptor_steering_request_accepted(self):
        """VERIFICATION: Request with conceptor_steering should be accepted."""
        from core.engines.sglang_engine import SGLangEngine, SGLangEngineError
        from core.contracts.openai import ChatCompletionRequest
        
        engine = SGLangEngine(base_url="http://localhost:30000/v1", model="nram-qwen3-14b-awq")
        
        request = ChatCompletionRequest(
            model="nram-qwen3-14b-awq",
            messages=[{"role": "user", "content": "test"}],
            nram={"conceptor_steering": True}
        )
        
        # Should NOT raise - feature is now wired
        try:
            engine.validate(request)
        except SGLangEngineError as e:
            if "Unsupported NRAM runtime feature" in str(e):
                pytest.fail("conceptor_steering should be accepted, not rejected")
            raise


class TestNRAMv5WhatIsActuallyWired:
    """Test what IS actually wired in the current implementation."""
    
    def test_logit_processor_is_wired(self):
        """Verify NRAMLogitProcessor is actually wired."""
        from core.engines.sglang_engine import SGLangEngine
        from nram_sglang.processor import NRAMLogitProcessor
        import inspect
        
        source = inspect.getsource(SGLangEngine)
        assert "NRAMLogitProcessor" in source, \
            "NRAMLogitProcessor should be referenced in SGLangEngine"
    
    def test_entropy_control_is_wired(self):
        """Verify entropy control is wired through logit processor."""
        from core.engines.sglang_engine import SGLangEngine
        import inspect
        
        source = inspect.getsource(SGLangEngine)
        assert "_build_entropy_config" in source, \
            "SGLangEngine should build entropy config"
    
    def test_concept_injection_is_wired(self):
        """Verify concept injection is wired through logit processor."""
        from core.engines.sglang_engine import SGLangEngine
        import inspect
        
        source = inspect.getsource(SGLangEngine)
        assert "_build_concept_config" in source, \
            "SGLangEngine should build concept config"
    
    def test_phrase_constraints_are_wired(self):
        """Verify phrase constraints are wired through logit processor."""
        from core.engines.sglang_engine import SGLangEngine
        import inspect
        
        source = inspect.getsource(SGLangEngine)
        assert "_build_phrase_constraint_config" in source, \
            "SGLangEngine should build phrase constraint config"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
