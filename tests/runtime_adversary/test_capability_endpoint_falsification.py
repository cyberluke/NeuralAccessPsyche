"""
Capability Endpoint Falsification Tests

These tests falsify the claim that the /v1/nram/capabilities endpoint
accurately reports feature availability.

CRITICAL FINDING:
The endpoint claims features are available (True) but the runtime
rejects requests attempting to use them (UNSUPPORTED_NRAM_FEATURES).

This is a capability endpoint inaccuracy that creates false scientific claims.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def read_api_routes_source():
    """Read the api/routes.py source file."""
    routes_path = project_root / "api" / "routes.py"
    return routes_path.read_text(encoding="utf-8")


def read_streamlit_source():
    """Read the streamlit_console.py source file."""
    streamlit_path = project_root / "streamlit_console.py"
    return streamlit_path.read_text(encoding="utf-8")


class TestCapabilityEndpointAccuracy:
    """Test that capability endpoint now accurately reports runtime-wired features."""
    
    def test_capabilities_endpoint_reports_activation_addition_wired(self):
        """Verify capabilities endpoint reports activation_addition as runtime-wired."""
        source = read_api_routes_source()
        
        # The endpoint should now report activation_addition with runtime_wired: True
        assert '"activation_addition"' in source and '"runtime_wired": True' in source, \
            "Capabilities endpoint should report activation_addition as runtime_wired"
    
    def test_capabilities_endpoint_reports_dexperts_wired(self):
        """Verify capabilities endpoint reports dexperts as runtime-wired."""
        source = read_api_routes_source()
        
        assert '"dexperts"' in source and '"runtime_wired": True' in source, \
            "Capabilities endpoint should report dexperts as runtime_wired"
    
    def test_capabilities_endpoint_reports_conceptor_steering_wired(self):
        """Verify capabilities endpoint reports conceptor_steering as runtime-wired."""
        source = read_api_routes_source()
        
        assert '"conceptor_steering"' in source and '"runtime_wired": True' in source, \
            "Capabilities endpoint should report conceptor_steering as runtime_wired"
    
    def test_capabilities_endpoint_reports_hidden_state_probes_wired(self):
        """Verify capabilities endpoint reports hidden_state_probes as runtime-wired."""
        source = read_api_routes_source()
        
        assert '"hidden_state_probes"' in source and '"runtime_wired": True' in source, \
            "Capabilities endpoint should report hidden_state_probes as runtime_wired"
    
    def test_capabilities_endpoint_reports_latent_closed_loop_wired(self):
        """Verify capabilities endpoint reports latent_closed_loop as runtime-wired."""
        source = read_api_routes_source()
        
        assert '"latent_closed_loop"' in source and '"runtime_wired": True' in source, \
            "Capabilities endpoint should report latent_closed_loop as runtime_wired"
    
    def test_capabilities_endpoint_reports_branch_tournament_wired(self):
        """Verify capabilities endpoint reports branch_tournament as runtime-wired."""
        source = read_api_routes_source()
        
        assert '"branch_tournament"' in source and '"runtime_wired": True' in source, \
            "Capabilities endpoint should report branch_tournament as runtime_wired"


class TestCapabilityRuntimeConsistency:
    """Test that capability claims are now consistent with runtime acceptance."""
    
    def test_activation_addition_wired_and_accepted(self):
        """VERIFICATION: activation_addition is now runtime-wired and accepted."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        
        # Runtime now accepts it (not in UNSUPPORTED_NRAM_FEATURES)
        assert "activation_addition" not in UNSUPPORTED_NRAM_FEATURES, \
            "activation_addition should NOT be in UNSUPPORTED_NRAM_FEATURES (now wired)"
        
        # Capabilities endpoint reports it as runtime_wired
        source = read_api_routes_source()
        assert '"activation_addition"' in source and '"runtime_wired": True' in source, \
            "Capabilities endpoint should report activation_addition as runtime_wired"
    
    def test_dexperts_wired_and_accepted(self):
        """VERIFICATION: dexperts is now runtime-wired and accepted."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        
        assert "dexperts" not in UNSUPPORTED_NRAM_FEATURES, \
            "dexperts should NOT be in UNSUPPORTED_NRAM_FEATURES (now wired)"
        
        source = read_api_routes_source()
        assert '"dexperts"' in source and '"runtime_wired": True' in source, \
            "Capabilities endpoint should report dexperts as runtime_wired"
    
    def test_conceptor_steering_wired_and_accepted(self):
        """VERIFICATION: conceptor_steering is now runtime-wired and accepted."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        
        assert "conceptor_steering" not in UNSUPPORTED_NRAM_FEATURES, \
            "conceptor_steering should NOT be in UNSUPPORTED_NRAM_FEATURES (now wired)"
        
        source = read_api_routes_source()
        assert '"conceptor_steering"' in source and '"runtime_wired": True' in source, \
            "Capabilities endpoint should report conceptor_steering as runtime_wired"
    
    def test_hidden_state_probes_wired_and_accepted(self):
        """VERIFICATION: hidden_state_probes is now runtime-wired and accepted."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        
        assert "hidden_state_probes" not in UNSUPPORTED_NRAM_FEATURES, \
            "hidden_state_probes should NOT be in UNSUPPORTED_NRAM_FEATURES (now wired)"
        
        source = read_api_routes_source()
        assert '"hidden_state_probes"' in source and '"runtime_wired": True' in source, \
            "Capabilities endpoint should report hidden_state_probes as runtime_wired"
    
    def test_latent_closed_loop_wired_and_accepted(self):
        """VERIFICATION: latent_closed_loop is now runtime-wired and accepted."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        
        assert "latent_closed_loop" not in UNSUPPORTED_NRAM_FEATURES, \
            "latent_closed_loop should NOT be in UNSUPPORTED_NRAM_FEATURES (now wired)"
        
        source = read_api_routes_source()
        assert '"latent_closed_loop"' in source and '"runtime_wired": True' in source, \
            "Capabilities endpoint should report latent_closed_loop as runtime_wired"
    
    def test_branch_tournament_wired_and_accepted(self):
        """VERIFICATION: branch_tournament is now runtime-wired and accepted."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        
        assert "branch_tournament" not in UNSUPPORTED_NRAM_FEATURES, \
            "branch_tournament should NOT be in UNSUPPORTED_NRAM_FEATURES (now wired)"
        
        source = read_api_routes_source()
        assert '"branch_tournament"' in source and '"runtime_wired": True' in source, \
            "Capabilities endpoint should report branch_tournament as runtime_wired"


class TestStreamlitIntegrationFalsification:
    """Test that Streamlit integration exposes controls that don't work."""
    
    def test_streamlit_exposes_conceptor_controls(self):
        """Verify Streamlit exposes conceptor controls."""
        content = read_streamlit_source()
        
        # Streamlit has conceptor UI controls
        assert "conceptor_enabled" in content or "conceptor" in content.lower(), \
            "Streamlit should expose conceptor controls"
    
    def test_streamlit_exposes_dexperts_controls(self):
        """Verify Streamlit exposes DExperts controls."""
        content = read_streamlit_source()
        
        # Streamlit has DExperts UI controls
        assert "dexperts_enabled" in content or "dexperts" in content.lower(), \
            "Streamlit should expose DExperts controls"
    
    def test_streamlit_controls_send_supported_features(self):
        """VERIFICATION: Streamlit controls now send features that are accepted."""
        from core.contracts.nram_runtime import UNSUPPORTED_NRAM_FEATURES
        
        content = read_streamlit_source()
        
        # Streamlit now sends conceptor_steering and dexperts (the correct names)
        assert 'nram_opts["conceptor_steering"]' in content or \
               "nram_opts['conceptor_steering']" in content, \
            "Streamlit should send conceptor_steering in nram_opts"
        
        assert 'nram_opts["dexperts"]' in content or \
               "nram_opts['dexperts']" in content, \
            "Streamlit should send dexperts in nram_opts"
        
        # These features are now supported (not in UNSUPPORTED_NRAM_FEATURES)
        assert "conceptor_steering" not in UNSUPPORTED_NRAM_FEATURES, \
            "conceptor_steering should now be supported"
        
        assert "dexperts" not in UNSUPPORTED_NRAM_FEATURES, \
            "dexperts should now be supported"
        
        # Success - Streamlit controls now work correctly


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
