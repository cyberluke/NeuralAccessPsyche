"""
Testy pro Hallucination Guard
"""
import pytest
from core.steering.hallucination_guard import (
    HallucinationGuard,
    check_hallucinations,
    sanitize_output
)


class TestHallucinationGuard:
    """Test suite pro hallucination guard."""
    
    def test_numeric_claim_detection(self):
        """Test detekce numerických tvrzení."""
        guard = HallucinationGuard()
        
        # Test percentage
        text = "Model dosáhl 95% přesnosti na benchmarku."
        result = guard.check_output(text)
        assert result.has_hallucinations
        assert any(d.claim_type == "numeric" for d in result.detections)
        
        # Test large numbers
        text = "Společnost má 10 milionů uživatelů."
        result = guard.check_output(text)
        assert result.has_hallucinations
        
        # Test time periods
        text = "Trénovali jsme model 24 hodin."
        result = guard.check_output(text)
        assert result.has_hallucinations
    
    def test_study_claim_detection(self):
        """Test detekce odkazů na studie."""
        guard = HallucinationGuard()
        
        text = "Podle studie z Harvardu je tento přístup efektivní."
        result = guard.check_output(text)
        assert result.has_hallucinations
        assert any(d.claim_type == "study" for d in result.detections)
        
        text = "Benchmark ukázal 50% zlepšení."
        result = guard.check_output(text)
        assert result.has_hallucinations
    
    def test_customer_claim_detection(self):
        """Test detekce zákaznických tvrzení."""
        guard = HallucinationGuard()
        
        text = "Zákazník hlásil 30% zvýšení produktivity."
        result = guard.check_output(text)
        assert result.has_hallucinations
        assert any(d.claim_type == "customer" for d in result.detections)
        
        text = "Fortune 500 společnost používá naši technologii."
        result = guard.check_output(text)
        assert result.has_hallucinations
    
    def test_no_hallucinations(self):
        """Test textu bez halucinací."""
        guard = HallucinationGuard()
        
        text = "Tento model je založen na transformer architektuře."
        result = guard.check_output(text)
        assert not result.has_hallucinations
        assert result.risk_score == 0.0
    
    def test_sanitize_output(self):
        """Test sanitizace výstupu."""
        guard = HallucinationGuard()
        
        text = "Model dosáhl 95% přesnosti podle studie z MIT."
        sanitized = guard.sanitize(text)
        
        assert "[UNVERIFIED:" in sanitized
        assert "95%" in sanitized or "95 %" in sanitized
    
    def test_risk_score_calculation(self):
        """Test výpočtu risk score."""
        guard = HallucinationGuard()
        
        # Single claim
        text1 = "Model má 95% přesnost."
        result1 = guard.check_output(text1)
        
        # Multiple claims
        text2 = "Model má 95% přesnost. Studie z Harvardu ukázala 50% zlepšení. Zákazník hlásil 10 milionů uživatelů."
        result2 = guard.check_output(text2)
        
        # More claims should have higher risk score
        assert result2.risk_score > result1.risk_score
    
    def test_convenience_functions(self):
        """Test pomocných funkcí."""
        text = "Model dosáhl 95% přesnosti."
        
        # check_hallucinations
        result = check_hallucinations(text)
        assert result.has_hallucinations
        
        # sanitize_output
        sanitized = sanitize_output(text)
        assert "[UNVERIFIED:" in sanitized
    
    def test_strict_mode(self):
        """Test strict mode."""
        guard_strict = HallucinationGuard(strict_mode=True)
        guard_normal = HallucinationGuard(strict_mode=False)
        
        # V strict mode by měl být více paranoidní
        # (momentálně implementace nerozlišuje, ale API je připraveno)
        text = "Model je efektivní."
        
        result_strict = guard_strict.check_output(text)
        result_normal = guard_normal.check_output(text)
        
        # Oba by neměly detekovat halucinace v tomto textu
        assert not result_strict.has_hallucinations
        assert not result_normal.has_hallucinations
    
    def test_mixed_content(self):
        """Test textu s různými typy tvrzení."""
        guard = HallucinationGuard()
        
        text = """
        Náš model dosáhl 98% přesnosti na ImageNet benchmarku.
        Podle studie z Stanfordu (2024) je tento přístup revoluční.
        Zákazník z Fortune 500 hlásil 40% snížení nákladů.
        Model má 500 milionů parametrů a trénovali jsme ho 72 hodin.
        """
        
        result = guard.check_output(text)
        assert result.has_hallucinations
        
        # Měl by detekovat všechny typy
        claim_types = {d.claim_type for d in result.detections}
        assert "numeric" in claim_types
        assert "study" in claim_types
        assert "customer" in claim_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
