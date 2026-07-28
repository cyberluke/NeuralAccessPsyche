"""
Hallucination Guard for NRAM v5

Detekuje a filtruje potenciální halucinace ve výstupech modelu.
Implementuje three-tier defense:
1. Numeric claim detection - kontrola číselných tvrzení
2. Study/benchmark claim detection - kontrola odkazů na studie
3. Customer claim detection - kontrola zákaznických tvrzení

Usage:
    guard = HallucinationGuard()
    result = guard.check_output(text)
    if result.has_hallucinations:
        text = guard.sanitize(text)
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ClaimDetection:
    """Výsledek detekce tvrzení."""
    text: str
    claim_type: str  # "numeric", "study", "benchmark", "customer"
    confidence: float  # 0.0-1.0
    evidence_required: bool
    start_pos: int
    end_pos: int


@dataclass
class HallucinationCheckResult:
    """Výsledek kontroly halucinací."""
    has_hallucinations: bool
    detections: List[ClaimDetection]
    risk_score: float  # 0.0-1.0
    sanitized_text: Optional[str] = None


class HallucinationGuard:
    """
    Guard against hallucinations in model outputs.
    
    Detekuje potenciálně problematická tvrzení a označuje je
    pro další verifikaci nebo sanitizaci.
    """
    
    # Numeric claim patterns
    NUMERIC_PATTERNS = [
        r'\b\d+(?:\.\d+)?\s*%',  # Percentages (removed \b at end)
        r'\b\d+(?:\.\d+)?\s*(?:percent|procent)\b',  # Percentages (word form)
        r'\b\d+(?:\.\d+)?\s*(?:million|milionu|milionů|miliony|mld|billion|bilionu|bilionů|biliony|trillion|trilionu|trilionů|triliony)\b',  # Large numbers
        r'\b\d+(?:\.\d+)?\s*(?:hours|hour|hodina|hodiny|hodin|days|day|den|dní|weeks|week|týden|týdny|týdnů|months|month|měsíc|měsíce|měsíců|years|year|rok|roky|let|roku)\b',  # Time periods
        r'\b\d+(?:\.\d+)?\s*(?:users|uživatelů|customers|zákazníků|clients|klientů)\b',  # User counts
        r'\b\d+(?:\.\d+)?\s*(?:GB|MB|TB|PB)\b',  # Data sizes
        r'\b\d+(?:\.\d+)?\s*(?:ms|s|min|h)\b',  # Time units
    ]
    
    # Study/benchmark claim patterns
    STUDY_PATTERNS = [
        r'\b(?:study|studie|research|výzkum|analysis|analýza)\s+(?:by|od|from|z)\b',
        r'\b(?:according to|podle)\s+(?:study|studie|research|výzkum)\b',
        r'\b(?:benchmark|test|experiment)\s+(?:showed|ukázal|demonstrated|prokázal)\b',
        r'\b(?:Harvard|Stanford|MIT|Oxford|Cambridge)\s+(?:study|studie|research|výzkum)\b',
        r'\b\d{4}\s+(?:study|studie|paper|článek|research|výzkum)\b',  # Year + study
    ]
    
    # Customer claim patterns
    CUSTOMER_PATTERNS = [
        r'\b(?:customer|zákazník|client|klient)\s+(?:reported|hlásil|said|řekl|claimed|tvrdil)\b',
        r'\b(?:company|firma|organization|organizace)\s+(?:achieved|dosáhla|reached|dosáhla)\b',
        r'\bFortune\s+500\b',  # Simplified Fortune 500 pattern
        r'\b(?:enterprise|podnik)\s+(?:uses|používá|adopted|přijala)\b',
        r'\b(?:increased|zvýšil|improved|zlepšil)\s+(?:by|o)\s+\d+(?:\.\d+)?\s*%',  # Improvement claims
    ]
    
    def __init__(self, strict_mode: bool = False):
        """
        Inicializace guardu.
        
        Args:
            strict_mode: Pokud True, označí i potenciálně neškodná tvrzení
        """
        self.strict_mode = strict_mode
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Kompilace regex patternů pro rychlejší matching."""
        self.numeric_re = [re.compile(p, re.IGNORECASE) for p in self.NUMERIC_PATTERNS]
        self.study_re = [re.compile(p, re.IGNORECASE) for p in self.STUDY_PATTERNS]
        self.customer_re = [re.compile(p, re.IGNORECASE) for p in self.CUSTOMER_PATTERNS]
    
    def check_output(self, text: str) -> HallucinationCheckResult:
        """
        Kontrola textu na potenciální halucinace.
        
        Args:
            text: Text ke kontrole
            
        Returns:
            HallucinationCheckResult s detekcemi a risk score
        """
        detections = []
        
        # Check numeric claims
        detections.extend(self._check_numeric_claims(text))
        
        # Check study/benchmark claims
        detections.extend(self._check_study_claims(text))
        
        # Check customer claims
        detections.extend(self._check_customer_claims(text))
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(detections)
        
        # Determine if hallucinations detected
        has_hallucinations = len(detections) > 0
        
        # Sanitize if needed
        sanitized_text = self.sanitize(text) if has_hallucinations else None
        
        return HallucinationCheckResult(
            has_hallucinations=has_hallucinations,
            detections=detections,
            risk_score=risk_score,
            sanitized_text=sanitized_text
        )
    
    def _check_numeric_claims(self, text: str) -> List[ClaimDetection]:
        """Detekce numerických tvrzení."""
        detections = []
        
        for pattern in self.numeric_re:
            for match in pattern.finditer(text):
                detections.append(ClaimDetection(
                    text=match.group(),
                    claim_type="numeric",
                    confidence=0.7,
                    evidence_required=True,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        return detections
    
    def _check_study_claims(self, text: str) -> List[ClaimDetection]:
        """Detekce odkazů na studie/benchmarky."""
        detections = []
        
        for pattern in self.study_re:
            for match in pattern.finditer(text):
                detections.append(ClaimDetection(
                    text=match.group(),
                    claim_type="study",
                    confidence=0.8,
                    evidence_required=True,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        return detections
    
    def _check_customer_claims(self, text: str) -> List[ClaimDetection]:
        """Detekce zákaznických tvrzení."""
        detections = []
        
        for pattern in self.customer_re:
            for match in pattern.finditer(text):
                detections.append(ClaimDetection(
                    text=match.group(),
                    claim_type="customer",
                    confidence=0.75,
                    evidence_required=True,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        return detections
    
    def _calculate_risk_score(self, detections: List[ClaimDetection]) -> float:
        """Výpočet risk score na základě detekcí."""
        if not detections:
            return 0.0
        
        # Weighted sum of confidences
        total_confidence = sum(d.confidence for d in detections)
        avg_confidence = total_confidence / len(detections)
        
        # Factor in number of detections
        count_factor = min(1.0, len(detections) / 10.0)
        
        return avg_confidence * (0.5 + 0.5 * count_factor)
    
    def sanitize(self, text: str) -> str:
        """
        Sanitizace textu - označení potenciálních halucinací.
        
        Args:
            text: Text k sanitizaci
            
        Returns:
            Sanitizovaný text s označenými tvrzeními
        """
        result = text
        
        # Mark numeric claims
        for pattern in self.numeric_re:
            result = pattern.sub(
                lambda m: f"[UNVERIFIED: {m.group()}]",
                result
            )
        
        # Mark study claims
        for pattern in self.study_re:
            result = pattern.sub(
                lambda m: f"[UNVERIFIED: {m.group()}]",
                result
            )
        
        # Mark customer claims
        for pattern in self.customer_re:
            result = pattern.sub(
                lambda m: f"[UNVERIFIED: {m.group()}]",
                result
            )
        
        return result
    
    def filter_output(self, text: str, threshold: float = 0.5) -> str:
        """
        Filtrování výstupu - odstranění vysoce rizikových tvrzení.
        
        Args:
            text: Text k filtrování
            threshold: Minimální risk score pro filtrování
            
        Returns:
            Filtrovaný text
        """
        result = self.check_output(text)
        
        if result.risk_score < threshold:
            return text
        
        # Remove or mark high-risk claims
        return self.sanitize(text)


# Convenience function
def check_hallucinations(text: str, strict: bool = False) -> HallucinationCheckResult:
    """
    Rychlá kontrola textu na halucinace.
    
    Args:
        text: Text ke kontrole
        strict: Strict mode
        
    Returns:
        HallucinationCheckResult
    """
    guard = HallucinationGuard(strict_mode=strict)
    return guard.check_output(text)


def sanitize_output(text: str) -> str:
    """
    Rychlá sanitizace textu.
    
    Args:
        text: Text k sanitizaci
        
    Returns:
        Sanitizovaný text
    """
    guard = HallucinationGuard()
    return guard.sanitize(text)
