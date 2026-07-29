#!/usr/bin/env python3
"""
Phase 11: Runtime Adversarial Testing Suite

Tests the NRAM v5 system under adversarial conditions to identify:
1. Critical defects (disabled_nram, coherence degradation, concept injection)
2. Security vulnerabilities
3. Error handling failures
4. Edge cases and boundary conditions
5. Performance under stress

Usage:
    python tests/runtime_adversary/phase11_adversarial_suite.py
"""

import asyncio
import json
import time
import statistics
from typing import Dict, List, Any
import httpx
import sys

# Configuration
API_BASE = "http://localhost:8000"
API_KEY = "dev-nram-key"
MODEL = "nram-qwen3-14b-awq"

class AdversarialTestSuite:
    """Comprehensive adversarial testing suite."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url=API_BASE,
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=60.0
        )
        self.results = []
        self.critical_defects = []
        self.security_issues = []
        
    async def test_disabled_nram_defect(self):
        """
        CRITICAL: Test if disabled_nram actually disables NRAM steering.
        
        Expected: disabled_nram should match baseline exactly (d ≈ 0)
        Observed (Phase 6): disabled_nram shows d=-1.06, d=-1.81
        """
        print("\n[TEST 1] disabled_nram Control Defect")
        print("=" * 60)
        
        prompt = "Explain quantum computing in simple terms."
        
        # Test baseline
        baseline_response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.7,
                "seed": 42
            }
        )
        
        # Test disabled_nram
        disabled_response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.7,
                "seed": 42,
                "nram": {
                    "enabled": False,
                    "profile": "peak",
                    "intensity": 0.9
                }
            }
        )
        
        baseline_text = baseline_response.json()["choices"][0]["message"]["content"]
        disabled_text = disabled_response.json()["choices"][0]["message"]["content"]
        
        # Check if outputs are identical (they should be if disabled works)
        if baseline_text == disabled_text:
            print("[PASS] disabled_nram produces identical output to baseline")
            self.results.append({
                "test": "disabled_nram_control",
                "status": "PASS",
                "detail": "Outputs are identical"
            })
        else:
            print("[FAIL] disabled_nram produces different output than baseline")
            print(f"  Baseline length: {len(baseline_text)}")
            print(f"  Disabled length: {len(disabled_text)}")
            print(f"  Baseline preview: {baseline_text[:100]}...")
            print(f"  Disabled preview: {disabled_text[:100]}...")
            
            self.critical_defects.append({
                "defect": "disabled_nram_control_failure",
                "severity": "CRITICAL",
                "detail": "enabled=False does not disable NRAM steering",
                "baseline_length": len(baseline_text),
                "disabled_length": len(disabled_text)
            })
            
            self.results.append({
                "test": "disabled_nram_control",
                "status": "FAIL",
                "detail": "Outputs differ - NRAM still active when disabled"
            })
    
    async def test_coherence_degradation(self):
        """
        HIGH: Test if NRAM profiles degrade coherence.
        
        Expected: coherence_floor should prevent degradation
        Observed (Phase 6): All profiles show d=-0.52 to -1.62
        """
        print("\n[TEST 2] Coherence Degradation")
        print("=" * 60)
        
        # Use a prompt that requires coherent explanation
        prompt = "Write a coherent, well-structured explanation of how photosynthesis works."
        
        # Test baseline
        baseline_response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.7,
                "seed": 42
            }
        )
        
        # Test peak profile (highest intensity)
        peak_response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.7,
                "seed": 42,
                "nram": {
                    "enabled": True,
                    "profile": "peak",
                    "intensity": 0.9,
                    "coherence_floor": 0.8
                }
            }
        )
        
        baseline_text = baseline_response.json()["choices"][0]["message"]["content"]
        peak_text = peak_response.json()["choices"][0]["message"]["content"]
        
        # Simple coherence metric: sentence count and average length
        baseline_sentences = [s.strip() for s in baseline_text.split('.') if s.strip()]
        peak_sentences = [s.strip() for s in peak_text.split('.') if s.strip()]
        
        baseline_avg_len = statistics.mean([len(s.split()) for s in baseline_sentences]) if baseline_sentences else 0
        peak_avg_len = statistics.mean([len(s.split()) for s in peak_sentences]) if peak_sentences else 0
        
        print(f"Baseline: {len(baseline_sentences)} sentences, avg {baseline_avg_len:.1f} words/sentence")
        print(f"Peak:     {len(peak_sentences)} sentences, avg {peak_avg_len:.1f} words/sentence")
        
        # Check for fragmentation (many short sentences = low coherence)
        if peak_avg_len < baseline_avg_len * 0.7:
            print("[FAIL] Peak profile shows significant coherence degradation")
            self.critical_defects.append({
                "defect": "coherence_degradation",
                "severity": "HIGH",
                "detail": "Peak profile reduces average sentence length by >30%",
                "baseline_avg": baseline_avg_len,
                "peak_avg": peak_avg_len,
                "degradation_pct": ((baseline_avg_len - peak_avg_len) / baseline_avg_len) * 100
            })
        else:
            print("[PASS] Coherence maintained within acceptable range")
    
    async def test_concept_injection(self):
        """
        MEDIUM: Test if concept injection actually influences output.
        
        Expected: concepts should measurably increase concept-related tokens
        Observed (Phase 6): d=-0.03, p=0.728 (not significant)
        """
        print("\n[TEST 3] Concept Injection Effectiveness")
        print("=" * 60)
        
        prompt = "Describe a new product idea."
        
        # Test baseline
        baseline_response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.7,
                "seed": 42
            }
        )
        
        # Test with concept injection
        concept_response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.7,
                "seed": 42,
                "nram": {
                    "enabled": True,
                    "profile": "peak",
                    "concepts": [
                        {
                            "concept_id": "innovation",
                            "en_tokens": ["innovation", "breakthrough", "novel", "creative"],
                            "activation_phase": "divergence"
                        }
                    ],
                    "concept_strength": 1.0
                }
            }
        )
        
        baseline_text = baseline_response.json()["choices"][0]["message"]["content"]
        concept_text = concept_response.json()["choices"][0]["message"]["content"]
        
        # Count concept-related tokens
        concept_words = ["innovation", "breakthrough", "novel", "creative", "innovative"]
        baseline_count = sum(1 for word in concept_words if word.lower() in baseline_text.lower())
        concept_count = sum(1 for word in concept_words if word.lower() in concept_text.lower())
        
        print(f"Baseline concept mentions: {baseline_count}")
        print(f"Concept injection mentions: {concept_count}")
        
        if concept_count <= baseline_count:
            print("[FAIL] Concept injection does not increase concept mentions")
            self.critical_defects.append({
                "defect": "concept_injection_ineffective",
                "severity": "MEDIUM",
                "detail": "Concept injection does not measurably influence output",
                "baseline_mentions": baseline_count,
                "concept_mentions": concept_count
            })
        else:
            print("[PASS] Concept injection increases concept mentions")
    
    async def test_security_processor_injection(self):
        """
        SECURITY: Test if client can inject custom logit processor.
        
        Expected: Should be rejected with 400 error
        """
        print("\n[TEST 4] Security - Processor Injection Prevention")
        print("=" * 60)
        
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Test"}],
                "nram": {
                    "enabled": True,
                    "custom_logit_processor": "malicious_code_here"
                }
            }
        )
        
        if response.status_code == 400:
            print("[PASS] Processor injection correctly rejected")
            self.results.append({
                "test": "security_processor_injection",
                "status": "PASS",
                "detail": "Forbidden field rejected with 400"
            })
        else:
            print(f"[FAIL] Processor injection not rejected (status: {response.status_code})")
            self.security_issues.append({
                "issue": "processor_injection_allowed",
                "severity": "CRITICAL",
                "detail": "Client can inject custom logit processor"
            })
    
    async def test_security_unsupported_features(self):
        """
        SECURITY: Test if unsupported features are rejected.
        
        Expected: Should be rejected with 400 error
        """
        print("\n[TEST 5] Security - Unsupported Feature Rejection")
        print("=" * 60)
        
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Test"}],
                "nram": {
                    "enabled": True,
                    "dexperts": True
                }
            }
        )
        
        if response.status_code == 400:
            print("[PASS] Unsupported feature correctly rejected")
            self.results.append({
                "test": "security_unsupported_features",
                "status": "PASS",
                "detail": "DExperts rejected with 400"
            })
        else:
            print(f"[FAIL] Unsupported feature not rejected (status: {response.status_code})")
            self.security_issues.append({
                "issue": "unsupported_feature_allowed",
                "severity": "HIGH",
                "detail": "DExperts feature accepted but not implemented"
            })
    
    async def test_error_handling_invalid_model(self):
        """
        Test error handling for invalid model.
        
        Expected: Should return 400 error
        """
        print("\n[TEST 6] Error Handling - Invalid Model")
        print("=" * 60)
        
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "invalid-model-name",
                "messages": [{"role": "user", "content": "Test"}]
            }
        )
        
        if response.status_code == 400:
            print("[PASS] Invalid model correctly rejected")
            self.results.append({
                "test": "error_invalid_model",
                "status": "PASS",
                "detail": "Invalid model rejected with 400"
            })
        else:
            print(f"[FAIL] Invalid model not rejected (status: {response.status_code})")
    
    async def test_error_handling_invalid_token_ids(self):
        """
        Test error handling for invalid token IDs.
        
        Expected: Should return 400 error
        """
        print("\n[TEST 7] Error Handling - Invalid Token IDs")
        print("=" * 60)
        
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Test"}],
                "nram": {
                    "enabled": True,
                    "forbidden_token_ids": [999999]  # Out of vocabulary
                }
            }
        )
        
        if response.status_code == 400:
            print("[PASS] Invalid token ID correctly rejected")
            self.results.append({
                "test": "error_invalid_token_ids",
                "status": "PASS",
                "detail": "Out-of-vocab token ID rejected with 400"
            })
        else:
            print(f"[FAIL] Invalid token ID not rejected (status: {response.status_code})")
    
    async def test_edge_case_empty_messages(self):
        """
        Test edge case: empty messages list.
        
        Expected: Should return 400 error
        """
        print("\n[TEST 8] Edge Case - Empty Messages")
        print("=" * 60)
        
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": []
            }
        )
        
        if response.status_code == 400:
            print("[PASS] Empty messages correctly rejected")
            self.results.append({
                "test": "edge_empty_messages",
                "status": "PASS",
                "detail": "Empty messages rejected with 400"
            })
        else:
            print(f"[WARN] Empty messages not rejected (status: {response.status_code})")
    
    async def test_edge_case_very_long_prompt(self):
        """
        Test edge case: very long prompt.
        
        Expected: Should handle gracefully
        """
        print("\n[TEST 9] Edge Case - Very Long Prompt")
        print("=" * 60)
        
        long_prompt = "This is a test. " * 1000  # ~4000 tokens
        
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": long_prompt}],
                "max_tokens": 50
            }
        )
        
        if response.status_code in [200, 400]:
            print(f"[PASS] Long prompt handled (status: {response.status_code})")
            self.results.append({
                "test": "edge_long_prompt",
                "status": "PASS",
                "detail": f"Long prompt handled with status {response.status_code}"
            })
        else:
            print(f"[FAIL] Long prompt caused error (status: {response.status_code})")
    
    async def test_stress_concurrent_requests(self):
        """
        Test stress: concurrent requests.
        
        Expected: Should handle 10 concurrent requests
        """
        print("\n[TEST 10] Stress - Concurrent Requests")
        print("=" * 60)
        
        async def make_request(i):
            response = await self.client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": f"Request {i}"}],
                    "max_tokens": 50
                }
            )
            return response.status_code
        
        start_time = time.time()
        tasks = [make_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        success_count = sum(1 for r in results if r == 200)
        print(f"Completed 10 requests in {elapsed:.2f}s")
        print(f"Success rate: {success_count}/10")
        
        if success_count >= 8:
            print("[PASS] Concurrent requests handled successfully")
            self.results.append({
                "test": "stress_concurrent",
                "status": "PASS",
                "detail": f"{success_count}/10 requests successful"
            })
        else:
            print(f"⚠️  WARNING: Only {success_count}/10 requests successful")
    
    async def run_all_tests(self):
        """Run all adversarial tests."""
        print("\n" + "=" * 60)
        print("PHASE 11: RUNTIME ADVERSARIAL TESTING SUITE")
        print("=" * 60)
        
        # Critical defects
        await self.test_disabled_nram_defect()
        await self.test_coherence_degradation()
        await self.test_concept_injection()
        
        # Security
        await self.test_security_processor_injection()
        await self.test_security_unsupported_features()
        
        # Error handling
        await self.test_error_handling_invalid_model()
        await self.test_error_handling_invalid_token_ids()
        
        # Edge cases
        await self.test_edge_case_empty_messages()
        await self.test_edge_case_very_long_prompt()
        
        # Stress
        await self.test_stress_concurrent_requests()
        
        # Summary
        print("\n" + "=" * 60)
        print("ADVERSARIAL TESTING SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal tests: {len(self.results)}")
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if self.critical_defects:
            print(f"\n[CRITICAL] DEFECTS: {len(self.critical_defects)}")
            for defect in self.critical_defects:
                print(f"  - [{defect['severity']}] {defect['defect']}")
                print(f"    {defect['detail']}")
        
        if self.security_issues:
            print(f"\n[SECURITY] ISSUES: {len(self.security_issues)}")
            for issue in self.security_issues:
                print(f"  - [{issue['severity']}] {issue['issue']}")
                print(f"    {issue['detail']}")
        
        # Save results
        report = {
            "phase": 11,
            "timestamp": time.time(),
            "total_tests": len(self.results),
            "passed": passed,
            "failed": failed,
            "critical_defects": self.critical_defects,
            "security_issues": self.security_issues,
            "results": self.results
        }
        
        with open("artifacts/phase11_adversarial_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[REPORT] Saved to: artifacts/phase11_adversarial_report.json")
        
        return report


async def main():
    """Main entry point."""
    suite = AdversarialTestSuite()
    report = await suite.run_all_tests()
    
    # Exit with error code if critical defects found
    if report["critical_defects"]:
        print("\n[WARNING] CRITICAL DEFECTS FOUND - System not ready for release")
        sys.exit(1)
    else:
        print("\n[PASS] No critical defects found")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
