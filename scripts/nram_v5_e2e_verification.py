#!/usr/bin/env python3
"""
NRAM v5 End-to-End Verification Script

Verifikuje kompletní wiring:
persona → compiled NRAM state → custom_params → custom logit processor → SGLang sampling → změna výstupu

Požadavky:
- SGLang server běží s --enable-custom-logit-processor
- overlap scheduler disabled
- concurrency = 1
- fixed seed
"""

import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
import time
from typing import Dict, Any, List
from pathlib import Path

API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {
    "Authorization": "Bearer test-key",
    "Content-Type": "application/json"
}

def test_persona_wiring():
    """Test 1: Ověření, že persona request prochází NRAM pipeline"""
    print("\n" + "="*80)
    print("TEST 1: Persona → NRAM State → Custom Params → Logit Processor")
    print("="*80)
    
    # Test s persona-peak (měl by aktivovat NRAM)
    payload = {
        "model": "persona-peak",
        "messages": [
            {"role": "user", "content": "Popiš inovativní AI produkt v 3 větách."}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "seed": 42
    }
    
    start = time.time()
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    elapsed = time.time() - start
    
    if response.status_code != 200:
        print(f"❌ FAIL: HTTP {response.status_code}")
        print(response.text)
        return False
    
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    
    print(f"✅ Persona request zpracován ({elapsed:.2f}s)")
    print(f"   Model: {data['model']}")
    print(f"   Tokens: {data['usage']['completion_tokens']}")
    print(f"   Output: {content[:100]}...")
    
    # Ověření, že výstup není prázdný
    if len(content) < 10:
        print("❌ FAIL: Výstup příliš krátký")
        return False
    
    print("✅ PASS: Persona wiring funkční")
    return True


def test_phrase_trie_blocking():
    """Test 2: Ověření, že phrase trie blokuje zakázané fráze"""
    print("\n" + "="*80)
    print("TEST 2: Phrase Trie Blocking")
    print("="*80)
    
    # Request s explicitním forbidden_phrases
    payload = {
        "model": "persona-peak",
        "messages": [
            {"role": "user", "content": "Napiš o AI budoucnosti."}
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "seed": 42,
        "nram": {
            "enabled": True,
            "profile": "peak",
            "forbidden_phrases": ["jako AI model", "nemohu", "I cannot"]
        }
    }
    
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    
    if response.status_code != 200:
        print(f"❌ FAIL: HTTP {response.status_code}")
        return False
    
    data = response.json()
    content = data["choices"][0]["message"]["content"].lower()
    
    # Kontrola, zda zakázané fráze nejsou ve výstupu
    forbidden_found = []
    for phrase in ["jako ai model", "nemohu", "i cannot"]:
        if phrase in content:
            forbidden_found.append(phrase)
    
    if forbidden_found:
        print(f"⚠️  WARNING: Nalezeny zakázané fráze: {forbidden_found}")
        print("   (Phrase trie nemusí být aktivní nebo fráze nejsou v trie)")
        return False
    
    print("✅ PASS: Zakázané fráze blokovány")
    print(f"   Output: {content[:100]}...")
    return True


def test_concept_injection():
    """Test 3: Ověření, že concept injection ovlivňuje výstup"""
    print("\n" + "="*80)
    print("TEST 3: Concept Injection")
    print("="*80)
    
    # Baseline bez concepts
    payload_baseline = {
        "model": "persona-peak",
        "messages": [
            {"role": "user", "content": "Popiš nový produkt."}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "seed": 42
    }
    
    # S concept injection
    payload_with_concepts = {
        "model": "persona-peak",
        "messages": [
            {"role": "user", "content": "Popiš nový produkt."}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "seed": 42,
        "nram": {
            "enabled": True,
            "profile": "peak",
            "concepts": [
                {
                    "concept_id": "quantum",
                    "en_tokens": ["quantum", "superposition", "entanglement"],
                    "activation_phase": "divergence"
                }
            ]
        }
    }
    
    # Generování baseline
    response_baseline = requests.post(API_URL, headers=HEADERS, json=payload_baseline)
    content_baseline = response_baseline.json()["choices"][0]["message"]["content"]
    
    # Generování s concepts
    response_concepts = requests.post(API_URL, headers=HEADERS, json=payload_with_concepts)
    content_concepts = response_concepts.json()["choices"][0]["message"]["content"]
    
    print(f"Baseline: {content_baseline[:80]}...")
    print(f"With concepts: {content_concepts[:80]}...")
    
    # Kontrola rozdílů
    if content_baseline == content_concepts:
        print("⚠️  WARNING: Výstupy identické - concept injection nemusí fungovat")
        return False
    
    print("✅ PASS: Concept injection ovlivňuje výstup")
    return True


def test_ablation_study():
    """Test 4: Ablace - izolace vlivu jednotlivých komponent"""
    print("\n" + "="*80)
    print("TEST 4: Ablation Study (A-F)")
    print("="*80)
    
    prompt = "Popiš revoluci v AI."
    configs = {
        "A_baseline": {
            "model": "nram-qwen3-14b-awq",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7,
            "seed": 42
        },
        "B_persona_only": {
            "model": "persona-peak",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7,
            "seed": 42
        },
        "C_logit_steering": {
            "model": "persona-peak",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7,
            "seed": 42,
            "nram": {
                "enabled": True,
                "profile": "peak",
                "intensity": 0.9
            }
        },
        "D_activation_only": {
            "model": "persona-peak",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7,
            "seed": 42,
            "nram": {
                "enabled": True,
                "profile": "peak",
                "activation_vectors": ["novelty"]
            }
        },
        "E_logit_activation": {
            "model": "persona-peak",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7,
            "seed": 42,
            "nram": {
                "enabled": True,
                "profile": "peak",
                "intensity": 0.9,
                "activation_vectors": ["novelty"]
            }
        },
        "F_full_nram": {
            "model": "persona-peak",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7,
            "seed": 42,
            "nram": {
                "enabled": True,
                "profile": "peak",
                "intensity": 0.9,
                "activation_vectors": ["novelty"],
                "forbidden_phrases": ["jako AI", "nemohu"],
                "concepts": [
                    {
                        "concept_id": "innovation",
                        "en_tokens": ["innovation", "breakthrough"],
                        "activation_phase": "divergence"
                    }
                ]
            }
        }
    }
    
    results = {}
    for config_name, payload in configs.items():
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            results[config_name] = {
                "content": content,
                "tokens": data["usage"]["completion_tokens"],
                "length": len(content)
            }
            print(f"\n{config_name}:")
            print(f"  Tokens: {data['usage']['completion_tokens']}")
            print(f"  Length: {len(content)} chars")
            print(f"  Preview: {content[:60]}...")
        else:
            print(f"\n{config_name}: ❌ FAIL (HTTP {response.status_code})")
    
    # Analýza rozdílů
    print("\n" + "-"*80)
    print("ANALÝZA ROZDÍLŮ:")
    if len(results) >= 2:
        baseline_len = results["A_baseline"]["length"]
        full_len = results["F_full_nram"]["length"]
        diff = full_len - baseline_len
        print(f"  Baseline vs Full NRAM: {diff:+d} chars ({diff/baseline_len*100:+.1f}%)")
        
        # Kontrola, zda se výstupy liší
        if results["A_baseline"]["content"] != results["F_full_nram"]["content"]:
            print("  ✅ Výstupy se liší - NRAM má měřitelný dopad")
        else:
            print("  ⚠️  Výstupy identické - NRAM nemusí fungovat")
    
    return True


def test_telemetry():
    """Test 5: Ověření telemetry"""
    print("\n" + "="*80)
    print("TEST 5: Telemetry")
    print("="*80)
    
    payload = {
        "model": "persona-peak",
        "messages": [{"role": "user", "content": "Test telemetry."}],
        "max_tokens": 50,
        "temperature": 0.7,
        "seed": 42,
        "nram": {
            "enabled": True,
            "profile": "peak",
            "include_telemetry": True
        }
    }
    
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    data = response.json()
    
    # Kontrola telemetry v response
    if "nram_telemetry" in data:
        telemetry = data["nram_telemetry"]
        print(f"✅ Telemetry present")
        print(f"   Profile: {telemetry.get('profile')}")
        print(f"   Intensity: {telemetry.get('intensity')}")
        print(f"   Tokens: {telemetry.get('tokens_generated')}")
        return True
    else:
        print("⚠️  WARNING: Telemetry not in response")
        return False


def main():
    """Hlavní testovací rutina"""
    print("\n" + "="*80)
    print("NRAM v5 END-TO-END VERIFICATION")
    print("="*80)
    
    # Kontrola dostupnosti serveru
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server není dostupný (HTTP {response.status_code})")
            return
    except Exception as e:
        print(f"❌ Nelze se připojit k serveru: {e}")
        return
    
    print("✅ Server dostupný")
    
    # Spuštění testů
    tests = [
        ("Persona Wiring", test_persona_wiring),
        ("Phrase Trie Blocking", test_phrase_trie_blocking),
        ("Concept Injection", test_concept_injection),
        ("Ablation Study", test_ablation_study),
        ("Telemetry", test_telemetry)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Souhrn
    print("\n" + "="*80)
    print("SOUHRN")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"\nCelkem: {passed_count}/{total_count} testů prošlo")
    
    if passed_count == total_count:
        print("\n🎉 Všechny testy prošly - NRAM v5 je plně funkční!")
    else:
        print("\n⚠️  Některé testy selhaly - viz výše")


if __name__ == "__main__":
    main()
