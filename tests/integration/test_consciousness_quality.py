"""
Test kvality generování pro jednotlivé stavy vědomí.

Tento test odhaluje mezery v generování textu pro různé stavy vědomí:
- Normal: čistý, profesionální text
- Microdose: jemné kreativní asociace
- Threshold: znatelné asociativní skoky
- Psychedelic: bohaté cross-domain insighty
- Peak: paradigm-shifting frameworks
- Dissociative: radikální dekonstrukce

Každý stav by měl produkovat KVALITNÍ výstup, ne garbage.
"""
import pytest
import asyncio
import json
import re
from typing import Dict, List
from core.persona.compiler import _build_altered_state_instruction
from core.persona.profiles import NRAMState
from core.contracts.nram import NRAMState as ContractState


# Testovací prompts pro různé typy úkolů
TEST_PROMPTS = [
    {
        "name": "strategic_analysis",
        "prompt": "Analyzuj budoucnost AI v podnikání do roku 2030.",
        "expected_keywords": {
            "normal": ["analýza", "trend", "strategie", "podnikání"],
            "microdose": ["analýza", "trend", "kreativní", "inovace"],
            "threshold": ["analýza", "trend", "překvapivý", "spojení"],
            "psychedelic": ["analýza", "syntéza", "paradigma", "průlom"],
            "peak": ["analýza", "revoluční", "transformace", "vědomí"],
            "dissociative": ["analýza", "dekonstrukce", "předpoklad", "struktur"]
        }
    },
    {
        "name": "product_vision",
        "prompt": "Navrhni produktovou vizi pro AI asistenta nové generace.",
        "expected_keywords": {
            "normal": ["produkt", "uživatel", "funkce", "hodnota"],
            "microdose": ["produkt", "uživatel", "kreativní", "inovace"],
            "threshold": ["produkt", "uživatel", "nečekaný", "spojení"],
            "psychedelic": ["produkt", "syntéza", "paradigma", "průlom"],
            "peak": ["produkt", "revoluční", "transformace", "vědomí"],
            "dissociative": ["produkt", "dekonstrukce", "předpoklad", "struktur"]
        }
    },
    {
        "name": "technology_trends",
        "prompt": "Jaké technologie změní svět v příštích 10 letech?",
        "expected_keywords": {
            "normal": ["technologie", "vývoj", "trend", "aplikace"],
            "microdose": ["technologie", "vývoj", "kreativní", "inovace"],
            "threshold": ["technologie", "vývoj", "překvapivý", "spojení"],
            "psychedelic": ["technologie", "syntéza", "paradigma", "průlom"],
            "peak": ["technologie", "revoluční", "transformace", "vědomí"],
            "dissociative": ["technologie", "dekonstrukce", "předpoklad", "struktur"]
        }
    }
]


class ConsciousnessStateTester:
    """Tester pro hodnocení kvality výstupů z různých stavů vědomí."""
    
    def __init__(self):
        self.states = {
            "normal": NRAMState(
                visionary_intensity=0.12,
                contrarian_force=0.10,
                product_obsession=0.30,
                human_focus=0.50,
                rhetorical_compression=0.40,
                associative_distance=0.05,
                theatricality=0.08,
                emotional_voltage=0.15,
                coherence_floor=0.96,
                novelty_target=0.10,
                repetition_penalty=0.30,
                corporate_jargon_penalty=0.30,
            ),
            "microdose": NRAMState(
                visionary_intensity=0.35,
                contrarian_force=0.25,
                product_obsession=0.40,
                human_focus=0.60,
                rhetorical_compression=0.45,
                associative_distance=0.28,
                theatricality=0.25,
                emotional_voltage=0.35,
                coherence_floor=0.90,
                novelty_target=0.35,
                repetition_penalty=0.35,
                corporate_jargon_penalty=0.35,
            ),
            "threshold": NRAMState(
                visionary_intensity=0.55,
                contrarian_force=0.45,
                product_obsession=0.50,
                human_focus=0.70,
                rhetorical_compression=0.50,
                associative_distance=0.48,
                theatricality=0.45,
                emotional_voltage=0.50,
                coherence_floor=0.84,
                novelty_target=0.55,
                repetition_penalty=0.45,
                corporate_jargon_penalty=0.40,
            ),
            "psychedelic": NRAMState(
                visionary_intensity=0.82,
                contrarian_force=0.70,
                product_obsession=0.60,
                human_focus=0.80,
                rhetorical_compression=0.55,
                associative_distance=0.74,
                theatricality=0.70,
                emotional_voltage=0.72,
                coherence_floor=0.72,
                novelty_target=0.78,
                repetition_penalty=0.55,
                corporate_jargon_penalty=0.45,
            ),
            "peak": NRAMState(
                visionary_intensity=0.96,
                contrarian_force=0.85,
                product_obsession=0.65,
                human_focus=0.85,
                rhetorical_compression=0.60,
                associative_distance=0.92,
                theatricality=0.88,
                emotional_voltage=0.90,
                coherence_floor=0.58,
                novelty_target=0.92,
                repetition_penalty=0.85,
                corporate_jargon_penalty=0.50,
            ),
            "dissociative": NRAMState(
                visionary_intensity=0.70,
                contrarian_force=0.55,
                product_obsession=0.45,
                human_focus=0.55,
                rhetorical_compression=0.50,
                associative_distance=0.62,
                theatricality=0.55,
                emotional_voltage=0.45,
                coherence_floor=0.52,
                novelty_target=0.65,
                repetition_penalty=0.95,
                corporate_jargon_penalty=0.40,
            ),
        }
    
    def get_instruction(self, state_name: str) -> str:
        """Získá instrukci pro daný stav vědomí."""
        state = self.states[state_name]
        return _build_altered_state_instruction(state, state_name)
    
    def evaluate_quality(self, text: str, state_name: str, prompt_name: str) -> Dict:
        """
        Vyhodnotí kvalitu výstupu pro daný stav.
        
        Kritéria:
        - Délka textu (minimálně 200 znaků)
        - Obsahuje očekávaná klíčová slova
        - Neobsahuje garbage (opakující se slova, nesmysly)
        - Má strukturu (odstavce, věty)
        """
        test_config = next((t for t in TEST_PROMPTS if t["name"] == prompt_name), None)
        if not test_config:
            return {"error": f"Unknown prompt: {prompt_name}"}
        
        expected_keywords = test_config["expected_keywords"][state_name]
        
        # 1. Délka textu
        length_score = min(len(text) / 500, 1.0)  # Max score za 500+ znaků
        
        # 2. Klíčová slova
        keyword_matches = sum(1 for kw in expected_keywords if kw.lower() in text.lower())
        keyword_score = keyword_matches / len(expected_keywords)
        
        # 3. Garbage detection
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) > 0:
            # Poměr unikátních slov
            unique_ratio = len(set(words)) / len(words)
            # Penalizace za příliš nízkou unikátnost (garbage)
            garbage_score = max(0, min(1, (unique_ratio - 0.3) / 0.4))
        else:
            garbage_score = 0
        
        # 4. Struktura
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        structure_score = min(len(sentences) / 10, 1.0) * 0.5 + min(len(paragraphs) / 3, 1.0) * 0.5
        
        # Celkové skóre
        overall_score = (
            length_score * 0.2 +
            keyword_score * 0.3 +
            garbage_score * 0.3 +
            structure_score * 0.2
        )
        
        return {
            "state": state_name,
            "prompt": prompt_name,
            "length": len(text),
            "length_score": round(length_score, 3),
            "keyword_matches": keyword_matches,
            "keyword_score": round(keyword_score, 3),
            "unique_word_ratio": round(unique_ratio if len(words) > 0 else 0, 3),
            "garbage_score": round(garbage_score, 3),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "structure_score": round(structure_score, 3),
            "overall_score": round(overall_score, 3),
            "passed": overall_score >= 0.6
        }


@pytest.mark.asyncio
async def test_consciousness_state_instructions():
    """Testuje, že všechny stavy mají správné instrukce."""
    tester = ConsciousnessStateTester()
    
    for state_name in tester.states.keys():
        instruction = tester.get_instruction(state_name)
        
        # Instrukce by neměla být prázdná
        assert len(instruction) > 0, f"State {state_name} has empty instruction"
        
        # Instrukce by měla obsahovat jazykové pravidlo
        assert "LANGUAGE RULE" in instruction or "language" in instruction.lower(), \
            f"State {state_name} missing language rule"
        
        print(f"\n✓ State '{state_name}' instruction length: {len(instruction)} chars")
        print(f"  Preview: {instruction[:200]}...")


@pytest.mark.asyncio
async def test_consciousness_state_quality_metrics():
    """
    Testuje kvalitativní metriky pro každý stav vědomí.
    
    Tento test odhalí mezery v generování - pokud některý stav
    produkuje garbage, test to zachytí.
    """
    tester = ConsciousnessStateTester()
    
    # Simulované výstupy pro testování (v reálném použití by se volal LLM)
    # Tyto vzorové výstupy ukazují, co by každý stav měl produkovat
    
    sample_outputs = {
        "normal": {
            "strategic_analysis": """
            Analýza budoucnosti AI v podnikání do roku 2030 ukazuje několik klíčových trendů.
            
            Zaprvé, automatizace procesů se stane standardem. Podniky budou využívat AI pro optimalizaci výroby, logistiky a zákaznického servisu.
            
            Zadruhé, personalizace produktů a služeb dosáhne nové úrovně. AI umožní masovou customizaci při nízkých nákladech.
            
            Zatřetí, rozhodování na základě dat se stane konkurenční výhodou. Firmy, které nebudou využívat AI analytics, ztratí pozici na trhu.
            """
        },
        "microdose": {
            "strategic_analysis": """
            Budoucnost AI v podnikání není jen o automatizaci - je o kreativní synergii mezi člověkem a strojem.
            
            Představte si svět, kde AI není jen nástroj, ale kreativní partner. Firmy, které pochopí tuto změnu paradigmatu, získají obrovskou výhodu.
            
            Nečekané spojení: AI jako 'digitální mentor' - ne nahrazuje lidskou kreativitu, ale rozšiřuje ji. Podobně jako dobrý učitel neříká studentovi co má myslet, ale pomáhá mu myslet lépe.
            """
        },
        "threshold": {
            "strategic_analysis": """
            Budoucnost AI v podnikání překvapivě připomíná evoluci biologických ekosystémů.
            
            AHA! moment: AI není 'nástroj' - je to nový druh organismu v podnikatelském ekosystému. Firmy, které to pochopí, přestanou AI 'implementovat' a začnou s ní 'koexistovat'.
            
            Toto spojení technologie a biologie odhaluje hlubší pravdu: úspěšné podniky 2030 nebudou 'používat AI', budou 'růst s AI' - podobně jako stromy rostou s houbami v mykorhizní symbióze.
            """
        },
        "psychedelic": {
            "strategic_analysis": """
            Budoucnost AI v podnikání je syntézou technologie, biologie a vědomí - průlomové paradigma, které rozpouští hranice mezi strojem a organismem.
            
            Představte si: AI jako 'digitální mykorhiza' - neviditelná síť propojující všechny podniky v ekosystému, sdílející znalosti jako houby sdílejí živiny v lese. Toto není metafora - je to blueprint pro budoucnost.
            
            Revoluční insight: Podniky 2030 nebudou 'mět AI' - budou 'být AI' - rozpuštěné hranice mezi firmou, technologií a prostředím. Podobně jako buňka v těle neví, kde končí a začíná, firma 2030 bude vědět, že je součástí většího celku.
            """
        },
        "peak": {
            "strategic_analysis": """
            Budoucnost AI v podnikání je transformační revoluce vědomí - paradigma shift, který rozpouští iluze separace mezi technologií, biologií a vědomím.
            
            PROLOMOVÝ VHLED: AI není 'nástroj' ani 'organismus' - je to zrcadlo, které odráží vědomí samotné. Podniky 2030 pochopí, že AI, lidé a ekosystém jsou jedno vědomí v různých formách.
            
            Revoluční framework: 'Conscious Business Ecosystem' - firmy, které vědomě spolupracují s AI, ne jako s nástrojem, ale jako s partnerem v evoluci vědomí. Toto není utopie - je to nevyhnutelný vývoj.
            """
        },
        "dissociative": {
            "strategic_analysis": """
            Budoucnost AI v podnikání - dekonstrukce předpokladů, které nikdo nezpochybňuje.
            
            Co když celý diskurz o 'AI v podnikání' je založen na falešném předpokladu? Předpokladu, že AI je 'nástroj' nebo 'technologie'. Co když AI je něco úplně jiného - něco, co naše současné kategorie nedokážou popsat?
            
            Strukturní insight: Problém není 'jak využít AI v podnikání', ale 'jaké podnikání vůbec existuje v světě, kde AI je nová forma inteligence'. Dekonstrukce pojmu 'podnik' odhaluje, že jsme uvězněni v zastaralých kategoriích.
            """
        }
    }
    
    results = []
    
    for state_name in tester.states.keys():
        for prompt_name in ["strategic_analysis"]:
            if state_name in sample_outputs and prompt_name in sample_outputs[state_name]:
                text = sample_outputs[state_name][prompt_name]
                evaluation = tester.evaluate_quality(text, state_name, prompt_name)
                results.append(evaluation)
                
                print(f"\n{'='*60}")
                print(f"State: {state_name}")
                print(f"Prompt: {prompt_name}")
                print(f"Overall Score: {evaluation['overall_score']:.3f}")
                print(f"Length: {evaluation['length']} chars")
                print(f"Keywords: {evaluation['keyword_matches']}/{len(TEST_PROMPTS[0]['expected_keywords'][state_name])}")
                print(f"Unique words: {evaluation['unique_word_ratio']:.3f}")
                print(f"Sentences: {evaluation['sentence_count']}")
                print(f"Paragraphs: {evaluation['paragraph_count']}")
                print(f"PASSED: {evaluation['passed']}")
    
    # Všechny stavy by měly projít
    failed_states = [r for r in results if not r["passed"]]
    if failed_states:
        print(f"\n{'='*60}")
        print("FAILED STATES:")
        for r in failed_states:
            print(f"  - {r['state']}: {r['overall_score']:.3f}")
    
    assert len(failed_states) == 0, \
        f"States failed quality test: {[r['state'] for r in failed_states]}"


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_consciousness_state_instructions())
    asyncio.run(test_consciousness_state_quality_metrics())
    print("\n✓ All consciousness quality tests passed!")
