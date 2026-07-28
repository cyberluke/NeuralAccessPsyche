"""
Section-by-section A/B test: Compare NRAM outputs with ChatGPT document sections
"""
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
import os
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer test-key"
}

OUTPUT_DIR = Path("d:/_SATIN_AI/NeuralAccessPsyche/ab_section_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define all sections from ChatGPT document with their context
SECTIONS = [
    {
        "id": "01",
        "name": "Shrnutí pro vedení",
        "heading": "## Shrnutí pro vedení",
        "context": """Tato sekce poskytuje executive summary celé strategické analýzy ChatGPT do roku 2035 a strategie V271. 
Zahrnuje: hlavní zjištění, investiční tezi pro V271, doporučenou strategii rozdělenou do 4 etap (0-2 roky, 3-5 let, 6-10 let, 2035+).
Klíčové body: V271 by nemělo soutěžit s ChatGPT v obecné konverzační inteligenci, ale stát se consumer continuity layer pro Evropu.""",
        "requirements": """Vytvoř executive summary v rozsahu 4-5 odstavců. Zahrň:
1. Hlavní zjištění o vývoji ChatGPT (paměť, agenti, multimodalita, apps)
2. Investiční teze pro V271 (consumer continuity layer)
3. Strategie rozdělená do 4 etap s časovými horizonty
4. Závěrečné doporučení pro leadership"""
    },
    {
        "id": "02",
        "name": "Výchozí stav ChatGPT 2026",
        "heading": "## Výchozí stav ChatGPT a veřejné signály OpenAI v roce 2026",
        "context": """Analýza aktuálního stavu ChatGPT k červenci 2026. Identifikuje 6 klíčových směrů vývoje:
1. Paměť a dlouhodobá kontinuita (saved memories, project-only memory)
2. Agentní akce (Operator, CUA, agent mode)
3. Výzkum a konektory (deep research, chat search, MCP)
4. Multimodalita a realtime audio (voice mode, Realtime API)
5. Workflow-specific surfaces (study mode, record mode, projects)
6. Platformizace (GPT Store, Apps in ChatGPT, Apps SDK)

Důležité statistiky: 800M+ týdenních uživatelů, 4M vývojářů.""",
        "requirements": """Vytvoř detailní analýzu výchozího stavu ChatGPT v roce 2026:
1. Úvodní odstavec s kontextem (2-3 věty)
2. 6 číslovaných sekcí pro každý směr vývoje s konkrétními příklady
3. Odstavec o důležitosti těchto signálů (včetně statistik)
4. Závěrečná analýza toho, co to znamená pro V271"""
    },
    {
        "id": "03",
        "name": "Tabulka veřejných signálů",
        "heading": "### Tabulka veřejných signálů a jejich význam pro predikci roku 2035",
        "context": """Tabulka mapuje veřejné signály z roku 2026 na jejich důsledky pro V271.
Signály: Memory/chat history, ChatGPT agent/Operator/CUA, Deep research/konektory, GPT Store/Apps, Realtime voice, Responses API/MCP.""",
        "requirements": """Vytvoř Markdown tabulku se 3 sloupci:
| Veřejný signál k roku 2026 | Co říká o vývoji ChatGPT | Důsledek pro V271 |

Zahrň 6 řádků pro všechny klíčové signály. Každá buňka by měla obsahovat 1-2 věty."""
    },
    {
        "id": "04",
        "name": "Technické trajektorie do 2035",
        "heading": "## Pravděpodobné technické trajektorie do roku 2035",
        "context": """Analýza 6 klíčových trendů:
1. Multimodální agenti (agentní orchestrátor napříč textem, hlasem, obrazem)
2. Persistentní paměť (kontinuitní profil uživatele)
3. Agent marketplace (ekosystém aplikací s reputací)
4. On-device a private inference (hybridní běh mezi zařízením a cloudem)
5. World-model planning (simulace alternativ postupu)
6. Realtime avatar presence (jeden AI profil přes text, hlas, avatar)
7. Identity a reputace (agent passports, ověřená oprávnění)""",
        "requirements": """Vytvoř detailní analýzu každého trendu:
1. Úvodní odstavec o hlavní trajektorii
2. 7 číslovaných sekcí pro každý trend
3. Každá sekce by měla obsahovat: popis trendu, veřejné důkazy z 2024-2026, pravděpodobný stav v 2035, úroveň jistoty (vysoká/střední/nízká)
4. Závěrečná syntéza"""
    },
    {
        "id": "05",
        "name": "Tabulka trajektorií",
        "heading": "### Tabulka trajektorií s hypotézami pro rok 2035",
        "context": """Tabulka převádí trendy na realistické hypotézy pro rok 2035 s úrovněmi jistoty.""",
        "requirements": """Vytvoř Markdown tabulku se 4 sloupci:
| Trajektorie | Veřejný důkaz 2024-2026 | Pravděpodobný stav v roce 2035 | Jistota odhadu |

Zahrň 7 řádků pro všechny trajektorie. Buňky by měly obsahovat 1-2 věty."""
    },
    {
        "id": "06",
        "name": "Spotřebitelské scénáře 2035",
        "heading": "## Spotřebitelské scénáře pro rok 2035",
        "context": """Popis 3 hlavních scénářů:
1. Osobní AI life console (dynamický briefing, kontinuální pracovní plocha)
2. Consumer agent commerce (schopnosti místo appků, kurátor a trust broker)
3. Ambientní multimodalita (hlas, desktop, mobil, avatar)

Včetně popisů 3 mockupů: Continuity Hub, Federovaný Agent Store, Realtime Avatar Session.""",
        "requirements": """Vytvoř detailní popis scénářů:
1. Úvodní odstavec o změně paradigmatu (méně chatbot, více kontinuita)
2. 3 číslované sekce pro každý scénář s konkrétními příklady použití
3. 3 podsekce pro mockupy (A, B, C) s popisem UI/UX
4. Závěrečná analýza pro V271"""
    },
    {
        "id": "07",
        "name": "Integrace ChatGPT a V271",
        "heading": "## Integrace ChatGPT a V271",
        "context": """5 integračních bodů:
1. V271 MCP Gateway (read-only a action-safe konektory)
2. Research object exchange (import/export research packů)
3. Identity a continuity bridge (exportovatelné profiles, consent scopes)
4. Federovaný Agent Store (agent card schema, reputační metadata)
5. Distribution arbitrage (ChatGPT jako akviziční kanál)""",
        "requirements": """Vytvoř detailní analýzu integrace:
1. Úvodní odstavec o principu platforma + continuity shell
2. 5 číslovaných sekcí pro každý integrační bod
3. Každá sekce by měla obsahovat: co přesně vybudovat, proč je to důležité, technické detaily
4. Závěrečná doporučení"""
    },
    {
        "id": "08",
        "name": "Tabulka integrací",
        "heading": "### Praktický backlog integrací",
        "context": """Tabulka prioritizuje integrační body podle důležitosti.""",
        "requirements": """Vytvoř Markdown tabulku se 4 sloupci:
| Integrace | Co přesně vybudovat | Proč je to důležité | Priorita |

Zahrň 7 řádků: V271 MCP Gateway, Research object exchange, Consent ledger, Agent card schema, Receipts a provenance, Native-to-ChatGPT app presence, Credential wallet.
Priority: P0, P1, P2."""
    },
    {
        "id": "09",
        "name": "Produktová roadmapa V271",
        "heading": "## Produktová roadmapa V271 zarovnaná s pravděpodobným vývojem ChatGPT",
        "context": """Roadmapa je postavena proti pravděpodobným milníkům ekosystému, ne proti nereálné ambici.
Pracovní hypotézy: 0-2 roky (apps/agents/konektory), 3-5 let (standardy tool interoperability), 6-10 let (kontinuita, identita, multimodalita).""",
        "requirements": """Vytvoř úvodní analýzu roadmapy:
1. Odstavec o principu zarovnání na trendy (ne neveřejné plány OpenAI)
2. 3 pracovní hypotézy s časovými horizonty
3. Analýza baseline předpokladů pro V271
4. Důležité upozornění o tom, co je inference vs. veřejně potvrzený požadavek"""
    },
    {
        "id": "10",
        "name": "Roadmapa po etapách",
        "heading": "### Roadmapa po etapách",
        "context": """4 etapy: 0-2 roky (continuity shell), 3-5 let (federovaný agent commerce layer), 6-10 let (osobní AI OS), 2035/2036 (stabilizovaný cílový stav).""",
        "requirements": """Vytvoř Markdown tabulku se 6 sloupci:
| Etapa | Hlavní cíl | Klíčové featury | Závislosti | Doporučený tým | Hlavní rizika |

Zahrň 4 řádky pro všechny etapy. Buňky by měly obsahovat 1-3 věty."""
    },
    {
        "id": "11",
        "name": "Prioritní backlog",
        "heading": "### Prioritní backlog podle obchodní důležitosti",
        "context": """Prioritizace features: P0 (Continuity graph, Consent ledger, MCP gateway), P1 (Research object exchange, Receipts, Federated Agent Store), P2 (Realtime avatar, On-device submodely), P3 (Scenario simulator).""",
        "requirements": """Vytvoř Markdown tabulku se 4 sloupci:
| Priorita | Feature | Důvod priority | Jak souvisí s vývojem ChatGPT |

Zahrň 9 řádků pro všechny features."""
    },
    {
        "id": "12",
        "name": "Závislosti roadmapy",
        "heading": "### Závislosti roadmapy",
        "context": """Dependency model: nejprve interní datová a trust vrstva, potom distribuce, marketplace a ambientní multimodalita.""",
        "requirements": """Vytvoř analýzu závislostí:
1. Odstavec o principu dependency modelu
2. Vizuální popis dependency graphu (textově)
3. Vysvětlení proč je důležité stavět v tomto pořadí
4. Rizika pokud se pořadí nedodrží"""
    },
    {
        "id": "13",
        "name": "Milníky pro rok 2035",
        "heading": "### Milníky pro rok 2035",
        "context": """Cílový stav formulovaný jako produktová věta: V271 je osobní AI shell pro kontinuitu, agenty, dokumenty, delegování akcí a monetizaci bez závislosti na jedné platformě.""",
        "requirements": """Vytvoř analýzu milníků:
1. Odstavec s produktovou větou pro rok 2035
2. Analýza co je potřeba pro dosažení tohoto stavu
3. Rizika pokud se nepostaví základní trust a continuity vrstva
4. Závěrečné doporučení"""
    },
    {
        "id": "14",
        "name": "Monetizace a experimenty",
        "heading": "## Monetizace, go-to-market, governance a experimenty k okamžitému spuštění",
        "context": """4 monetizační linie: consumer subscription, revenue share z agentů, privacy-premium, creator economy.
3 go-to-market segmenty: knowledge-heavy individuals, privacy-sensitive Europeans, creators/curators.
4 regulatorní témata: AI Act, GDPR, biometrická citlivost, IP/provenance.
3 principy architektury důvěry: scope-by-scope consent, receipts by default, portability.
9 experimentů s prioritami.""",
        "requirements": """Vytvoř komplexní analýzu:
1. 4 sekce pro monetizační linie
2. 3 sekce pro go-to-market segmenty
3. 4 sekce pro regulatorní témata
4. 3 sekce pro principy důvěry
5. Markdown tabulka s 9 experimenty (Experiment, Hypotéza, Jak měřit úspěch, Priorita)"""
    },
    {
        "id": "15",
        "name": "Executive roadmap summary",
        "heading": "## Executive roadmap summary",
        "context": """8 klíčových rozhodnutí pro leadership + konkrétní další kroky (30, 60, 90, 180 dnů, 12 měsíců).""",
        "requirements": """Vytvoř executive summary:
1. 8 číslovaných rozhodnutí (zadrté, zadruhé, atd.)
2. Konkrétní kroky s časovými horizonty (30, 60, 90, 180 dnů, 12 měsíců)
3. Závěrečný odstavec o oddělení toho co víme vs. co je vize
4. Finální doporučení pro V271"""
    }
]

# Consciousness states to test
STATES = [
    ("persona-normal", "normal"),
    ("persona-microdose", "microdose"),
    ("persona-threshold", "threshold"),
    ("persona-psychedelic", "psychedelic"),
    ("persona-peak", "peak"),
    ("persona-dissociative", "dissociative"),
    ("nram-moe-orchestrator", "moe")
]


def generate_section_output(section: dict, model: str, state_name: str) -> dict:
    """Generate output for a specific section and model."""
    
    prompt = f"""Na základě následujícího kontextu z ChatGPT strategického dokumentu vytvoř sekci "{section['name']}".

KONTEXT SEKCE:
{section['context']}

POŽADAVKY:
{section['requirements']}

DŮLEŽITÉ:
- Používej češtinu
- Buď strukturovaný a detailní
- Zahrň konkrétní příklady, data a doporučení
- Pokud jsou požadovány tabulky, vytvoř je v Markdown formátu
- Dodržuj požadovaný formát (odstavce, číslované sekce, tabulky)
"""
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        
        return {
            "success": True,
            "content": content,
            "length": len(content),
            "tokens": usage.get("completion_tokens", 0),
            "model": model,
            "state": state_name
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": model,
            "state": state_name
        }


def save_section_output(section_id: str, section_name: str, state_name: str, content: str, metadata: dict):
    """Save section output to file."""
    filename = f"section_{section_id}_{state_name}.md"
    filepath = OUTPUT_DIR / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {section_name}\n\n")
        f.write(f"**Model**: {metadata['model']}\n")
        f.write(f"**Stav**: {state_name}\n")
        f.write(f"**Délka**: {metadata['length']} znaků\n")
        f.write(f"**Tokeny**: {metadata['tokens']}\n\n")
        f.write("---\n\n")
        f.write(content)
    
    return filepath


def main():
    """Run section-by-section A/B test."""
    print("="*80)
    print("SECTION-BY-SECTION A/B TEST")
    print("="*80)
    print(f"\nCelkem sekcí: {len(SECTIONS)}")
    print(f"Celkem stavů: {len(STATES)}")
    print(f"Celkem generování: {len(SECTIONS) * len(STATES)}")
    print(f"\nVýstupy budou uloženy do: {OUTPUT_DIR}")
    print("\n" + "="*80)
    
    results = []
    
    for section in SECTIONS:
        print(f"\n{'='*80}")
        print(f"SEKCE {section['id']}: {section['name']}")
        print(f"{'='*80}")
        
        for model, state_name in STATES:
            print(f"\n  Generuji pro stav: {state_name}...", end=" ")
            
            result = generate_section_output(section, model, state_name)
            
            if result["success"]:
                filepath = save_section_output(
                    section["id"],
                    section["name"],
                    state_name,
                    result["content"],
                    result
                )
                print(f"✓ ({result['length']} znaků, {result['tokens']} tokenů)")
                result["filepath"] = str(filepath)
            else:
                print(f"✗ Chyba: {result['error']}")
            
            results.append({
                "section_id": section["id"],
                "section_name": section["name"],
                **result
            })
    
    # Save summary
    summary_path = OUTPUT_DIR / "test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print("TEST DOKONČEN")
    print("="*80)
    print(f"\nCelkem úspěšných generování: {sum(1 for r in results if r['success'])}/{len(results)}")
    print(f"Souhrn uložen do: {summary_path}")
    print(f"\nVšechny výstupy jsou v: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
