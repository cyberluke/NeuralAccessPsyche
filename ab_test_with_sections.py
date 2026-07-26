"""
A/B Test: Generate NRAM outputs for each ChatGPT section with delays
"""
import requests
import time
from pathlib import Path
import json

# Configuration
API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer test-key"
}

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

def generate_with_retry(payload, max_retries=3, delay=5):
    """Generate with retry logic and delays."""
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"      Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(delay)
            else:
                raise e

def generate_section_output(section_file, section_title, model, state_name):
    """Generate output for a specific section and model."""
    
    # Read the ChatGPT section
    section_content = section_file.read_text(encoding='utf-8')
    
    # Create state-specific prompts to encourage different perspectives
    state_prompts = {
        'normal': 'Buď strukturovaný, analytický, založený na faktech.',
        'microdose': 'Přidej subtilní kreativní prvky a nečekané souvislosti.',
        'threshold': 'Buď odvážnější, použij nečekané analogie a metafory.',
        'psychedelic': 'Buď vysoce kreativní, poetický, použij originální metafory a nekonvenční perspektivy.',
        'peak': 'Generuj vizionářský, transformační obsah s hlubokými vhledy.',
        'dissociative': 'Analyzuj z odstupu, buď kritický, odhal skryté předpoklady.',
        'moe': 'Integruj různé perspektivy do koherentního celku.'
    }
    
    state_instruction = state_prompts.get(state_name, '')
    
    prompt = f"""Na základě následující sekce z ChatGPT dokumentu vytvoř vlastní verzi této sekce.

SEKCE: {section_title}

OBSAH SEKCE Z CHATGPT:
{section_content}

POŽADAVKY:
- Vytvoř vlastní verzi této sekce
- {state_instruction}
- Nepapouškuj původní obsah, ale generuj originální perspektivu
- Zachov stejné téma, ale přistup k němu kreativně
- Buď strukturovaný a detailní
- Zahrň konkrétní příklady, data a doporučení
- Pokud jsou v původní sekci tabulky, vytvoř je v Markdown formátu
- Odpovídej v češtině
"""
    
    # Add explicit NRAM parameters based on state
    nram_params = {}
    if state_name != 'moe':
        intensity_map = {
            'normal': 0.1,
            'microdose': 0.35,
            'threshold': 0.58,
            'psychedelic': 0.84,
            'peak': 1.0,
            'dissociative': 0.70
        }
        nram_params = {
            "enabled": True,
            "intensity": intensity_map.get(state_name, 0.5)
        }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.7
    }
    
    if nram_params:
        payload["nram"] = nram_params
    
    try:
        result = generate_with_retry(payload)
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

def main():
    """Run A/B test section by section with delays."""
    print("="*80)
    print("A/B TEST: NRAM vs ChatGPT (Section by Section)")
    print("="*80)
    
    # Get all ChatGPT sections
    chatgpt_dir = Path("d:/_SATIN_AI/NeuralAccessPsyche/chatgpt_sections")
    section_files = sorted(chatgpt_dir.glob("chatgpt_*.md"))
    
    # Filter out index file
    section_files = [f for f in section_files if f.name != "00_index.md"]
    
    print(f"\nCelkem sekcí: {len(section_files)}")
    print(f"Celkem stavů: {len(STATES)}")
    print(f"Celkem generování: {len(section_files) * len(STATES)}")
    print(f"\nPauza mezi požadavky: 3 sekundy")
    print("\n" + "="*80)
    
    # Create output directory
    output_dir = Path("d:/_SATIN_AI/NeuralAccessPsyche/ab_test_sections")
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for section_file in section_files:
        # Extract section number and title from filename
        parts = section_file.stem.split('_', 2)
        section_num = parts[1]
        section_title = parts[2].replace('-', ' ').title()
        
        print(f"\n{'='*80}")
        print(f"SEKCE {section_num}: {section_title}")
        print(f"{'='*80}")
        
        for model, state_name in STATES:
            print(f"\n  Generuji pro stav: {state_name}...", end=" ")
            
            result = generate_section_output(section_file, section_title, model, state_name)
            
            if result["success"]:
                # Save output
                output_file = output_dir / f"section_{section_num}_{state_name}.md"
                output_file.write_text(result["content"], encoding='utf-8')
                
                print(f"✓ ({result['length']} znaků, {result['tokens']} tokenů)")
                result["filepath"] = str(output_file)
            else:
                print(f"✗ Chyba: {result['error']}")
            
            results.append({
                "section_num": section_num,
                "section_title": section_title,
                **result
            })
            
            # Delay between requests to avoid overloading
            time.sleep(3)
    
    # Save summary
    summary_path = output_dir / "test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print("TEST DOKONČEN")
    print("="*80)
    print(f"\nCelkem úspěšných generování: {sum(1 for r in results if r['success'])}/{len(results)}")
    print(f"Souhrn uložen do: {summary_path}")
    print(f"\nVšechny výstupy jsou v: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
