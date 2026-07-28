"""
Complete missing A/B test sections
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

STATES = [
    ("persona-normal", "normal"),
    ("persona-microdose", "microdose"),
    ("persona-threshold", "threshold"),
    ("persona-psychedelic", "psychedelic"),
    ("persona-peak", "peak"),
    ("persona-dissociative", "dissociative"),
    ("nram-moe-orchestrator", "moe")
]

# Missing combinations to generate
MISSING = {
    "03": ["dissociative", "psychedelic", "peak", "moe"],
    "04": ["normal", "microdose", "threshold", "psychedelic"],
    "05": ["threshold"],
    "09": ["moe"],
    "10": ["normal", "microdose", "threshold", "psychedelic", "peak", "dissociative", "moe"],
    "11": ["normal", "microdose", "threshold", "psychedelic", "peak", "dissociative", "moe"],
}

def generate_with_retry(payload, max_retries=5, delay=10):
    """Generate with retry logic and longer delays."""
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
    
    section_content = section_file.read_text(encoding='utf-8')
    
    prompt = f"""Na základě následující sekce z ChatGPT dokumentu vytvoř vlastní verzi této sekce.

SEKCE: {section_title}

OBSAH SEKCE Z CHATGPT:
{section_content}

POŽADAVKY:
- Vytvoř vlastní verzi této sekce
- Zachov stejné téma a strukturu
- Buď strukturovaný a detailní
- Zahrň konkrétní příklady, data a doporučení
- Pokud jsou v původní sekci tabulky, vytvoř je v Markdown formátu
- Odpovídej v češtině
"""
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.7
    }
    
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
    """Complete missing A/B test sections."""
    print("="*80)
    print("COMPLETING MISSING A/B TEST SECTIONS")
    print("="*80)
    
    chatgpt_dir = Path("d:/_SATIN_AI/NeuralAccessPsyche/chatgpt_sections")
    output_dir = Path("d:/_SATIN_AI/NeuralAccessPsyche/ab_test_sections")
    output_dir.mkdir(exist_ok=True)
    
    section_files = sorted(chatgpt_dir.glob("chatgpt_*.md"))
    section_files = [f for f in section_files if f.name != "00_index.md"]
    
    # Map section numbers to files
    section_map = {}
    for f in section_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            section_num = parts[1]
            section_map[section_num] = f
    
    total_to_generate = sum(len(states) for states in MISSING.values())
    print(f"\nCelkem chybějících generování: {total_to_generate}")
    print(f"Pauza mezi požadavky: 5 sekund")
    print(f"Max retries: 5")
    print("\n" + "="*80)
    
    results = []
    generated = 0
    
    for section_num, missing_states in MISSING.items():
        if section_num not in section_map:
            print(f"\n⚠ Section {section_num} not found in chatgpt_sections")
            continue
        
        section_file = section_map[section_num]
        section_title = section_file.stem.replace(f"chatgpt_{section_num}_", "").replace("-", " ").title()
        
        print(f"\n{'='*80}")
        print(f"SEKCE {section_num}: {section_title}")
        print(f"{'='*80}")
        
        for model, state_name in STATES:
            if state_name not in missing_states:
                continue
            
            print(f"\n  Generuji pro stav: {state_name}...", end=" ")
            
            result = generate_section_output(section_file, section_title, model, state_name)
            
            if result["success"]:
                output_file = output_dir / f"section_{section_num}_{state_name}.md"
                output_file.write_text(result["content"], encoding='utf-8')
                
                print(f"✓ ({result['length']} znaků, {result['tokens']} tokenů)")
                generated += 1
                result["filepath"] = str(output_file)
            else:
                print(f"✗ Chyba: {result['error']}")
            
            results.append({
                "section_num": section_num,
                "section_title": section_title,
                **result
            })
            
            # Longer delay to avoid overloading
            time.sleep(5)
    
    # Save summary
    summary_path = output_dir / "completion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print("COMPLETION FINISHED")
    print("="*80)
    print(f"\nVygenerováno: {generated}/{total_to_generate}")
    print(f"Souhrn uložen do: {summary_path}")
    print(f"\nVšechny výstupy jsou v: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
