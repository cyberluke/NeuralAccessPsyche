"""
Analyze differences between consciousness states in A/B test outputs
"""
import os
from pathlib import Path
from collections import defaultdict
import re

def count_words(text):
    """Count words in text."""
    return len(text.split())

def count_unique_phrases(text):
    """Count unique phrases (3-grams)."""
    words = text.lower().split()
    if len(words) < 3:
        return 0
    phrases = set()
    for i in range(len(words) - 2):
        phrase = ' '.join(words[i:i+3])
        phrases.add(phrase)
    return len(phrases)

def analyze_section(section_num):
    """Analyze all consciousness states for a given section."""
    states = ['normal', 'microdose', 'threshold', 'psychedelic', 'peak', 'dissociative', 'moe']
    results = {}
    
    for state in states:
        filename = f"section_{section_num}_{state}.md"
        filepath = Path(f"d:/_SATIN_AI/NeuralAccessPsyche/ab_test_sections/{filename}")
        
        if not filepath.exists():
            continue
        
        content = filepath.read_text(encoding='utf-8')
        
        results[state] = {
            'chars': len(content),
            'words': count_words(content),
            'unique_phrases': count_unique_phrases(content),
            'content': content
        }
    
    return results

def compare_states(results):
    """Compare outputs across consciousness states."""
    if not results:
        return
    
    print("\n" + "="*80)
    print("POROVNÁNÍ CONSCIOUSNESS STATES")
    print("="*80)
    
    # Print statistics
    print("\n📊 STATISTIKY:")
    print(f"{'State':<15} {'Chars':<10} {'Words':<10} {'Unique Phrases':<15}")
    print("-" * 80)
    
    for state, data in results.items():
        print(f"{state:<15} {data['chars']:<10} {data['words']:<10} {data['unique_phrases']:<15}")
    
    # Find most similar and most different
    states = list(results.keys())
    if len(states) >= 2:
        # Compare normal vs psychedelic
        if 'normal' in results and 'psychedelic' in results:
            normal_words = set(results['normal']['content'].lower().split())
            psychedelic_words = set(results['psychedelic']['content'].lower().split())
            
            overlap = len(normal_words & psychedelic_words)
            normal_only = len(normal_words - psychedelic_words)
            psychedelic_only = len(psychedelic_words - normal_words)
            
            print("\n🔍 POROVNÁNÍ NORMAL vs PSYCHEDELIC:")
            print(f"  Slovní zásoba overlap: {overlap} slov")
            print(f"  Pouze v NORMAL: {normal_only} slov")
            print(f"  Pouze v PSYCHEDELIC: {psychedelic_only} slov")
            
            # Show some unique phrases from psychedelic
            if psychedelic_only > 0:
                print(f"\n  Příklady unikátních frází v PSYCHEDELIC:")
                psychedelic_phrases = []
                words = results['psychedelic']['content'].lower().split()
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+3])
                    if phrase not in results['normal']['content'].lower():
                        psychedelic_phrases.append(phrase)
                        if len(psychedelic_phrases) >= 5:
                            break
                
                for i, phrase in enumerate(psychedelic_phrases, 1):
                    print(f"    {i}. {phrase}")

def main():
    """Main analysis function."""
    print("="*80)
    print("ANALÝZA ROZDÍLŮ MEZI CONSCIOUSNESS STATES")
    print("="*80)
    
    # Analyze a few representative sections
    sections_to_analyze = ['01', '05', '10']  # Introduction, Consumer Scenarios, Milestones
    
    for section_num in sections_to_analyze:
        print(f"\n\n{'#'*80}")
        print(f"# SEKCE {section_num}")
        print(f"{'#'*80}")
        
        results = analyze_section(section_num)
        
        if results:
            compare_states(results)
        else:
            print(f"\n⚠️  Žádná data pro sekci {section_num}")
    
    print("\n\n" + "="*80)
    print("ANALÝZA DOKONČENA")
    print("="*80)

if __name__ == "__main__":
    main()
