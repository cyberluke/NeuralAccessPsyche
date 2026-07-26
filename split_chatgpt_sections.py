"""
Split ChatGPT document into sections based on section titles
"""
import re
from pathlib import Path

def split_chatgpt_document():
    """Split ChatGPT document into sections based on section titles."""
    
    # Read the ChatGPT document
    chatgpt_file = Path("C:/Users/lukes.COREI9/Downloads/chatgpt.txt")
    content = chatgpt_file.read_text(encoding='utf-8')
    
    # Create output directory
    output_dir = Path("d:/_SATIN_AI/NeuralAccessPsyche/chatgpt_sections")
    output_dir.mkdir(exist_ok=True)
    
    # Define section titles we know from the document
    section_titles = [
        "Shrnutí pro vedení",
        "Výchozí stav ChatGPT a veřejné signály OpenAI v roce 2026",
        "Pravděpodobné technické trajektorie do roku 2035",
        "Spotřebitelské scénáře pro rok 2035",
        "Integrace ChatGPT a V271",
        "Produktová roadmapa V271 zarovnaná s pravděpodobným vývojem ChatGPT",
        "Roadmapa po etapách",
        "Prioritní backlog podle obchodní důležitosti",
        "Závislosti roadmapy",
        "Milníky pro rok 2035",
        "Monetizace, go-to-market, governance a experimenty k okamžitému spuštění",
        "Executive roadmap summary"
    ]
    
    # Split content by section titles
    sections = []
    lines = content.split('\n')
    
    current_section = {"title": "Introduction", "content": []}
    
    for line in lines:
        # Check if this line is a section title
        is_title = False
        for title in section_titles:
            if title in line and len(line.strip()) < 100:
                # Save previous section
                if current_section["content"]:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": title,
                    "content": [line]
                }
                is_title = True
                break
        
        if not is_title:
            current_section["content"].append(line)
    
    # Add last section
    if current_section["content"]:
        sections.append(current_section)
    
    # Save each section
    for i, section in enumerate(sections, 1):
        # Clean title for filename
        title_clean = re.sub(r'[^\w\s-]', '', section["title"])
        title_clean = re.sub(r'[-\s]+', '-', title_clean)
        title_clean = title_clean.strip('-').lower()[:50]  # Limit length
        
        # Create filename
        filename = f"chatgpt_{i:02d}_{title_clean}.md"
        filepath = output_dir / filename
        
        # Write section
        section_content = '\n'.join(section["content"])
        filepath.write_text(section_content, encoding='utf-8')
        
        print(f"✓ Section {i}: {section['title']} ({len(section_content)} chars)")
    
    print(f"\n✓ Split ChatGPT document into {len(sections)} sections")
    print(f"✓ Sections saved to: {output_dir}")
    
    # Create index file
    index_file = output_dir / "00_index.md"
    index_content = "# ChatGPT Document Sections\n\n"
    for i, section in enumerate(sections, 1):
        title_clean = re.sub(r'[^\w\s-]', '', section["title"])
        title_clean = re.sub(r'[-\s]+', '-', title_clean)
        title_clean = title_clean.strip('-').lower()[:50]
        filename = f"chatgpt_{i:02d}_{title_clean}.md"
        index_content += f"{i}. {section['title']}\n"
        index_content += f"   File: {filename}\n\n"
    
    index_file.write_text(index_content, encoding='utf-8')
    print(f"✓ Created index file: {index_file}")

if __name__ == "__main__":
    split_chatgpt_document()
