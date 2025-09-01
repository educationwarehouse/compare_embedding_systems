import re
from typing import List, Dict

def split_markdown_sections(file_path: str) -> List[Dict[str, str]]:
    """
    Split markdown bestand op basis van # titels
    Return lijst van {'title': str, 'content': str} dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Bestand {file_path} niet gevonden!")
        return []

    # Split op regels die beginnen met # gevolgd door spatie
    sections = re.split(r'(?m)^# (.+)$', text)
    
    # sections[0] is tekst voor eerste # header (waarschijnlijk leeg)
    # daarna afwisselend titel, content
    parsed_sections = []
    
    for i in range(1, len(sections), 2):
        title = sections[i].strip()
        content = sections[i + 1].strip() if i + 1 < len(sections) else ""
        if content:  # Skip lege secties
            parsed_sections.append({
                'title': title,
                'content': content
            })
    
    return parsed_sections

def create_sample_files():
    """Maak voorbeeld markdown bestanden voor testing"""
    
    # Voorbeeld visies.md
    visies_content = """# Duurzaamheid en Milieu
Onze organisatie streeft naar een volledig circulaire economie waarbij alle materialen worden hergebruikt. We geloven dat technologie en natuur hand in hand moeten gaan voor een duurzame toekomst.

# Innovatie en Technologie  
Wij omarmen emerging technologies zoals AI, blockchain en IoT om maatschappelijke uitdagingen op te lossen. Innovation is de sleutel tot vooruitgang.

# Sociale Rechtvaardigheid
Gelijke kansen voor iedereen, ongeacht achtergrond. We streven naar inclusieve gemeenschappen waar diversiteit wordt gevierd en ieders stem wordt gehoord.

# Onderwijs en Ontwikkeling
Levenslang leren is essentieel in onze snel veranderende wereld. We investeren in educatieve programma's die mensen voorbereiden op de banen van de toekomst.

# Gemeenschapsvorming
Sterke lokale gemeenschappen zijn de basis van een gezonde samenleving. We faciliteren verbindingen tussen mensen en groepen om sociale cohesie te bevorderen.
"""

    # Voorbeeld queries.md
    queries_content = """# Hoe kunnen we klimaatverandering tegengaan?
Deze vraag richt zich op concrete acties en strategieën om de opwarming van de aarde tegen te gaan en de impact van menselijke activiteiten te verminderen.

# Wat is de rol van kunstmatige intelligentie in de toekomst?
Een onderzoek naar hoe AI onze samenleving zal transformeren en welke kansen en risico's dit met zich meebrengt.

# Hoe zorgen we voor meer gelijkheid in de samenleving?
Strategieën en beleidsmaatregelen om ongelijkheden te verminderen en eerlijke kansen voor iedereen te creëren.

# Welke vaardigheden hebben mensen nodig voor de arbeidsmarkt van morgen?
Een analyse van toekomstige competenties en hoe we mensen kunnen voorbereiden op veranderende werkomstandigheden.

# Hoe kunnen technologie en gemeenschappen elkaar versterken?
Onderzoek naar de synergie tussen digitale innovatie en lokale gemeenschapsontwikkeling.
"""

    with open('visies.md', 'w', encoding='utf-8') as f:
        f.write(visies_content)
    
    with open('queries.md', 'w', encoding='utf-8') as f:
        f.write(queries_content)
    
    print("✓ Voorbeeld bestanden 'visies.md' en 'queries.md' aangemaakt")

if __name__ == "__main__":
    # Test functionaliteit
    create_sample_files()
    
    visies = split_markdown_sections('visies.md')
    queries = split_markdown_sections('queries.md')
    
    print(f"Gevonden {len(visies)} visies en {len(queries)} queries")
    print("\nVoorbeeld visie:")
    if visies:
        print(f"Titel: {visies[0]['title']}")
        print(f"Content: {visies[0]['content'][:100]}...")

