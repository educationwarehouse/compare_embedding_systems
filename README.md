# Embedding Model Vergelijkings Toolkit

Een uitgebreide toolkit voor het testen en vergelijken van verschillende embedding modellen (Ollama en HuggingFace) met meerdere similarity methoden voor Nederlandse semantische matching.

Onstaan in "intensieve samenwerking" met collega Robin, alswel Perplexity: https://www.perplexity.ai/search/schrijf-me-een-stukje-python-c-CClnjxbjRiOhpBO8Lr0E6Q

## 📋 Overzicht

Dit pakket biedt een complete oplossing voor het evalueren van embedding modellen op Nederlandse teksten. Het ondersteunt zowel lokale HuggingFace modellen als remote Ollama modellen, en test verschillende similarity methoden om de beste combinatie te vinden voor semantische matching.

### Hoofdfuncties

- **Mixed Model Support**: Test zowel Ollama als HuggingFace modellen in één run
- **Meerdere Similarity Methoden**: Cosine, DIEM, QB-normalized, Angular, Hybrid
- **Intra-Model Evaluatie**: Vergelijkt prestaties binnen elk model apart
- **Nederlandse Focus**: Geoptimaliseerd voor Nederlandse semantische matching
- **Automatische Evaluatie**: Ranking-based accuracy metrics
- **Uitgebreide Rapportage**: Gedetailleerde performance analyse

## 🚀 Installatie

### Vereisten

```bash
pip install numpy scipy requests python-dotenv pathlib
pip install llama-index llama-index-embeddings-huggingface llama-index-embeddings-ollama
```

### Ollama Setup (optioneel)

Als je Ollama modellen wilt testen:

1. Installeer Ollama: https://ollama.ai/
2. Start Ollama server: `ollama serve`
3. Download modellen: `ollama pull nomic-embed-text`

### Environment Configuratie

Maak een `.env` bestand (optioneel):

```env
OLLAMA_SERVER_URL=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text
EMBEDDING_STORAGE_DIR=embeddings_db
```

## 📁 Bestandsstructuur

```
├── compare_models.py          # Hoofdscript voor model vergelijking
├── remote_ollama_embeddings.py # Embedding services (Ollama + HuggingFace)
├── models_to_test.py          # Configuratie van te testen modellen
├── markdown_parser.py         # Markdown parser voor test data
├── visies.md                  # Test visies (wordt automatisch aangemaakt)
├── queries.md                 # Test queries (wordt automatisch aangemaakt)
├── embeddings_db/             # Database voor embeddings
└── comparison_results/        # Resultaten van vergelijkingen
```

## 🔧 Configuratie

### Modellen Configureren

Bewerk `models_to_test.py` om je gewenste modellen toe te voegen:

```python
MODELS_TO_TEST = [
    # Ollama modellen
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
    
    # HuggingFace modellen
    "jegormeister/bert-base-dutch-cased-snli",
    "GroNLP/bert-base-dutch-cased",
    "pdelobelle/robbert-v2-dutch-base",
]
```

### Test Data Voorbereiden

Het systeem gebruikt markdown bestanden met sectie headers:

**visies.md** - De documenten om doorheen te zoeken:
```markdown
# Duurzaamheid en Milieu
Onze organisatie streeft naar een volledig circulaire economie...

# Innovatie en Technologie  
Wij omarmen emerging technologies zoals AI...
```

**queries.md** - De zoekopdrachten:
```markdown
# Hoe kunnen we klimaatverandering tegengaan?
Deze vraag richt zich op concrete acties...

# Wat is de rol van kunstmatige intelligentie?
Een onderzoek naar hoe AI onze samenleving...
```

## 🏃‍♂️ Gebruik

### Basis Gebruik

1. **Genereer voorbeeldbestanden** (eerste keer):
```bash
python markdown_parser.py
```

2. **Start volledige model vergelijking**:
```bash
python compare_models.py
```

### Stap-voor-stap Proces

Het script voert automatisch de volgende stappen uit:

1. **Laadt test data** uit `visies.md` en `queries.md`
2. **Embeddings genereren** voor alle visies met alle modellen
3. **Query processing** - zoekt beste matches per model/methode
4. **Performance evaluatie** - berekent accuracy metrics
5. **Rapportage** - genereert uitgebreide analyse
6. **Resultaten opslaan** - JSON bestanden in `comparison_results/`

## 📊 Similarity Methoden

Het systeem test verschillende similarity methoden:

- **`cosine`**: Klassieke cosine similarity
- **`diem`**: Dimension Insensitive Euclidean Metric
- **`qb_cosine`**: Query Background normalized cosine
- **`angular`**: Angular distance gebaseerde similarity
- **`hybrid`**: Gewogen combinatie van beste methoden
- **`comprehensive`**: Alle methoden tegelijk

## 📈 Evaluatie Metrics

Voor elke model+methode combinatie wordt gemeten:

- **Top-1 Accuracy**: Percentage correcte #1 voorspellingen
- **Top-3 Accuracy**: Percentage matches in top 3
- **Average Expected Position**: Gemiddelde positie van verwachte match
- **Discrimination Quality**: Hoe goed onderscheidt het model tussen matches
- **Score Range & Standard Deviation**: Spreiding van similarity scores

## 📋 Resultaten Interpreteren

### Console Output

Tijdens het draaien zie je real-time feedback:

```
🔍 Query: Hoe kunnen we klimaatverandering tegengaan?
   Verwachte match: 'Duurzaamheid en Milieu'
  📊 Model: nomic-embed-text (ollama)
    - cosine      : pos 1, gap 15.2%, quality high
    - diem        : pos 2, gap 8.1%, quality medium
    - qb_cosine   : pos 1, gap 22.3%, quality high
```

### JSON Resultaten

Gedetailleerde resultaten worden opgeslagen in `comparison_results/`:

- `mixed_model_comparison_YYYYMMDD_HHMMSS.json` - Volledige resultaten
- `mixed_performance_analysis_YYYYMMDD_HHMMSS.json` - Performance metrics

### Rapport Samenvatting

Het script genereert een uitgebreid rapport met:

- **Top Model+Methode Combinaties** gerangschikt op prestatie
- **Ollama vs HuggingFace Vergelijking** 
- **Aanbevelingen** voor beste configuratie
- **Performance Analyse** per model type

## 🔍 Voorbeeld Output

```
=== MIXED MODEL COMPARISON REPORT ===

TOP MODEL + METHOD COMBINATIES:
  1. 🤗 jegormeister/bert-base-dutch-cased-snli + qb_cosine | Top1: 80.0% | Score: 85.2
  2. 🦙 nomic-embed-text + hybrid | Top1: 75.0% | Score: 82.1
  3. 🤗 GroNLP/bert-base-dutch-cased + cosine | Top1: 70.0% | Score: 78.5

🏆 Beste combinatie overall:
   🤗 Model: jegormeister/bert-base-dutch-cased-snli
   🔧 Methode: qb_cosine
   📊 Top1 Accuracy: 80.0%
```

## 🛠️ Geavanceerd Gebruik

### Custom Modellen Toevoegen

Voeg nieuwe modellen toe aan `models_to_test.py`:

```python
MODELS_TO_TEST.append("jouw-custom-model")

# Voor expliciete type definitie:
MODEL_TYPES["jouw-custom-model"] = "huggingface"  # of "ollama"
```

### Eigen Test Data

Vervang `visies.md` en `queries.md` met je eigen data. Zorg ervoor dat:

- Elke sectie begint met `# Titel`
- Verwachte matches zijn gedefinieerd in `EXPECTED_MATCHES` in `compare_models.py`

### Background Corpus Aanpassen

Voor QB-normalization kun je de background corpus grootte aanpassen:

```python
# In compare_models.py
comparator = AdvancedEmbeddingComparator(background_corpus_size=200)
```

## 🐛 Troubleshooting

### Ollama Connectie Problemen

```bash
# Check of Ollama draait
curl http://localhost:11434/api/tags

# Start Ollama als het niet draait
ollama serve
```

### HuggingFace Model Download Issues

```bash
# Pre-download modellen
python -c "from transformers import AutoModel; AutoModel.from_pretrained('GroNLP/bert-base-dutch-cased')"
```

### Memory Issues

Voor grote modellen, verhoog je systeem memory of test minder modellen tegelijk.

### Lege Resultaten

- Controleer of `visies.md` en `queries.md` bestaan en gevuld zijn
- Verificeer dat `EXPECTED_MATCHES` correct gedefinieerd is
- Check console output voor error messages

## 📚 API Referentie

### UnifiedEmbeddingTokenizer

```python
tokenizer = UnifiedEmbeddingTokenizer()
record = tokenizer.embed_text("tekst", model="nomic-embed-text", model_type="ollama")
```

### AdvancedEmbeddingComparator

```python
comparator = AdvancedEmbeddingComparator()
comparator.set_background_corpus(embeddings_list)
similarities = comparator.rank_similarities(query_record, candidates, method="hybrid")
```

## 🤝 Bijdragen

Bijdragen zijn welkom! Vooral voor:

- Nieuwe similarity methoden
- Ondersteuning voor meer model types
- Nederlandse language-specific optimalisaties
- Performance verbeteringen

## 📄 Licentie

Dit project is open source. Zie de individuele model licenties voor specifieke gebruiksvoorwaarden.

## 🔗 Links

- [Ollama](https://ollama.ai/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [LlamaIndex](https://docs.llamaindex.ai/)
- [Nederlandse BERT modellen](https://huggingface.co/models?language=nl)
