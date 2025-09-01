# Hardcoded lijst van modellen om te testen
MODELS_TO_TEST = [
    # Ollama modellen
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
    "bge-large",
    "bge-m3",
    
    # HuggingFace modellen (worden automatisch gedetecteerd)
    "jegormeister/bert-base-dutch-cased-snli",
    
    "GroNLP/bert-base-dutch-cased",
    "pdelobelle/robbert-v2-dutch-base",
]

# Model type mapping (optioneel, voor expliciete controle)
MODEL_TYPES = {
    "jegormeister/bert-base-dutch-cased-snli": "huggingface",
    "GroNLP/bert-base-dutch-cased": "huggingface",
    "pdelobelle/robbert-v2-dutch-base": "huggingface",
    # Ollama modellen worden auto-gedetecteerd
}
