#!/usr/bin/env python3
"""
Model Vergelijking Script voor Embedding Modellen
Vergelijkt verschillende Ollama embedding modellen op basis van visies en queries
"""

import os
import sys
from typing import List, Dict, Tuple
from pathlib import Path
import json
from datetime import datetime

# Import onze eigen modules
from remote_ollama_embeddings import RemoteOllamaEmbeddingTokenizer, EmbeddingComparator, EmbeddingRecord
from markdown_parser import split_markdown_sections

# Hardcoded lijst van modellen om te testen
MODELS_TO_TEST = [
    "nomic-embed-text",
    "mxbai-embed-large", 
    "all-minilm",
    "bge-large",
    'bge-m3',
    'V4lentin1879/jina-bert-code-f16',
    'olfh/teuken-7b-instruct-commercial-v0.4:7b-instruct-q4_K_M', 
    'olfh/teuken-7b-instruct-commercial-v0.4:7b-instruct-q8_0',
    'snowflake-arctic-embed2',
    'jobautomation/OpenEuroLLM-Dutch',
]

class ModelComparison:
    def __init__(self):
        self.tokenizer = RemoteOllamaEmbeddingTokenizer()
        self.comparator = EmbeddingComparator()
        self.results_dir = Path("comparison_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def embed_visies(self, visies: List[Dict[str, str]]) -> Dict[str, List[EmbeddingRecord]]:
        """
        Embed alle visies met alle modellen
        Return: {model_name: [EmbeddingRecord, ...]}
        """
        model_embeddings = {}
        
        print("=== Embedden van Visies ===")
        
        for model in MODELS_TO_TEST:
            print(f"\nModel: {model}")
            model_embeddings[model] = []
            
            for i, visie in enumerate(visies, 1):
                try:
                    # Combineer titel en content voor embedding
                    full_text = f"{visie['title']}: {visie['content']}"
                    
                    print(f"  {i}/{len(visies)}: {visie['title'][:50]}...")
                    
                    record = self.tokenizer.embed_text(full_text, model=model)
                    embedding_id = self.tokenizer.save_embedding(record)
                    
                    # Voeg metadata toe voor makkelijke referentie
                    record.title = visie['title']
                    record.section_content = visie['content']
                    
                    model_embeddings[model].append(record)
                    
                except Exception as e:
                    print(f"    ✗ Fout met {model}: {e}")
                    continue
        
        return model_embeddings
    
    def find_top_matches(self, query_record: EmbeddingRecord, 
                        visie_records: List[EmbeddingRecord], 
                        top_k: int = 3) -> List[Tuple[EmbeddingRecord, float]]:
        """
        Vind top K beste matches voor een query
        Return: [(EmbeddingRecord, similarity_score), ...]
        """
        matches = []
        
        for visie_record in visie_records:
            similarity = self.comparator.cosine_similarity(
                query_record.embedding, 
                visie_record.embedding
            )
            matches.append((visie_record, similarity))
        
        # Sorteer op similarity score (hoogste eerst)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:top_k]
    
    def process_queries(self, queries: List[Dict[str, str]], 
                       model_embeddings: Dict[str, List[EmbeddingRecord]]) -> Dict:
        """
        Verwerk alle queries tegen alle model embeddings
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'models_tested': MODELS_TO_TEST,
            'num_visies': len(list(model_embeddings.values())[0]) if model_embeddings else 0,
            'num_queries': len(queries),
            'comparisons': []
        }
        
        print("\n=== Verwerken van Queries ===")
        
        for query_idx, query in enumerate(queries, 1):
            print(f"\nQuery {query_idx}/{len(queries)}: {query['title']}")
            
            query_result = {
                'query_title': query['title'],
                'query_content': query['content'],
                'model_results': {}
            }
            
            # Test elke query tegen elk model
            for model in MODELS_TO_TEST:
                if model not in model_embeddings:
                    continue
                    
                try:
                    # Embed de query met dit model
                    query_text = f"{query['title']}: {query['content']}"
                    query_record = self.tokenizer.embed_text(query_text, model=model)
                    
                    # Vind beste matches
                    top_matches = self.find_top_matches(
                        query_record, 
                        model_embeddings[model], 
                        top_k=3
                    )
                    
                    # Sla resultaten op
                    model_results = []
                    for match_record, similarity in top_matches:
                        model_results.append({
                            'visie_title': getattr(match_record, 'title', 'Unknown'),
                            'visie_content': getattr(match_record, 'section_content', '')[:200] + '...',
                            'similarity_score': float(similarity),
                            'embedding_id': match_record.embedding_id
                        })
                    
                    query_result['model_results'][model] = model_results
                    
                    # Print top resultaat
                    if model_results:
                        best_match = model_results[0]
                        print(f"  {model}: {best_match['visie_title']} (score: {best_match['similarity_score']:.4f})")
                
                except Exception as e:
                    print(f"  ✗ Fout met {model}: {e}")
                    query_result['model_results'][model] = []
            
            results['comparisons'].append(query_result)
        
        return results
    
    def generate_report(self, results: Dict):
        """Genereer een leesbaar rapport van de resultaten"""
        
        print("\n" + "="*80)
        print("MODEL VERGELIJKING RAPPORT")
        print("="*80)
        
        print(f"Tijdstip: {results['timestamp']}")
        print(f"Modellen getest: {', '.join(results['models_tested'])}")
        print(f"Aantal visies: {results['num_visies']}")
        print(f"Aantal queries: {results['num_queries']}")
        
        # Analyse per query
        for comparison in results['comparisons']:
            print(f"\n{'='*60}")
            print(f"QUERY: {comparison['query_title']}")
            print(f"{'='*60}")
            print(f"Inhoud: {comparison['query_content'][:150]}...")
            
            print(f"\nTop 3 resultaten per model:")
            
            for model, model_results in comparison['model_results'].items():
                print(f"\n  📊 {model.upper()}:")
                
                if not model_results:
                    print("    (Geen resultaten)")
                    continue
                
                for i, result in enumerate(model_results, 1):
                    print(f"    {i}. {result['visie_title']} (score: {result['similarity_score']:.4f})")
                    print(f"       {result['visie_content'][:100]}...")
        
        # Samenvatting
        print(f"\n{'='*80}")
        print("SAMENVATTING")
        print("="*80)
        
        # Bereken gemiddelde scores per model
        model_avg_scores = {}
        for model in results['models_tested']:
            scores = []
            for comparison in results['comparisons']:
                if model in comparison['model_results']:
                    model_results = comparison['model_results'][model]
                    if model_results:
                        # Neem de beste score voor deze query
                        scores.append(model_results[0]['similarity_score'])
            
            if scores:
                model_avg_scores[model] = sum(scores) / len(scores)
            else:
                model_avg_scores[model] = 0.0
        
        # Sorteer modellen op gemiddelde score
        sorted_models = sorted(model_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\nModel Ranking (op basis van gemiddelde beste match score):")
        for i, (model, avg_score) in enumerate(sorted_models, 1):
            print(f"  {i}. {model}: {avg_score:.4f}")
        
        # Sla gedetailleerde resultaten op
        results_file = self.results_dir / f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nGedetailleerde resultaten opgeslagen in: {results_file}")

def main():
    """Hoofdfunctie voor model vergelijking"""
    
    # Controleer of bestanden bestaan
    if not Path('visies.md').exists() or not Path('queries.md').exists():
        print("❌ Bestanden 'visies.md' en/of 'queries.md' niet gevonden!")
        print("Voer eerst 'python markdown_parser.py' uit om voorbeeldbestanden te maken")
        return
    
    # Laad markdown bestanden
    print("📁 Laden van markdown bestanden...")
    visies = split_markdown_sections('visies.md')
    queries = split_markdown_sections('queries.md')
    
    if not visies:
        print("❌ Geen visies gevonden in visies.md")
        return
    
    if not queries:
        print("❌ Geen queries gevonden in queries.md")
        return
    
    print(f"✅ Geladen: {len(visies)} visies, {len(queries)} queries")
    
    # Start vergelijking
    comparison = ModelComparison()
    
    # Embed alle visies
    model_embeddings = comparison.embed_visies(visies)
    
    if not model_embeddings:
        print("❌ Geen embeddings gegenereerd")
        return
    
    # Verwerk queries
    results = comparison.process_queries(queries, model_embeddings)
    
    # Genereer rapport
    comparison.generate_report(results)

if __name__ == "__main__":
    main()

