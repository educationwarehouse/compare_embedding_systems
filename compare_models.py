#!/usr/bin/env python3
"""
Complete Model & Method Comparison Script
Vergelijkt Ollama en HuggingFace embedding modellen met verschillende similarity methoden
Inclusief intra-model evaluatie en ranking-based accuraatheid
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict

# Import model lijst
from models_to_test import MODELS_TO_TEST, MODEL_TYPES

# Import onze eigen modules
from remote_ollama_embeddings import (
    RemoteOllamaEmbeddingTokenizer, 
    UnifiedEmbeddingTokenizer,        # Je gebruikt alleen deze
    AdvancedEmbeddingComparator, 
    EmbeddingRecord
)
from markdown_parser import split_markdown_sections

# Vergelijkingsmethoden om te testen
COMPARISON_METHODS_TO_TEST = [
    "cosine",           # Klassieke cosine similarity
    "diem",             # Dimension Insensitive Euclidean Metric
    "qb_cosine",        # Query Background normalized cosine
    "angular",          # Angular distance
    "hybrid",           # Gewogen combinatie van beste methoden
    "comprehensive",    # Alle methoden tegelijk (gebruikt hybrid score)
]

# Verwachte matches per query (voor evaluatie)
EXPECTED_MATCHES = {
    "Hoe kunnen we klimaatverandering tegengaan?": "Klimaat Actie",
    "Wat is de rol van kunstmatige intelligentie in de toekomst?": "Technologie en Ethiek", 
    "Hoe zorgen we voor meer gelijkheid in de samenleving?": "Gelijkheid en Inclusie",
    "Welke vaardigheden hebben mensen nodig voor de arbeidsmarkt van morgen?": "Toekomstige Vaardigheden",
    "Hoe kunnen technologie en gemeenschappen elkaar versterken?": "Digitale Gemeenschappen"
}

class CompleteModelMethodComparison:
    def __init__(self):
        self.unified_tokenizer = UnifiedEmbeddingTokenizer()
        self.comparator = AdvancedEmbeddingComparator()
        self.results_dir = Path("comparison_results")
        self.results_dir.mkdir(exist_ok=True)
        self.model_comparators = {}  # Per-model comparators voor QB-normalization
    
    def detect_model_type(self, model: str) -> str:
        """Detecteer automatisch model type"""
        if model in MODEL_TYPES:
            return MODEL_TYPES[model]
        
        # Auto-detectie regels
        if "/" in model or any(marker in model.lower() for marker in ["bert", "robbert", "jegormeister"]):
            return "huggingface"
        else:
            return "ollama"
    
    def embed_visies(self, visies: List[Dict[str, str]]) -> Dict[str, List[EmbeddingRecord]]:
        """Embed alle visies met zowel Ollama als HuggingFace modellen"""
        model_embeddings = {}
        
        print("=== Embedden van Visies (Mixed Ollama + HuggingFace) ===")
        
        for model in MODELS_TO_TEST:
            model_type = self.detect_model_type(model)
            print(f"\nModel: {model} (type: {model_type})")
            model_embeddings[model] = []
            
            for i, visie in enumerate(visies, 1):
                try:
                    # Combineer titel en content voor embedding
                    full_text = f"{visie['title']}: {visie['content']}"
                    
                    print(f"  {i}/{len(visies)}: {visie['title'][:50]}...")
                    
                    record = self.unified_tokenizer.embed_text(
                        full_text, 
                        model=model, 
                        model_type=model_type
                    )
                    
                    embedding_id = self.unified_tokenizer.save_embedding(record)
                    
                    # Voeg metadata toe voor makkelijke referentie
                    record.title = visie['title']
                    record.section_content = visie['content']
                    
                    model_embeddings[model].append(record)
                    
                except Exception as e:
                    print(f"    ✗ Fout met {model} ({model_type}): {e}")
                    continue
        
        # Setup background corpus per model (belangrijk: binnen elk model apart!)
        for model, embeddings_list in model_embeddings.items():
            if embeddings_list:
                model_embeddings_array = [record.embedding for record in embeddings_list]
                # Maak een nieuwe comparator instance per model
                self.model_comparators[model] = AdvancedEmbeddingComparator()
                self.model_comparators[model].set_background_corpus(model_embeddings_array)
                print(f"✓ Background corpus voor {model}: {len(model_embeddings_array)} embeddings")
        
        return model_embeddings
    
    def evaluate_ranking_accuracy(self, query: Dict[str, str], 
                                 rankings: List[Dict], 
                                 expected_match: str) -> Dict:
        """
        Evalueer ranking accuraatheid binnen een model
        Return: accuracy metrics gebaseerd op verwachte match positie
        """
        # Zoek positie van verwachte match
        expected_position = None
        for i, result in enumerate(rankings):
            if result['visie_title'] == expected_match:
                expected_position = i + 1  # 1-indexed
                break
        
        # Bereken intra-model score spreiding
        if len(rankings) >= 2:
            scores = [r['similarity_score'] for r in rankings]
            score_range = max(scores) - min(scores)
            score_std = np.std(scores) if len(scores) > 1 else 0.0
            
            # Relatieve score gap tussen top matches
            top_score = rankings[0]['similarity_score']
            second_score = rankings[1]['similarity_score'] if len(rankings) > 1 else top_score
            relative_gap = (top_score - second_score) / top_score if top_score > 0 else 0.0
        else:
            score_range = 0.0
            score_std = 0.0
            relative_gap = 0.0
        
        return {
            'expected_match': expected_match,
            'expected_position': expected_position,
            'is_top_match': expected_position == 1 if expected_position else False,
            'is_in_top3': expected_position <= 3 if expected_position else False,
            'score_range': float(score_range),
            'score_std': float(score_std),
            'relative_gap': float(relative_gap),  # Hoe duidelijk onderscheidend zijn de scores?
            'discrimination_quality': 'high' if relative_gap > 0.05 else 'medium' if relative_gap > 0.02 else 'low'
        }
    
    def rank_visies_intra_model(self, query: Dict[str, str], 
                               visie_records: Dict[str, List[EmbeddingRecord]], 
                               top_k: int = 5) -> Dict[str, Dict[str, Dict]]:
        """
        Rank visies binnen elk model apart met alle methoden
        Return: {model: {method: {rankings: [], evaluation: {}}}}
        """
        results = {}
        expected_match = EXPECTED_MATCHES.get(query['title'], None)
        
        print(f"\n🔍 Query: {query['title']}")
        if expected_match:
            print(f"   Verwachte match: '{expected_match}'")
        
        for model, embeddings in visie_records.items():
            if not embeddings:
                continue
                
            model_type = self.detect_model_type(model)
            print(f"  📊 Model: {model} ({model_type})")
            results[model] = {}
            
            # Gebruik model-specifieke comparator
            comparator = self.model_comparators.get(model, self.comparator)
            
            try:
                # Embed de query met hetzelfde model
                query_text = f"{query['title']}: {query['content']}"
                query_record = self.unified_tokenizer.embed_text(
                    query_text, 
                    model=model, 
                    model_type=model_type
                )
                
                for method in COMPARISON_METHODS_TO_TEST:
                    try:
                        # Rank binnen dit model
                        ranked = comparator.rank_similarities(
                            query_record, embeddings, method=method, top_k=top_k
                        )
                        
                        # Converteer naar serializable format
                        method_rankings = []
                        for record, similarity in ranked:
                            method_rankings.append({
                                'visie_title': getattr(record, 'title', 'Onbekend'),
                                'visie_content': getattr(record, 'section_content', '')[:100] + '...',
                                'similarity_score': float(similarity),
                                'embedding_id': record.embedding_id,
                                'embedding_type': record.embedding_type
                            })
                        
                        # Evalueer ranking accuraatheid
                        evaluation = {}
                        if expected_match:
                            evaluation = self.evaluate_ranking_accuracy(
                                query, method_rankings, expected_match
                            )
                        
                        results[model][method] = {
                            'rankings': method_rankings,
                            'evaluation': evaluation
                        }
                        
                        # Print evaluatie resultaat
                        if expected_match and evaluation:
                            pos = evaluation.get('expected_position', 'niet gevonden')
                            gap = evaluation.get('relative_gap', 0) * 100
                            quality = evaluation.get('discrimination_quality', 'unknown')
                            print(f"    - {method:12}: pos {pos}, gap {gap:.1f}%, quality {quality}")
                        else:
                            top_title = method_rankings[0]['visie_title'] if method_rankings else 'Geen'
                            top_score = method_rankings[0]['similarity_score'] if method_rankings else 0
                            print(f"    - {method:12}: '{top_title}' ({top_score:.4f})")
                            
                    except Exception as e:
                        print(f"    ✗ Fout met methode {method}: {e}")
                        results[model][method] = {'rankings': [], 'evaluation': {}}
                
            except Exception as e:
                print(f"    ✗ Fout met query embedding voor {model}: {e}")
                for method in COMPARISON_METHODS_TO_TEST:
                    results[model][method] = {'rankings': [], 'evaluation': {}}
        
        return results
    
    def process_queries_intra_model(self, queries: List[Dict[str, str]], 
                                   model_embeddings: Dict[str, List[EmbeddingRecord]]) -> Dict:
        """Verwerk alle queries met intra-model evaluatie"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'models_tested': MODELS_TO_TEST,
            'methods_tested': COMPARISON_METHODS_TO_TEST,
            'evaluation_approach': 'intra_model_ranking_mixed_embeddings',
            'expected_matches': EXPECTED_MATCHES,
            'model_types': {model: self.detect_model_type(model) for model in MODELS_TO_TEST},
            'num_visies': len(list(model_embeddings.values())[0]) if model_embeddings else 0,
            'num_queries': len(queries),
            'query_results': []
        }
        
        print("\n=== Mixed Intra-Model Query Processing ===")
        
        for query_idx, query in enumerate(queries, 1):
            print(f"\n📝 Query {query_idx}/{len(queries)}: {query['title']}")
            
            query_rankings = self.rank_visies_intra_model(query, model_embeddings, top_k=5)
            
            query_result = {
                'query_title': query['title'],
                'query_content': query['content'],
                'expected_match': EXPECTED_MATCHES.get(query['title']),
                'model_method_results': query_rankings
            }
            
            results['query_results'].append(query_result)
        
        return results
    
    def analyze_mixed_model_performance(self, results: Dict) -> Dict:
        """Analyseer prestaties van mixed Ollama/HuggingFace modellen"""
        
        print("\n=== Mixed Model Performance Analyse ===")
        
        performance_analysis = {}
        
        # Groepeer modellen op type
        ollama_models = []
        hf_models = []
        
        for model in MODELS_TO_TEST:
            model_type = self.detect_model_type(model)
            if model_type == "huggingface":
                hf_models.append(model)
            else:
                ollama_models.append(model)
        
        print(f"Ollama modellen: {len(ollama_models)}")
        print(f"HuggingFace modellen: {len(hf_models)}")
        
        # Collect metrics per model+method
        for model in MODELS_TO_TEST:
            model_type = self.detect_model_type(model)
            performance_analysis[model] = {}
            print(f"\n📊 Model: {model} ({model_type})")
            
            method_performances = []
            
            for method in COMPARISON_METHODS_TO_TEST:
                metrics = {
                    'top1_accuracy': 0,      # Percentage correct #1 predictions
                    'top3_accuracy': 0,      # Percentage in top 3
                    'avg_expected_position': 0,  # Gemiddelde positie van verwachte match
                    'avg_discrimination': 0,     # Gemiddelde relative gap
                    'high_quality_discrimination': 0,  # Percentage met goede onderscheiding
                    'num_queries': 0,
                    'model_type': model_type
                }
                
                valid_evaluations = []
                
                # Verzamel evaluaties voor deze model+method combinatie
                for query_result in results['query_results']:
                    model_methods = query_result['model_method_results'].get(model, {})
                    method_result = model_methods.get(method, {})
                    evaluation = method_result.get('evaluation', {})
                    
                    if evaluation and 'expected_position' in evaluation:
                        valid_evaluations.append(evaluation)
                
                if valid_evaluations:
                    metrics['num_queries'] = len(valid_evaluations)
                    
                    # Top-1 accuracy
                    top1_hits = sum(1 for e in valid_evaluations if e.get('is_top_match', False))
                    metrics['top1_accuracy'] = (top1_hits / len(valid_evaluations)) * 100
                    
                    # Top-3 accuracy  
                    top3_hits = sum(1 for e in valid_evaluations if e.get('is_in_top3', False))
                    metrics['top3_accuracy'] = (top3_hits / len(valid_evaluations)) * 100
                    
                    # Gemiddelde positie (alleen voor gevonden matches)
                    positions = [e['expected_position'] for e in valid_evaluations 
                               if e['expected_position'] is not None]
                    if positions:
                        metrics['avg_expected_position'] = np.mean(positions)
                    
                    # Discrimination quality
                    gaps = [e.get('relative_gap', 0) for e in valid_evaluations]
                    metrics['avg_discrimination'] = np.mean(gaps) * 100
                    
                    high_quality = sum(1 for e in valid_evaluations 
                                     if e.get('discrimination_quality') == 'high')
                    metrics['high_quality_discrimination'] = (high_quality / len(valid_evaluations)) * 100
                
                performance_analysis[model][method] = metrics
                
                # Print samenvatting
                if metrics['num_queries'] > 0:
                    print(f"  {method:12} | Top1: {metrics['top1_accuracy']:5.1f}% | "
                          f"Top3: {metrics['top3_accuracy']:5.1f}% | "
                          f"AvgPos: {metrics['avg_expected_position']:4.1f} | "
                          f"Discrim: {metrics['avg_discrimination']:5.1f}%")
                    
                    method_performances.append((
                        method, 
                        metrics['top1_accuracy'], 
                        metrics['top3_accuracy'],
                        metrics['avg_discrimination']
                    ))
                else:
                    print(f"  {method:12} | Geen data beschikbaar")
            
            # Sorteer methoden voor dit model
            if method_performances:
                # Sorteer op top1_accuracy, dan op top3_accuracy, dan op discrimination
                method_performances.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
                print(f"\n  🏆 Beste methoden voor {model}:")
                for i, (method, top1, top3, discrim) in enumerate(method_performances[:3], 1):
                    print(f"    {i}. {method} (Top1: {top1:.1f}%, Top3: {top3:.1f}%, Discrim: {discrim:.1f}%)")
        
        return performance_analysis
    
    def generate_mixed_model_report(self, results: Dict, performance_analysis: Dict):
        """Genereer rapport voor mixed Ollama/HuggingFace vergelijking"""
        
        print("\n" + "="*100)
        print("MIXED MODEL COMPARISON REPORT (Ollama + HuggingFace)")
        print("="*100)
        
        print(f"Evaluatie Methode: Intra-model ranking-based vergelijking")
        print(f"Model types: {results.get('model_types', {})}")
        print(f"Verwachte matches: {len(EXPECTED_MATCHES)} gedefinieerd")
        
        # Groepeer prestaties per model type
        ollama_combos = []
        hf_combos = []
        all_combinations = []
        
        for model in MODELS_TO_TEST:
            model_type = self.detect_model_type(model)
            for method in COMPARISON_METHODS_TO_TEST:
                metrics = performance_analysis.get(model, {}).get(method, {})
                if metrics.get('num_queries', 0) > 0:
                    # Composite score: weight top1 heavily, add top3 and discrimination
                    composite_score = (
                        metrics['top1_accuracy'] * 0.5 +
                        metrics['top3_accuracy'] * 0.3 + 
                        metrics['avg_discrimination'] * 0.2
                    )
                    
                    combo = (
                        model, method, model_type,
                        metrics['top1_accuracy'],
                        metrics['top3_accuracy'],
                        metrics['avg_discrimination'],
                        composite_score
                    )
                    
                    all_combinations.append(combo)
                    
                    if model_type == "huggingface":
                        hf_combos.append(combo)
                    else:
                        ollama_combos.append(combo)
        
        all_combinations.sort(key=lambda x: x[6], reverse=True)  # Sort by composite score
        
        # Overall beste combinaties
        print(f"\n{'='*60}")
        print("TOP MODEL + METHOD COMBINATIES (Mixed)")
        print("="*60)
        
        print("\nRanking op composite accuracy score:")
        for i, (model, method, model_type, top1, top3, discrim, composite) in enumerate(all_combinations[:15], 1):
            type_indicator = "🤗" if model_type == "huggingface" else "🦙"
            print(f"  {i:2d}. {type_indicator} {model:25} + {method:12} | "
                  f"Top1: {top1:5.1f}% | Top3: {top3:5.1f}% | "
                  f"Discrim: {discrim:5.1f}% | Score: {composite:5.1f}")
        
        # Vergelijking tussen model types
        print(f"\n{'='*60}")
        print("OLLAMA vs HUGGINGFACE VERGELIJKING")
        print("="*60)
        
        if ollama_combos and hf_combos:
            ollama_combos.sort(key=lambda x: x[6], reverse=True)
            hf_combos.sort(key=lambda x: x[6], reverse=True)
            
            print(f"\n🦙 Beste Ollama combinatie:")
            best_ollama = ollama_combos[0]
            print(f"   {best_ollama[0]} + {best_ollama[1]}")
            print(f"   Top1: {best_ollama[3]:.1f}% | Top3: {best_ollama[4]:.1f}% | Score: {best_ollama[6]:.1f}")
            
            print(f"\n🤗 Beste HuggingFace combinatie:")
            best_hf = hf_combos[0]
            print(f"   {best_hf[0]} + {best_hf[1]}")
            print(f"   Top1: {best_hf[3]:.1f}% | Top3: {best_hf[4]:.1f}% | Score: {best_hf[6]:.1f}")
            
            # Gemiddelde prestatie per type
            ollama_avg = np.mean([combo[6] for combo in ollama_combos])
            hf_avg = np.mean([combo[6] for combo in hf_combos])
            
            print(f"\n📊 Gemiddelde prestatie:")
            print(f"   🦙 Ollama modellen: {ollama_avg:.1f}")
            print(f"   🤗 HuggingFace modellen: {hf_avg:.1f}")
            
            if hf_avg > ollama_avg:
                print(f"   → HuggingFace presteert {hf_avg - ollama_avg:.1f} punten beter")
            else:
                print(f"   → Ollama presteert {ollama_avg - hf_avg:.1f} punten beter")
        
        # Final recommendations
        print(f"\n{'='*60}")
        print("AANBEVELINGEN (Mixed Model Evaluatie)")
        print("="*60)
        
        if all_combinations:
            best_combo = all_combinations[0]
            type_emoji = "🤗" if best_combo[2] == "huggingface" else "🦙"
            
            print(f"\n🏆 Beste combinatie overall voor Nederlandse semantische matching:")
            print(f"   {type_emoji} Model: {best_combo[0]}")
            print(f"   🔧 Methode: {best_combo[1]}")
            print(f"   📊 Top1 Accuracy: {best_combo[3]:.1f}%")
            print(f"   📊 Top3 Accuracy: {best_combo[4]:.1f}%")
            print(f"   📊 Discrimination Quality: {best_combo[5]:.1f}%")
            
            print(f"\n💡 Deze combinatie:")
            if best_combo[3] >= 80:
                print("   ✅ Excellent - vindt de juiste match in >80% van de gevallen")
            elif best_combo[3] >= 60:
                print("   ✅ Goed - vindt de juiste match in >60% van de gevallen")  
            else:
                print("   ⚠️  Matig - meer tuning nodig voor betere accuracy")
                
            if best_combo[5] >= 10:
                print("   ✅ Hoge discrimination - duidelijk onderscheid tussen matches")
            else:
                print("   ⚠️  Lage discrimination - scores liggen dicht bij elkaar")
                
            if best_combo[2] == "huggingface":
                print("   🎯 HuggingFace model - lokaal draaiend, geen server vereist")
            else:
                print("   🌐 Ollama model - vereist Ollama server")
    
    def save_mixed_results(self, results: Dict, performance_analysis: Dict):
        """Sla mixed model resultaten op"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Hoofdresultaten
        results_file = self.results_dir / f"mixed_model_comparison_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Performance analyse
        analysis_file = self.results_dir / f"mixed_performance_analysis_{timestamp}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(performance_analysis, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Mixed model resultaten opgeslagen:")
        print(f"   - Hoofdresultaten: {results_file}")
        print(f"   - Performance analyse: {analysis_file}")

def main():
    """Hoofdfunctie voor complete mixed model vergelijking"""
    
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
    print(f"🎯 Verwachte matches gedefinieerd voor {len(EXPECTED_MATCHES)} queries")
    
    # Toon model configuratie
    ollama_count = sum(1 for m in MODELS_TO_TEST if "/" not in m and "bert" not in m.lower())
    hf_count = len(MODELS_TO_TEST) - ollama_count
    print(f"🔧 Te testen: {len(MODELS_TO_TEST)} modellen ({ollama_count} Ollama + {hf_count} HuggingFace)")
    
    comparison = CompleteModelMethodComparison()
    
    # Embed alle visies met mixed modellen
    print("\n=== Starting Mixed Model Embedding Process ===")
    model_embeddings = comparison.embed_visies(visies)
    
    if not model_embeddings:
        print("❌ Geen embeddings gegenereerd")
        return
    
    # Verwerk queries met intra-model evaluatie
    results = comparison.process_queries_intra_model(queries, model_embeddings)
    
    # Analyseer mixed model prestaties
    performance_analysis = comparison.analyze_mixed_model_performance(results)
    
    # Genereer uitgebreid rapport
    comparison.generate_mixed_model_report(results, performance_analysis)
    
    # Sla resultaten op
    comparison.save_mixed_results(results, performance_analysis)

if __name__ == "__main__":
    main()

