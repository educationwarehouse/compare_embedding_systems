from dotenv import load_dotenv
import os
import requests
import json
import numpy as np
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Laad .env configuratie
load_dotenv()

# Configuratie variabelen
OLLAMA_SERVER_URL = os.getenv('OLLAMA_SERVER_URL', 'http://localhost:11434')
DEFAULT_MODEL = os.getenv('OLLAMA_MODEL', 'nomic-embed-text')
STORAGE_DIR = os.getenv('EMBEDDING_STORAGE_DIR', 'embeddings_db')

@dataclass
class EmbeddingRecord:
    """Data class voor het opslaan van embedding metadata"""
    text: str
    embedding: np.ndarray
    model: str
    timestamp: str
    embedding_id: str

    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'embedding': self.embedding.tolist(),
            'model': self.model,
            'timestamp': self.timestamp,
            'embedding_id': self.embedding_id
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            text=data['text'],
            embedding=np.array(data['embedding']),
            model=data['model'],
            timestamp=data['timestamp'],
            embedding_id=data['embedding_id']
        )

class RemoteOllamaEmbeddingTokenizer:
    """Ollama remote embedding service met eigen opslagformaat"""
    
    def __init__(self, server_url: str = None, model: str = None, storage_dir: str = None):
        self.server_url = (server_url or OLLAMA_SERVER_URL).rstrip('/')
        self.model = model or DEFAULT_MODEL
        self.storage_dir = Path(storage_dir or STORAGE_DIR)
        self.storage_dir.mkdir(exist_ok=True)
        self.db_path = self.storage_dir / "embeddings.db"
        self._init_database()
        self._validate_connection()

    def _init_database(self):
        """Initialiseer SQLite database voor metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                file_path TEXT NOT NULL,
                embedding_size INTEGER NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def _validate_connection(self):
        """Test connectie met remote Ollama server"""
        try:
            response = requests.get(f"{self.server_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✓ Verbonden met Ollama server op {self.server_url}")
            else:
                print(f"⚠ Waarschuwing: Server antwoordt met status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠ Kan niet verbinden met Ollama server: {e}")

    def generate_embedding_id(self, text: str, model: str = None) -> str:
        """Genereer unieke ID voor embedding inclusief model naam"""
        model_name = model or self.model
        text_hash = hashlib.md5(text.encode()).hexdigest()
        model_hash = hashlib.md5(model_name.encode()).hexdigest()[:4]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"emb_{model_hash}_{timestamp}_{text_hash[:8]}"

    def embed_text(self, text: str, model: str = None) -> EmbeddingRecord:
        """Genereer embedding voor tekst via remote Ollama server"""
        embedding_model = model or self.model
        url = f"{self.server_url}/api/embed"
        
        payload = {
            "model": embedding_model,
            "input": [text]
        }
        
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code != 200:
                raise RuntimeError(f"Embedding request failed: {response.status_code} - {response.text}")

            result = response.json()
            
            if 'embeddings' not in result or len(result['embeddings']) == 0:
                raise RuntimeError("Geen embeddings ontvangen van server")

            embedding = np.array(result['embeddings'][0], dtype=np.float32)

            record = EmbeddingRecord(
                text=text,
                embedding=embedding,
                model=embedding_model,
                timestamp=datetime.now().isoformat(),
                embedding_id=self.generate_embedding_id(text, embedding_model)
            )

            return record

        except requests.exceptions.Timeout:
            raise RuntimeError("Timeout bij het genereren van embedding")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Netwerk fout: {e}")

    def save_embedding(self, record: EmbeddingRecord) -> str:
        """Sla embedding op in eigen formaat (NPY + metadata)"""
        # Sla embedding array op als .npy bestand
        embedding_file = self.storage_dir / f"{record.embedding_id}.npy"
        np.save(embedding_file, record.embedding)

        # Sla metadata op in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO embeddings 
            (id, text, model, timestamp, file_path, embedding_size)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            record.embedding_id,
            record.text,
            record.model,
            record.timestamp,
            str(embedding_file),
            len(record.embedding)
        ))
        conn.commit()
        conn.close()

        # Sla ook JSON metadata op voor backup
        metadata_file = self.storage_dir / f"{record.embedding_id}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'text': record.text,
                'model': record.model,
                'timestamp': record.timestamp,
                'embedding_id': record.embedding_id,
                'embedding_size': len(record.embedding)
            }, f, ensure_ascii=False, indent=2)

        return record.embedding_id

    def load_embedding(self, embedding_id: str) -> Optional[EmbeddingRecord]:
        """Laad embedding op basis van ID"""
        try:
            # Haal metadata op uit database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT text, model, timestamp, file_path 
                FROM embeddings WHERE id = ?
            ''', (embedding_id,))

            result = cursor.fetchone()
            conn.close()

            if not result:
                return None

            text, model, timestamp, file_path = result

            # Laad embedding array
            embedding = np.load(file_path)

            return EmbeddingRecord(
                text=text,
                embedding=embedding,
                model=model,
                timestamp=timestamp,
                embedding_id=embedding_id
            )

        except Exception as e:
            print(f"Fout bij laden embedding {embedding_id}: {e}")
            return None

    def list_embeddings(self, model_filter: str = None) -> List[Dict]:
        """Lijst alle opgeslagen embeddings, optioneel gefilterd op model"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if model_filter:
            cursor.execute('''
                SELECT id, text, model, timestamp, embedding_size
                FROM embeddings
                WHERE model = ?
                ORDER BY timestamp DESC
            ''', (model_filter,))
        else:
            cursor.execute('''
                SELECT id, text, model, timestamp, embedding_size
                FROM embeddings
                ORDER BY timestamp DESC
            ''')

        results = cursor.fetchall()
        conn.close()

        return [
            {
                'id': row[0],
                'text': row[1][:100] + ('...' if len(row[1]) > 100 else ''),
                'model': row[2],
                'timestamp': row[3],
                'embedding_size': row[4]
            }
            for row in results
        ]

class EmbeddingComparator:
    """Tool voor het vergelijken van embeddings"""
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Bereken cosine similarity tussen twee embeddings"""
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if norm_product == 0:
            return 0.0
        return float(dot_product / norm_product)

    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Bereken Euclidische afstand tussen twee embeddings"""
        return float(np.linalg.norm(emb1 - emb2))

    @staticmethod
    def manhattan_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Bereken Manhattan afstand tussen twee embeddings"""
        return float(np.sum(np.abs(emb1 - emb2)))

    def compare_embeddings(self, record1: EmbeddingRecord, record2: EmbeddingRecord) -> Dict:
        """Uitgebreide vergelijking tussen twee embeddings"""
        similarities = {
            'cosine_similarity': self.cosine_similarity(record1.embedding, record2.embedding),
            'euclidean_distance': self.euclidean_distance(record1.embedding, record2.embedding),
            'manhattan_distance': self.manhattan_distance(record1.embedding, record2.embedding),
        }

        return {
            'text1': record1.text,
            'text2': record2.text,
            'model1': record1.model,
            'model2': record2.model,
            'similarities': similarities,
            'cross_model_comparison': record1.model != record2.model,
            'timestamp_comparison': {
                'record1_time': record1.timestamp,
                'record2_time': record2.timestamp
            }
        }

    def batch_compare(self, records: List[EmbeddingRecord]) -> List[Dict]:
        """Vergelijk alle combinaties van embeddings in een batch"""
        comparisons = []
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                comparison = self.compare_embeddings(records[i], records[j])
                comparisons.append(comparison)
        return comparisons

def demo_multi_model_comparison():
    """Demonstreer vergelijking tussen verschillende modellen"""
    print("=== Multi-Model Embedding Vergelijking Demo ===\n")
    
    tokenizer = RemoteOllamaEmbeddingTokenizer()
    comparator = EmbeddingComparator()
    
    # Test tekst
    test_text = "Machine learning algoritmen zijn krachtige tools voor data analyse."
    
    # Verschillende modellen (pas aan naar beschikbare modellen)
    models_to_test = [
        "nomic-embed-text",
        "mxbai-embed-large",
        # Voeg hier andere embedding modellen toe die beschikbaar zijn
    ]
    
    records = []
    
    print("1. Genereren embeddings met verschillende modellen...")
    for model in models_to_test:
        try:
            print(f"   Genereren embedding met {model}...")
            record = tokenizer.embed_text(test_text, model=model)
            embedding_id = tokenizer.save_embedding(record)
            records.append(record)
            print(f"   ✓ Opgeslagen als: {embedding_id}")
        except Exception as e:
            print(f"   ✗ Fout met model {model}: {e}")
    
    print(f"\n2. Vergelijken van {len(records)} embeddings...")
    comparisons = comparator.batch_compare(records)
    
    for comparison in comparisons:
        print(f"\n   Vergelijking: {comparison['model1']} vs {comparison['model2']}")
        print(f"   Cosine similarity: {comparison['similarities']['cosine_similarity']:.4f}")
        print(f"   Euclidische afstand: {comparison['similarities']['euclidean_distance']:.4f}")
        print(f"   Cross-model: {'Ja' if comparison['cross_model_comparison'] else 'Nee'}")

if __name__ == '__main__':
    # Configuratie info
    print("=== Remote Ollama Embedding Tokenizer ===")
    print(f"Server URL: {OLLAMA_SERVER_URL}")
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"Storage directory: {STORAGE_DIR}\n")
    
    # Uncomment de volgende regel om de demo uit te voeren
    # demo_multi_model_comparison()
    
    # Basis gebruik voorbeeld
    tokenizer = RemoteOllamaEmbeddingTokenizer()
    comparator = EmbeddingComparator()
    
    # Voorbeeld teksten
    text1 = "Kunstmatige intelligentie transformeert de moderne wereld."
    text2 = "AI technologie verandert onze samenleving fundamenteel."
    
    print("Genereren van embeddings...")
    try:
        record1 = tokenizer.embed_text(text1)
        record2 = tokenizer.embed_text(text2)
        
        id1 = tokenizer.save_embedding(record1)
        id2 = tokenizer.save_embedding(record2)
        
        print(f"Embedding 1 opgeslagen: {id1}")
        print(f"Embedding 2 opgeslagen: {id2}")
        
        # Vergelijk embeddings
        comparison = comparator.compare_embeddings(record1, record2)
        print(f"\nSimilarity score: {comparison['similarities']['cosine_similarity']:.4f}")
        
        # Toon alle opgeslagen embeddings
        print("\nOpgeslagen embeddings:")
        for emb in tokenizer.list_embeddings():
            print(f"- {emb['id']}: {emb['text']} (model: {emb['model']})")
            
    except Exception as e:
        print(f"Fout: {e}")
        print("Zorg ervoor dat je Ollama server draait en het model beschikbaar is.")

