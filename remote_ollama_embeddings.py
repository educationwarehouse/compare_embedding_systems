from dotenv import load_dotenv
import os
import requests
import json
import numpy as np
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
from scipy import stats
import warnings

# HuggingFace dependencies via LlamaIndex
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.embeddings.ollama import OllamaEmbedding
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    print("⚠ LlamaIndex dependencies niet beschikbaar. Installeer: pip install llama-index llama-index-embeddings-huggingface llama-index-embeddings-ollama")

# Load environment variables
load_dotenv()

OLLAMA_SERVER_URL = os.getenv('OLLAMA_SERVER_URL', 'http://localhost:11434')
DEFAULT_OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'nomic-embed-text')
STORAGE_DIR = os.getenv('EMBEDDING_STORAGE_DIR', 'embeddings_db')

@dataclass
class EmbeddingRecord:
    """Data class voor het opslaan van embedding metadata"""
    text: str
    embedding: np.ndarray
    model: str
    timestamp: str
    embedding_id: str
    embedding_type: str = "ollama"  # "ollama" of "huggingface"

    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'embedding': self.embedding.tolist(),
            'model': self.model,
            'timestamp': self.timestamp,
            'embedding_id': self.embedding_id,
            'embedding_type': self.embedding_type
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            text=data['text'],
            embedding=np.array(data['embedding']),
            model=data['model'],
            timestamp=data['timestamp'],
            embedding_id=data['embedding_id'],
            embedding_type=data.get('embedding_type', 'ollama')
        )

class RemoteOllamaEmbeddingTokenizer:
    """Ollama remote embedding service"""
    
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, storage_dir: str = STORAGE_DIR, server_url: str = OLLAMA_SERVER_URL):
        self.model = model
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.db_path = self.storage_dir / "embeddings.db"
        self.server_url = server_url.rstrip('/')
        self._init_database()
        self._validate_server()

    def _init_database(self):
        """Initialiseer SQLite database met auto-migratie"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings';")
        table_exists = cursor.fetchone()
        
        if table_exists:
            # Check if embedding_type column exists
            cursor.execute("PRAGMA table_info(embeddings)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'embedding_type' not in columns:
                print("🔧 Migrating database schema...")
                try:
                    cursor.execute('PRAGMA foreign_keys=off;')
                    cursor.execute('BEGIN TRANSACTION;')
                    
                    # Create new table
                    cursor.execute("""
                        CREATE TABLE new_embeddings (
                            id TEXT PRIMARY KEY,
                            text TEXT NOT NULL,
                            model TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            file_path TEXT NOT NULL,
                            embedding_size INTEGER NOT NULL,
                            embedding_type TEXT DEFAULT 'ollama'
                        )
                    """)
                    
                    # Copy existing data
                    cursor.execute("""
                        INSERT INTO new_embeddings (id, text, model, timestamp, file_path, embedding_size, embedding_type)
                        SELECT id, text, model, timestamp, file_path, embedding_size, 'ollama' FROM embeddings
                    """)
                    
                    # Replace table
                    cursor.execute("DROP TABLE embeddings;")
                    cursor.execute("ALTER TABLE new_embeddings RENAME TO embeddings;")
                    cursor.execute('COMMIT;')
                    cursor.execute('PRAGMA foreign_keys=on;')
                    
                    print("✅ Database migration successful!")
                    
                except Exception as e:
                    cursor.execute('ROLLBACK;')
                    print(f"❌ Migration failed: {e}")
                    raise
        else:
            # Create new table
            cursor.execute("""
                CREATE TABLE embeddings (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    model TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    embedding_size INTEGER NOT NULL,
                    embedding_type TEXT DEFAULT 'ollama'
                )
            """)
        
        conn.commit()
        conn.close()

    def _validate_server(self):
        """Test connectie met remote Ollama server"""
        try:
            response = requests.get(f"{self.server_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✓ Verbonden met Ollama server op {self.server_url}")
            else:
                print(f"⚠ Waarschuwing: Server antwoordt met status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠ Kan niet verbinden met Ollama server: {e}")

    def generate_embedding_id(self, text: str) -> str:
        """Genereer unieke ID voor embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_hash = hashlib.md5(self.model.encode()).hexdigest()[:6]
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
            response.raise_for_status()
            
            result = response.json()
            
            if 'embeddings' not in result or len(result['embeddings']) == 0:
                raise RuntimeError("Geen embeddings ontvangen van server")

            embedding = np.array(result['embeddings'][0], dtype=np.float32)

            record = EmbeddingRecord(
                text=text,
                embedding=embedding,
                model=embedding_model,
                timestamp=datetime.now().isoformat(),
                embedding_id=self.generate_embedding_id(text),
                embedding_type="ollama"
            )

            return record

        except requests.exceptions.Timeout:
            raise RuntimeError("Timeout bij het genereren van embedding")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Netwerk fout: {e}")

    def save_embedding(self, record: EmbeddingRecord) -> str:
        """Sla embedding op in eigen formaat (NPY + metadata)"""
        embedding_file = self.storage_dir / f"{record.embedding_id}.npy"
        np.save(embedding_file, record.embedding)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings 
            (id, text, model, timestamp, file_path, embedding_size, embedding_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record.embedding_id,
            record.text,
            record.model,
            record.timestamp,
            str(embedding_file),
            len(record.embedding),
            record.embedding_type
        ))
        conn.commit()
        conn.close()

        return record.embedding_id

    def load_embedding(self, embedding_id: str) -> Optional[EmbeddingRecord]:
        """Laad embedding op basis van ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT text, model, timestamp, file_path, embedding_type 
                FROM embeddings WHERE id = ?
            """, (embedding_id,))

            result = cursor.fetchone()
            conn.close()

            if not result:
                return None

            text, model, timestamp, file_path, embedding_type = result
            embedding = np.load(file_path)

            return EmbeddingRecord(
                text=text,
                embedding=embedding,
                model=model,
                timestamp=timestamp,
                embedding_id=embedding_id,
                embedding_type=embedding_type or "ollama"
            )

        except Exception as e:
            print(f"Fout bij laden embedding {embedding_id}: {e}")
            return None

class HuggingFaceLlamaIndexEmbedding:
    """HuggingFace embedding service using LlamaIndex"""
    
    def __init__(self, model_name: str = "jegormeister/bert-base-dutch-cased-snli", storage_dir: str = STORAGE_DIR):
        if not LLAMA_INDEX_AVAILABLE:
            raise RuntimeError("LlamaIndex dependencies niet geïnstalleerd. Installeer: pip install llama-index llama-index-embeddings-huggingface")
        
        self.model_name = model_name
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.db_path = self.storage_dir / "embeddings.db"
        self._init_database()
        
        try:
            self.embedder = HuggingFaceEmbedding(model_name=model_name)
            print(f"✓ HuggingFace model geladen via LlamaIndex: {model_name}")
        except Exception as e:
            print(f"⚠ Fout bij laden HuggingFace model {model_name}: {e}")
            raise

    def _init_database(self):
        """Initialiseer SQLite database (hergebruikt bestaande structuur)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                file_path TEXT NOT NULL,
                embedding_size INTEGER NOT NULL,
                embedding_type TEXT DEFAULT 'huggingface'
            )
        """)
        conn.commit()
        conn.close()

    def generate_embedding_id(self, text: str) -> str:
        """Genereer unieke ID voor embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_hash = hashlib.md5(self.model_name.encode()).hexdigest()[:6]
        return f"hf_{model_hash}_{timestamp}_{text_hash[:8]}"

    def embed_text(self, text: str) -> EmbeddingRecord:
        """Genereer embedding voor tekst via HuggingFace via LlamaIndex"""
        try:
            # LlamaIndex HuggingFaceEmbedding.embed() expects a list
            embedding_vector = self.embedder.get_text_embedding(text)
            embedding = np.array(embedding_vector, dtype=np.float32)

            record = EmbeddingRecord(
                text=text,
                embedding=embedding,
                model=self.model_name,
                timestamp=datetime.now().isoformat(),
                embedding_id=self.generate_embedding_id(text),
                embedding_type="huggingface"
            )

            return record

        except Exception as e:
            raise RuntimeError(f"HuggingFace embedding generatie gefaald: {e}")

    def save_embedding(self, record: EmbeddingRecord) -> str:
        """Sla HuggingFace embedding op"""
        embedding_file = self.storage_dir / f"{record.embedding_id}.npy"
        np.save(embedding_file, record.embedding)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings 
            (id, text, model, timestamp, file_path, embedding_size, embedding_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record.embedding_id,
            record.text,
            record.model,
            record.timestamp,
            str(embedding_file),
            len(record.embedding),
            record.embedding_type
        ))
        conn.commit()
        conn.close()

        return record.embedding_id

    def load_embedding(self, embedding_id: str) -> Optional[EmbeddingRecord]:
        """Laad embedding op basis van ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT text, model, timestamp, file_path, embedding_type 
                FROM embeddings WHERE id = ?
            """, (embedding_id,))

            result = cursor.fetchone()
            conn.close()

            if not result:
                return None

            text, model, timestamp, file_path, embedding_type = result
            
            # Check if file exists
            if not Path(file_path).exists():
                return None
                
            embedding = np.load(file_path)

            return EmbeddingRecord(
                text=text,
                embedding=embedding,
                model=model,
                timestamp=timestamp,
                embedding_id=embedding_id,
                embedding_type=embedding_type or "huggingface"
            )

        except Exception as e:
            print(f"Fout bij laden embedding {embedding_id}: {e}")
            return None

class UnifiedEmbeddingTokenizer:
    """Unifeert Ollama en HuggingFace embeddings via LlamaIndex"""
    
    def __init__(self, storage_dir: str = STORAGE_DIR):
        self.storage_dir = storage_dir
        self.ollama_tokenizer = None
        self.hf_tokenizers = {}  # Cache per model
    
    def get_ollama_tokenizer(self):
        """Krijg Ollama tokenizer (lazy loading)"""
        if self.ollama_tokenizer is None:
            self.ollama_tokenizer = RemoteOllamaEmbeddingTokenizer(storage_dir=self.storage_dir)
        return self.ollama_tokenizer
    
    def get_hf_tokenizer(self, model: str = "jegormeister/bert-base-dutch-cased-snli"):
        """Krijg HuggingFace tokenizer (lazy loading met caching)"""
        if model not in self.hf_tokenizers:
            self.hf_tokenizers[model] = HuggingFaceLlamaIndexEmbedding(model_name=model, storage_dir=self.storage_dir)
        return self.hf_tokenizers[model]
    
    def detect_model_type(self, model: str) -> str:
        """Auto-detecteer model type"""
        if "/" in model or any(marker in model.lower() for marker in ["bert", "robbert", "jegormeister"]):
            return "huggingface"
        else:
            return "ollama"
    
    def embed_text(self, text: str, model: str, model_type: str = "auto") -> EmbeddingRecord:
        """Embed tekst met automatische detectie van model type"""
        
        if model_type == "auto":
            model_type = self.detect_model_type(model)
        
        if model_type == "huggingface":
            tokenizer = self.get_hf_tokenizer(model)
            return tokenizer.embed_text(text)
        elif model_type == "ollama":
            tokenizer = self.get_ollama_tokenizer()
            return tokenizer.embed_text(text, model=model)
        else:
            raise ValueError(f"Onbekend model type: {model_type}")
    
    def save_embedding(self, record: EmbeddingRecord) -> str:
        """Sla embedding op (delegeert naar juiste tokenizer)"""
        if record.embedding_type == "huggingface":
            return self.get_hf_tokenizer(record.model).save_embedding(record)
        else:
            return self.get_ollama_tokenizer().save_embedding(record)
    
    def load_embedding(self, embedding_id: str) -> Optional[EmbeddingRecord]:
        """Laad een bestaande embedding op basis van ID"""
        # Probeer eerst Ollama tokenizer
        if self.ollama_tokenizer:
            record = self.ollama_tokenizer.load_embedding(embedding_id)
            if record:
                return record
        
        # Probeer dan HuggingFace tokenizers
        for hf_tokenizer in self.hf_tokenizers.values():
            record = hf_tokenizer.load_embedding(embedding_id)
            if record:
                return record
        
        return None

    def generate_embedding_id(self, text: str, model: str) -> str:
        """Genereer embedding ID voor gegeven tekst en model"""
        model_type = self.detect_model_type(model)
        
        if model_type == "huggingface":
            hf_tokenizer = self.get_hf_tokenizer(model)
            return hf_tokenizer.generate_embedding_id(text)
        else:
            ollama_tokenizer = self.get_ollama_tokenizer()
            # We need to temporarily set the model for ID generation
            original_model = ollama_tokenizer.model
            ollama_tokenizer.model = model
            embedding_id = ollama_tokenizer.generate_embedding_id(text)
            ollama_tokenizer.model = original_model
            return embedding_id

# De rest van de AdvancedEmbeddingComparator klasse blijft hetzelfde...
class AdvancedEmbeddingComparator:
    """Geavanceerde tool voor het vergelijken van embeddings met meerdere methoden"""
    
    def __init__(self, background_corpus_size: int = 100):
        self.background_vectors = None
        self.background_corpus_size = background_corpus_size
        self._cached_stats = {}
    
    def set_background_corpus(self, embeddings: List[np.ndarray]):
        """Stel achtergrond corpus in voor QB-normalization"""
        if len(embeddings) > self.background_corpus_size:
            indices = np.random.choice(len(embeddings), self.background_corpus_size, replace=False)
            self.background_vectors = [embeddings[i] for i in indices]
        else:
            self.background_vectors = embeddings
        print(f"✓ Background corpus ingesteld met {len(self.background_vectors)} vectoren")
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Klassieke cosine similarity"""
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if norm_product == 0:
            return 0.0
        return float(dot_product / norm_product)
    
    @staticmethod
    def diem_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """DIEM - Dimension Insensitive Euclidean Metric"""
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        dimension_factor = 1.0 / np.sqrt(len(emb1))
        normalized_dist = euclidean_dist * dimension_factor
        return 1.0 / (1.0 + normalized_dist)
    
    def qb_normalized_cosine(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """QB-Norm gecorrigeerde cosine similarity"""
        base_cosine = self.cosine_similarity(emb1, emb2)
        
        if self.background_vectors is None:
            warnings.warn("Geen background corpus ingesteld, gebruik gewone cosine similarity")
            return base_cosine
        
        cache_key = f"{id(emb1)}_{id(emb2)}"
        if cache_key not in self._cached_stats:
            bg_scores_1 = [self.cosine_similarity(emb1, bg) for bg in self.background_vectors]
            bg_scores_2 = [self.cosine_similarity(emb2, bg) for bg in self.background_vectors]
            
            combined_bg = bg_scores_1 + bg_scores_2
            mean_bg = np.mean(combined_bg)
            std_bg = np.std(combined_bg)
            
            self._cached_stats[cache_key] = (mean_bg, std_bg)
        
        mean_bg, std_bg = self._cached_stats[cache_key]
        
        if std_bg > 0:
            return (base_cosine - mean_bg) / std_bg
        return base_cosine
    
    @staticmethod
    def angular_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Angular distance - alternatief voor cosine similarity"""
        cosine_sim = AdvancedEmbeddingComparator.cosine_similarity(emb1, emb2)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        angular_dist = np.arccos(cosine_sim) / np.pi
        return 1.0 - angular_dist
    
    @staticmethod
    def euclidean_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Euclidean distance converted to similarity"""
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        return 1.0 / (1.0 + euclidean_dist)
    
    def comprehensive_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
        """Bereken alle similarity metrics"""
        return {
            'cosine': self.cosine_similarity(emb1, emb2),
            'diem': self.diem_similarity(emb1, emb2),
            'qb_cosine': self.qb_normalized_cosine(emb1, emb2),
            'angular': self.angular_distance(emb1, emb2),
            'euclidean': self.euclidean_similarity(emb1, emb2)
        }
    
    def hybrid_similarity(self, emb1: np.ndarray, emb2: np.ndarray, 
                         weights: Dict[str, float] = None) -> float:
        """Gewogen combinatie van verschillende similarity metrics"""
        if weights is None:
            weights = {
                'diem': 0.35,
                'qb_cosine': 0.25,
                'cosine': 0.20,
                'angular': 0.15,
                'euclidean': 0.05
            }
        
        similarities = self.comprehensive_similarity(emb1, emb2)
        
        weighted_sum = sum(similarities[method] * weight 
                          for method, weight in weights.items() 
                          if method in similarities)
        total_weight = sum(weights.values())
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def compare_embeddings(self, record1: EmbeddingRecord, record2: EmbeddingRecord, 
                          method: str = 'comprehensive') -> Dict:
        """Uitgebreide vergelijking tussen twee embeddings"""
        
        if method == 'comprehensive':
            similarities = self.comprehensive_similarity(record1.embedding, record2.embedding)
            primary_score = self.hybrid_similarity(record1.embedding, record2.embedding)
        elif method == 'hybrid':
            similarities = {'hybrid': self.hybrid_similarity(record1.embedding, record2.embedding)}
            primary_score = similarities['hybrid']
        elif method == 'cosine':
            primary_score = self.cosine_similarity(record1.embedding, record2.embedding)
            similarities = {method: primary_score}
        elif method == 'diem':
            primary_score = self.diem_similarity(record1.embedding, record2.embedding)
            similarities = {method: primary_score}
        elif method == 'qb_cosine':
            primary_score = self.qb_normalized_cosine(record1.embedding, record2.embedding)
            similarities = {method: primary_score}
        elif method == 'angular':
            primary_score = self.angular_distance(record1.embedding, record2.embedding)
            similarities = {method: primary_score}
        elif method == 'euclidean':
            primary_score = self.euclidean_similarity(record1.embedding, record2.embedding)
            similarities = {method: primary_score}
        else:
            primary_score = self.cosine_similarity(record1.embedding, record2.embedding)
            similarities = {'cosine': primary_score}
        
        return {
            'text1': record1.text,
            'text2': record2.text,
            'model1': record1.model,
            'model2': record2.model,
            'primary_score': primary_score,
            'similarities': similarities,
            'comparison_method': method,
            'cross_model_comparison': record1.model != record2.model,
            'timestamp_comparison': {
                'record1_time': record1.timestamp,
                'record2_time': record2.timestamp
            }
        }
    
    def rank_similarities(self, query_record: EmbeddingRecord, 
                         candidate_records: List[EmbeddingRecord],
                         method: str = 'hybrid', top_k: int = None) -> List[Tuple[EmbeddingRecord, float]]:
        """Rank candidates by similarity to query"""
        similarities = []
        
        for candidate in candidate_records:
            comparison = self.compare_embeddings(query_record, candidate, method)
            similarities.append((candidate, comparison['primary_score']))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            return similarities[:top_k]
        return similarities

# Backward compatibility
class EmbeddingComparator(AdvancedEmbeddingComparator):
    """Backward compatibility wrapper"""
    
    def compare_embeddings(self, record1: EmbeddingRecord, record2: EmbeddingRecord) -> Dict:
        """Oude interface - gebruikt nu comprehensive comparison"""
        return super().compare_embeddings(record1, record2, method='comprehensive')

if __name__ == "__main__":
    print("=== Remote Ollama + HuggingFace Embedding System (LlamaIndex) ===")
    print(f"Ollama server URL: {OLLAMA_SERVER_URL}")
    print(f"Default Ollama model: {DEFAULT_OLLAMA_MODEL}")
    print(f"Storage directory: {STORAGE_DIR}")
    print(f"LlamaIndex available: {LLAMA_INDEX_AVAILABLE}")
    
    # Test beide embeddings
    unified = UnifiedEmbeddingTokenizer()
    
    texts = [
        "Machine learning algoritmen zijn krachtige tools.",
        "Kunstmatige intelligentie verandert de wereld."
    ]
    
    models_to_test = [
        ("nomic-embed-text", "ollama"),
        ("jegormeister/bert-base-dutch-cased-snli", "huggingface")
    ]
    
    for text in texts:
        print(f"\nTekst: {text}")
        for model, model_type in models_to_test:
            try:
                record = unified.embed_text(text, model=model, model_type=model_type)
                embedding_id = unified.save_embedding(record)
                print(f"  ✓ {model_type}: {model} → {embedding_id} (dim: {len(record.embedding)})")
            except Exception as e:
                print(f"  ✗ {model_type}: {model} → Fout: {e}")

