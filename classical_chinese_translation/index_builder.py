#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Chinese Translation Index Builder

This module builds and manages vector indexes for semantic search,
supporting multiple granularity levels for improved translation accuracy.
Inspired by the semantic search engine pattern from 03_semantic_search.py.
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
from sentence_transformers import util

from project_config import (
    load_model, 
    print_model_info,
    INDEX_DIR,
    SENTENCE_INDEX_FILE,
    PARAGRAPH_INDEX_FILE,
    FULL_TEXT_INDEX_FILE
)
from data_processor import DataProcessor, TextPair, SentencePair, ClausePair


class EmbeddingIndex:
    """
    Vector index for semantic search with cosine similarity.
    
    This class manages embeddings and provides efficient similarity search
    capabilities. It follows patterns from 02_similarity_calculation.py
    and 03_semantic_search.py for best practices.
    
    Attributes:
        embeddings: Numpy array of normalized embeddings
        texts: List of original texts
        metadata: List of metadata dictionaries
        model: Embedding model instance
        index_type: Type identifier for this index
    """
    
    def __init__(self):
        """Initialize an empty embedding index."""
        self.embeddings: Optional[np.ndarray] = None
        self.embeddings_normalized: Optional[np.ndarray] = None
        self.texts: List[str] = []
        self.metadata: List[dict] = []
        self.model = None
        self.index_type: str = ""
        
    def load_model(self):
        """
        Load the embedding model if not already loaded.
        
        Returns:
            The loaded model instance
        """
        if self.model is None:
            print("\n🔄 Loading Qwen3-Embedding model...")
            print_model_info()
            self.model = load_model(device='cuda')
            print("✅ Model loaded successfully!")
        return self.model
    
    def build_index(self, texts: List[str], metadata: List[dict], 
                   index_type: str = "sentence", batch_size: int = 32):
        """
        Build vector index from texts with pre-computed normalization.
        
        This method generates embeddings and normalizes them for efficient
        cosine similarity computation.
        
        Args:
            texts: List of texts to index
            metadata: List of metadata dictionaries (one per text)
            index_type: Index type identifier ("sentence", "clause", "full_text")
            batch_size: Batch size for encoding
            
        Returns:
            Self for method chaining
        """
        if self.model is None:
            self.load_model()
        
        self.texts = texts
        self.metadata = metadata
        self.index_type = index_type
        
        print(f"\n📊 Building {index_type} index with {len(texts)} entries...")
        
        # Generate embeddings in batches
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch_texts, 
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Pre-normalize for cosine similarity
            )
            all_embeddings.append(batch_embeddings)
        
        self.embeddings = np.vstack(all_embeddings)
        
        # Pre-compute normalized embeddings for fast search
        self._normalize_embeddings()
        
        print(f"✅ Index built successfully! Shape: {self.embeddings.shape}")
        return self
    
    def _normalize_embeddings(self):
        """
        Pre-normalize embeddings for efficient cosine similarity.
        
        Normalization allows using dot product instead of full cosine computation.
        """
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        self.embeddings_normalized = self.embeddings / norms
    
    def search(self, query: str, top_k: int = 5, 
              similarity_threshold: float = 0.0) -> List[Tuple[str, dict, float]]:
        """
        Perform semantic search using cosine similarity.
        
        This method follows the pattern from 03_semantic_search.py for
        efficient similarity computation.
        
        Args:
            query: Query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (text, metadata, similarity) tuples sorted by similarity
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        self.load_model()
        
        # Generate and normalize query embedding
        query_embedding = self.model.encode(
            query, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Compute cosine similarity using dot product (embeddings are normalized)
        if self.embeddings_normalized is None:
            self._normalize_embeddings()
        
        similarities = np.dot(self.embeddings_normalized, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results with threshold filtering
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= similarity_threshold:
                results.append((self.texts[idx], self.metadata[idx], sim))
        
        return results
    
    def search_batch(self, queries: List[str], top_k: int = 5,
                    similarity_threshold: float = 0.0) -> List[List[Tuple[str, dict, float]]]:
        """
        Batch search for multiple queries (more efficient).
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of result lists
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        self.load_model()
        
        # Batch encode queries
        query_embeddings = self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        if self.embeddings_normalized is None:
            self._normalize_embeddings()
        
        # Compute all similarities at once
        all_similarities = np.dot(self.embeddings_normalized, query_embeddings.T)
        
        all_results = []
        for i in range(len(queries)):
            similarities = all_similarities[:, i]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                sim = float(similarities[idx])
                if sim >= similarity_threshold:
                    results.append((self.texts[idx], self.metadata[idx], sim))
            
            all_results.append(results)
        
        return all_results
    
    def get_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Compute similarity matrix for a list of texts.
        
        Useful for analysis and visualization, following pattern
        from 02_similarity_calculation.py.
        
        Args:
            texts: List of texts to compare
            
        Returns:
            Square similarity matrix
        """
        self.load_model()
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Compute pairwise cosine similarity
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        return similarity_matrix
    
    def save(self, filepath: str):
        """
        Save index to file.
        
        Args:
            filepath: Path to save the index
        """
        data = {
            'embeddings': self.embeddings,
            'embeddings_normalized': self.embeddings_normalized,
            'texts': self.texts,
            'metadata': self.metadata,
            'index_type': self.index_type
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Index saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load index from file.
        
        Args:
            filepath: Path to load the index from
            
        Returns:
            Self for method chaining
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.embeddings_normalized = data.get('embeddings_normalized')
        self.texts = data['texts']
        self.metadata = data['metadata']
        self.index_type = data['index_type']
        
        # Ensure normalized embeddings exist
        if self.embeddings_normalized is None:
            self._normalize_embeddings()
        
        print(f"✅ Loaded {len(self.texts)} entries from {filepath}")
        return self
    
    def __len__(self) -> int:
        """Return number of indexed items."""
        return len(self.texts)


# Additional index file paths
CLAUSE_INDEX_FILE = os.path.join(INDEX_DIR, "clause_index.pkl")


class TranslationIndexBuilder:
    """
    Multi-granularity translation index builder.
    
    Builds three levels of indexes:
    - Clause: Fine-grained phrases
    - Sentence: Individual sentences
    - Full text: Complete poems/prose
    
    This follows the multi-granularity approach from 01_basic_usage.py
    for better matching at different text lengths.
    """
    
    def __init__(self):
        """Initialize the index builder."""
        self.data_processor = DataProcessor()
        self.clause_index = EmbeddingIndex()
        self.sentence_index = EmbeddingIndex()
        self.full_text_index = EmbeddingIndex()
        self._shared_model = None
        
    def _get_shared_model(self):
        """
        Get shared model instance to avoid loading multiple times.
        
        Returns:
            The shared model instance
        """
        if self._shared_model is None:
            print("\n🔄 Loading Qwen3-Embedding model...")
            print_model_info()
            self._shared_model = load_model(device='cuda')
            print("✅ Model loaded successfully!")
        return self._shared_model
    
    def build_all_indexes(self, force_rebuild: bool = False,
                         build_clause: bool = False):
        """
        Build all translation indexes.
        
        Args:
            force_rebuild: If True, rebuild even if indexes exist
            build_clause: If True, also build clause-level index
        """
        # Load and process data
        self.data_processor.load_all_data()
        self.data_processor.create_sentence_pairs()
        
        if build_clause:
            self.data_processor.create_clause_pairs()
        
        self.data_processor.print_statistics()
        
        # Build sentence index (primary)
        if force_rebuild or not os.path.exists(SENTENCE_INDEX_FILE):
            self._build_sentence_index()
        else:
            print(f"📂 Sentence index exists: {SENTENCE_INDEX_FILE}")
            self.sentence_index.load(SENTENCE_INDEX_FILE)
        
        # Build clause index (optional, for fine-grained matching)
        if build_clause:
            if force_rebuild or not os.path.exists(CLAUSE_INDEX_FILE):
                self._build_clause_index()
            else:
                print(f"📂 Clause index exists: {CLAUSE_INDEX_FILE}")
                self.clause_index.load(CLAUSE_INDEX_FILE)
        
        # Build full text index (optional, skip if GPU memory is limited)
        # Commented out by default to avoid memory issues
        # if force_rebuild or not os.path.exists(FULL_TEXT_INDEX_FILE):
        #     self._build_full_text_index()
        
        print("\n✅ All indexes built/loaded successfully!")
    
    def _build_sentence_index(self):
        """Build sentence-level index."""
        print("\n" + "=" * 60)
        print("Building Sentence-Level Index")
        print("=" * 60)
        
        texts = []
        metadata = []
        
        for sp in self.data_processor.sentence_pairs:
            texts.append(sp.classical)
            metadata.append({
                'title': sp.title,
                'classical': sp.classical,
                'modern': sp.modern,
                'sentence_idx': sp.sentence_idx,
                'full_classical': sp.full_classical,
                'full_modern': sp.full_modern,
                'type': 'sentence'
            })
        
        self.sentence_index.model = self._get_shared_model()
        self.sentence_index.build_index(texts, metadata, index_type="sentence")
        self.sentence_index.save(SENTENCE_INDEX_FILE)
    
    def _build_clause_index(self):
        """Build clause-level index for fine-grained matching."""
        print("\n" + "=" * 60)
        print("Building Clause-Level Index")
        print("=" * 60)
        
        texts = []
        metadata = []
        
        for cp in self.data_processor.clause_pairs:
            texts.append(cp.classical)
            metadata.append({
                'title': cp.title,
                'classical': cp.classical,
                'modern': cp.modern,
                'clause_idx': cp.clause_idx,
                'parent_classical': cp.parent_classical,
                'type': 'clause'
            })
        
        self.clause_index.model = self._get_shared_model()
        self.clause_index.build_index(texts, metadata, index_type="clause")
        self.clause_index.save(CLAUSE_INDEX_FILE)
    
    def _build_full_text_index(self):
        """Build full text index."""
        print("\n" + "=" * 60)
        print("Building Full Text Index")
        print("=" * 60)
        
        texts = []
        metadata = []
        
        for tp in self.data_processor.text_pairs:
            texts.append(tp.classical)
            metadata.append({
                'title': tp.title,
                'author': tp.author,
                'classical': tp.classical,
                'modern': tp.modern,
                'source_file': tp.source_file,
                'type': 'full_text'
            })
        
        self.full_text_index.model = self._get_shared_model()
        self.full_text_index.build_index(texts, metadata, index_type="full_text", batch_size=8)
        self.full_text_index.save(FULL_TEXT_INDEX_FILE)
    
    def get_sentence_index(self) -> EmbeddingIndex:
        """
        Get the sentence index, loading if necessary.
        
        Returns:
            Sentence-level EmbeddingIndex
        """
        if self.sentence_index.embeddings is None:
            if os.path.exists(SENTENCE_INDEX_FILE):
                self.sentence_index.load(SENTENCE_INDEX_FILE)
            else:
                raise ValueError("Sentence index not found. Run build_all_indexes() first.")
        return self.sentence_index
    
    def get_clause_index(self) -> EmbeddingIndex:
        """
        Get the clause index, loading if necessary.
        
        Returns:
            Clause-level EmbeddingIndex
        """
        if self.clause_index.embeddings is None:
            if os.path.exists(CLAUSE_INDEX_FILE):
                self.clause_index.load(CLAUSE_INDEX_FILE)
            else:
                raise ValueError("Clause index not found. Run build_all_indexes(build_clause=True).")
        return self.clause_index
    
    def get_full_text_index(self) -> EmbeddingIndex:
        """
        Get the full text index, loading if necessary.
        
        Returns:
            Full text EmbeddingIndex
        """
        if self.full_text_index.embeddings is None:
            if os.path.exists(FULL_TEXT_INDEX_FILE):
                self.full_text_index.load(FULL_TEXT_INDEX_FILE)
            else:
                raise ValueError("Full text index not found.")
        return self.full_text_index


def main():
    """Build indexes and test search functionality."""
    print("=" * 60)
    print("Classical Chinese Translation Index Builder")
    print("=" * 60)
    
    builder = TranslationIndexBuilder()
    builder.build_all_indexes(force_rebuild=True)
    
    # Test search functionality
    print("\n" + "=" * 60)
    print("Testing Search Functionality")
    print("=" * 60)
    
    test_queries = [
        "梳洗罢",
        "过尽千帆皆不是",
        "不患寡而患不均",
    ]
    
    sentence_index = builder.get_sentence_index()
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 40)
        
        results = sentence_index.search(query, top_k=3)
        for i, (text, meta, sim) in enumerate(results, 1):
            print(f"{i}. Similarity: {sim:.4f}")
            print(f"   Classical: {meta['classical']}")
            print(f"   Modern: {meta['modern'][:60]}...")


if __name__ == "__main__":
    main()
