#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Index Builder for Classical Chinese Translation

This module builds optimized vector indexes using the SmartDataProcessor
for improved translation accuracy.

Features:
1. Quality-weighted indexing - Higher quality pairs get priority
2. Multi-level indexes - Sentence and clause levels
3. Metadata-rich entries - Full context for better matching
4. Efficient batch processing with GPU acceleration
"""
import os
import json
import pickle
import hashlib
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

from project_config import (
    load_model,
    print_model_info,
    INDEX_DIR,
    DATA_DIR,
    get_model_name,
)
from smart_data_processor import SmartDataProcessor, AlignedPair


# Index file paths
SMART_INDEX_FILE = os.path.join(INDEX_DIR, "smart_sentence_index.pkl")
CLAUSE_INDEX_FILE = os.path.join(INDEX_DIR, "smart_clause_index.pkl")
INDEX_METADATA_FILE = os.path.join(INDEX_DIR, "smart_index_metadata.json")


def compute_text_hash(text: str) -> str:
    """Compute a stable hash for classical text content."""
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


class SmartEmbeddingIndex:
    """
    Smart embedding index with quality-aware search.
    
    Features:
    - Pre-normalized embeddings for fast cosine similarity
    - Quality score boosting for better results
    - Batch search for efficiency
    """
    
    def __init__(self):
        """Initialize empty index."""
        self.embeddings: Optional[np.ndarray] = None
        self.embeddings_norm: Optional[np.ndarray] = None
        self.texts: List[str] = []
        self.metadata: List[dict] = []
        self.text_hashes: List[str] = []
        self.model = None
        self.index_type: str = ""
    
    def load_model(self):
        """Load embedding model if not loaded."""
        if self.model is None:
            print("\n🔄 Loading Qwen3-Embedding model...")
            print_model_info()
            self.model = load_model(device='cuda')
            print("✅ Model loaded!")
        return self.model
    
    def build_from_pairs(self, pairs: List[AlignedPair], 
                         index_type: str = "sentence",
                         batch_size: int = 64,
                         reuse_cache: Optional[Dict[str, np.ndarray]] = None,
                         num_workers: int = 4,
                         chunk_size: int = 1024):
        """
        Build index from aligned pairs with optimized GPU pipeline.
        
        Args:
            pairs: List of AlignedPair objects
            index_type: Type identifier for this index
            batch_size: GPU micro-batch size for encoding
            reuse_cache: Optional cache of existing embeddings
            num_workers: CPU workers for tokenization / data loading
            chunk_size: Number of texts processed per encode cycle
        """
        import gc
        import time
        import sys
        
        print(f"\n{'='*60}")
        print(f"[STEP 1/4] Preparing data structures...")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        reuse_cache = reuse_cache or {}
        self.texts = []
        self.metadata = []
        self.text_hashes = []
        self.index_type = index_type
        
        total = len(pairs)
        if total == 0:
            raise ValueError("No pairs provided for index building.")
        
        print(f"  Total pairs to process: {total}")
        sys.stdout.flush()
        
        embeddings: List[Optional[np.ndarray]] = [None] * total
        pending_texts: List[str] = []
        pending_indices: List[int] = []
        
        # Fast cache lookup
        print(f"\n[STEP 2/4] Checking cache...")
        for idx, p in enumerate(pairs):
            classical = p.classical.strip()
            text_hash = compute_text_hash(classical)
            
            # Store metadata immediately
            self.texts.append(classical)
            self.metadata.append({
                'title': p.title,
                'author': p.author,
                'classical': classical,
                'modern': p.modern,
                'alignment_score': p.alignment_score,
                'position': p.position,
                'type': index_type,
                'text_hash': text_hash
            })
            self.text_hashes.append(text_hash)
            
            # Check cache
            cached = reuse_cache.get(text_hash)
            if cached is not None:
                embeddings[idx] = cached
            else:
                pending_indices.append(idx)
                pending_texts.append(classical)
        
        reuse_count = total - len(pending_texts)
        print(f"  ♻️ Cache hits: {reuse_count} embeddings reused")
        print(f"  📝 New texts to encode: {len(pending_texts)}")
        
        pending_count = len(pending_texts)
        if pending_count > 0:
            print(f"\n{'='*60}")
            print(f"[STEP 3/4] Loading embedding model (GPU Optimized)...")
            print(f"{'='*60}")
            sys.stdout.flush()
            
            # Load model on GPU
            self.load_model()
            device = self.model.device
            print(f"  ✅ Model loaded on: {device}")
            
            print(f"\n{'='*60}")
            print(f"[STEP 4/4] Generating embeddings (GPU Pipeline)")
            print(f"           Total: {pending_count} | Batch: {batch_size} | Workers: {num_workers}")
            print(f"{'='*60}")
            sys.stdout.flush()
            start_time = time.time()
            
            # Optimized encoding loop
            # Sort by length to minimize padding (huge speedup for transformers)
            text_lengths = [len(t) for t in pending_texts]
            sorted_indices = np.argsort(text_lengths)  # Shortest first to warm-up
            
            sorted_texts = [pending_texts[i] for i in sorted_indices]
            original_indices = [pending_indices[i] for i in sorted_indices]
            
            worker_count = num_workers
            if worker_count is None or worker_count < 0:
                cpu_count = os.cpu_count() or 4
                worker_count = max(1, cpu_count // 2)
            
            adaptive_batch = batch_size
            chunk_cap = max(chunk_size, adaptive_batch * 16)
            chunk_cap = min(chunk_cap, pending_count)
            
            print("  🚀 Starting optimized GPU encoding...")
            print(f"  ⚙️ Worker threads: {worker_count} | Chunk size: {chunk_cap}")
            
            progress = tqdm(total=pending_count, desc="Encoding", unit="text")
            torch.cuda.empty_cache()
            
            start = 0
            while start < pending_count:
                end = min(start + chunk_cap, pending_count)
                chunk_texts = sorted_texts[start:end]
                chunk_targets = original_indices[start:end]
                
                while True:
                    try:
                        chunk_embeddings = self.model.encode(
                            chunk_texts,
                            batch_size=adaptive_batch,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            device=device
                        )
                        break
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and adaptive_batch > 1:
                            adaptive_batch = max(1, adaptive_batch // 2)
                            chunk_cap = max(adaptive_batch * 32, adaptive_batch)
                            print(f"\n  ⚠️ GPU OOM detected. Reducing batch to {adaptive_batch}, chunk to {chunk_cap}")
                            torch.cuda.empty_cache()
                            continue
                        print(f"\n  ⚠️ GPU Error: {e}")
                        print("  🔄 Falling back to safe sequential mode...")
                        torch.cuda.empty_cache()
                        for text, target_idx in zip(chunk_texts, chunk_targets):
                            emb = self.model.encode(
                                text,
                                batch_size=1,
                                show_progress_bar=False,
                                convert_to_numpy=True,
                                normalize_embeddings=True,
                                device=device
                            )
                            embeddings[target_idx] = emb
                            progress.update(1)
                        chunk_embeddings = None
                        break
                
                if chunk_embeddings is not None:
                    for emb, target_idx in zip(chunk_embeddings, chunk_targets):
                        embeddings[target_idx] = emb
                    processed = end - start
                    progress.update(processed)
                
                start = end
            
            progress.close()

            total_time = time.time() - start_time
            print(f"\n  ⏱️ Total encoding time: {int(total_time//60)}m{int(total_time%60)}s")
            print(f"  📊 Average speed: {pending_count/total_time:.1f} texts/sec")
            
        else:
            print(f"\n[STEP 3/4] Skipped - All embeddings from cache")
            print(f"[STEP 4/4] Skipped - No new texts to encode")
        
        print(f"\n{'='*60}")
        print(f"[FINAL] Assembling index...")
        print(f"{'='*60}")
        
        # Filter None
        valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
        if not valid_indices:
            raise RuntimeError("No valid embeddings generated!")
            
        embeddings = [embeddings[i] for i in valid_indices]
        self.texts = [self.texts[i] for i in valid_indices]
        self.metadata = [self.metadata[i] for i in valid_indices]
        self.text_hashes = [self.text_hashes[i] for i in valid_indices]
        
        self.embeddings = np.vstack(embeddings)
        self._normalize()
        
        print(f"  ✅ Index built! Shape: {self.embeddings.shape}")
        sys.stdout.flush()
    
    def _normalize(self):
        """Pre-normalize embeddings for fast search."""
        if self.embeddings is None:
            return
        
        # Ensure numpy array and handle None/Object types
        if isinstance(self.embeddings, list):
            # Filter None values first
            valid_embeddings = [e for e in self.embeddings if e is not None]
            if not valid_embeddings:
                self.embeddings = np.zeros((0, 0))
            else:
                try:
                    self.embeddings = np.vstack(valid_embeddings)
                except ValueError:
                    # Handle dimension mismatch
                    self.embeddings = np.array(valid_embeddings)

        if not isinstance(self.embeddings, np.ndarray):
            self.embeddings = np.array(self.embeddings)
            
        # Handle empty embeddings
        if self.embeddings.size == 0:
            self.embeddings_norm = self.embeddings
            return

        # Check for None or NaN in object arrays
        if self.embeddings.dtype == object:
             print("⚠️ Warning: Embeddings array contains objects/None, attempting to fix...")
             try:
                 # Filter out rows that are None or not arrays
                 valid_rows = []
                 valid_indices = []
                 for i, row in enumerate(self.embeddings):
                     if row is not None and isinstance(row, (np.ndarray, list)):
                         valid_rows.append(row)
                         valid_indices.append(i)
                 
                 if valid_rows:
                     self.embeddings = np.vstack(valid_rows)
                     # We also need to update texts/metadata to match
                     if len(self.texts) > len(valid_rows):
                         print(f"ℹ️ Dropping {len(self.texts) - len(valid_rows)} invalid entries")
                         self.texts = [self.texts[i] for i in valid_indices]
                         self.metadata = [self.metadata[i] for i in valid_indices]
                         self.text_hashes = [self.text_hashes[i] for i in valid_indices]
                 else:
                     self.embeddings = np.zeros((0, 0))
             except Exception as e:
                 print(f"❌ Failed to fix object array: {e}")
                 return

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings_norm = self.embeddings / norms
    
    def search(self, query: str, top_k: int = 5,
               min_similarity: float = 0.0,
               quality_boost: bool = True) -> List[Tuple[str, dict, float]]:
        """
        Search with optional quality boosting.
        
        Args:
            query: Query text
            top_k: Number of results
            min_similarity: Minimum similarity threshold
            quality_boost: Whether to boost by alignment quality
            
        Returns:
            List of (text, metadata, score) tuples
        """
        if self.embeddings_norm is None:
            raise ValueError("Index not built!")
        
        self.load_model()
        
        # Encode query
        query_emb = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Compute similarities
        similarities = np.dot(self.embeddings_norm, query_emb)
        
        # Apply quality boost if enabled
        if quality_boost:
            quality_scores = np.array([m.get('alignment_score', 1.0) for m in self.metadata])
            # Gentle boost: multiply by (0.9 + 0.1 * quality)
            boosted_scores = similarities * (0.9 + 0.1 * quality_scores)
        else:
            boosted_scores = similarities
        
        # Get top-k
        top_indices = np.argsort(boosted_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(boosted_scores[idx])
            raw_sim = float(similarities[idx])
            if raw_sim >= min_similarity:
                # Store both boosted and raw scores
                meta = self.metadata[idx].copy()
                meta['raw_similarity'] = raw_sim
                results.append((self.texts[idx], meta, score))
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[str, dict, float]]]:
        """Batch search for multiple queries."""
        self.load_model()
        
        query_embs = self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        all_sims = np.dot(self.embeddings_norm, query_embs.T)
        
        results = []
        for i in range(len(queries)):
            sims = all_sims[:, i]
            top_indices = np.argsort(sims)[::-1][:top_k]
            
            query_results = []
            for idx in top_indices:
                score = float(sims[idx])
                query_results.append((self.texts[idx], self.metadata[idx], score))
            results.append(query_results)
        
        return results
    
    def save(self, filepath: str):
        """Save index to file."""
        data = {
            'embeddings': self.embeddings,
            'embeddings_norm': self.embeddings_norm,
            'texts': self.texts,
            'metadata': self.metadata,
            'index_type': self.index_type,
            'text_hashes': self.text_hashes
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Index saved to {filepath}")
    
    def load(self, filepath: str):
        """Load index from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.embeddings_norm = data.get('embeddings_norm')
        self.texts = data['texts']
        self.metadata = data['metadata']
        self.index_type = data['index_type']
        self.text_hashes = data.get('text_hashes') or [
            compute_text_hash(text) for text in self.texts
        ]
        
        if self.embeddings_norm is None:
            self._normalize()
        
        print(f"✅ Loaded {len(self.texts)} entries from {filepath}")
        return self


class SmartIndexBuilder:
    """
    Smart index builder using SmartDataProcessor.
    """
    
    def __init__(self):
        """Initialize the builder."""
        self.processor = SmartDataProcessor()
        self.sentence_index = SmartEmbeddingIndex()
        self._model = None
    
    def _compute_data_signature(self) -> Tuple[str, Dict[str, int]]:
        """Compute a lightweight fingerprint of the data directory."""
        poem_dirs = []
        total_files = 0
        total_bytes = 0
        latest_mtime = 0

        if not os.path.exists(DATA_DIR):
            return "", {}

        for poem_dir in sorted(os.listdir(DATA_DIR)):
            poem_path = os.path.join(DATA_DIR, poem_dir)
            if not os.path.isdir(poem_path):
                continue
            poem_dirs.append(poem_dir)

            for fname in sorted(os.listdir(poem_path)):
                fpath = os.path.join(poem_path, fname)
                if not os.path.isfile(fpath):
                    continue
                total_files += 1
                total_bytes += os.path.getsize(fpath)
                latest_mtime = max(latest_mtime, int(os.path.getmtime(fpath)))

        payload = {
            "poem_count": len(poem_dirs),
            "file_count": total_files,
            "total_bytes": total_bytes,
            "latest_mtime": latest_mtime,
        }
        signature = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return signature, payload

    def _load_metadata(self) -> Dict[str, Any]:
        """Load existing metadata if present."""
        if not os.path.exists(INDEX_METADATA_FILE):
            return {}
        try:
            with open(INDEX_METADATA_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _save_metadata(self, metadata: Dict[str, Any]):
        """Persist metadata to disk."""
        os.makedirs(INDEX_DIR, exist_ok=True)
        with open(INDEX_METADATA_FILE, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)

    def _needs_rebuild(
        self,
        force_rebuild: bool,
        metadata: Dict[str, Any],
        data_signature: str,
        min_quality: float,
    ) -> bool:
        """Decide whether the index must be rebuilt."""
        if force_rebuild:
            print("🔁 Force rebuild requested.")
            return True
        if not os.path.exists(SMART_INDEX_FILE):
            print("ℹ️ No existing smart index found. Building from scratch.")
            return True
        if not metadata:
            print("ℹ️ Metadata missing. Rebuilding smart index.")
            return True
        if metadata.get("data_signature") != data_signature:
            print("🔄 New data detected. Rebuilding smart index.")
            return True
        if metadata.get("min_quality") != min_quality:
            print("🔄 Quality threshold changed. Rebuilding smart index.")
            return True
        if metadata.get("model") != get_model_name():
            print("🔄 Model configuration changed. Rebuilding smart index.")
            return True
        return False

    def _load_embedding_cache(self) -> Dict[str, np.ndarray]:
        """Load embeddings from existing index for reuse."""
        if not os.path.exists(SMART_INDEX_FILE):
            return {}
        try:
            print("♻️ Loading existing embeddings for reuse...")
            existing = SmartEmbeddingIndex()
            existing.load(SMART_INDEX_FILE)
            cache = {
                text_hash: emb
                for text_hash, emb in zip(existing.text_hashes, existing.embeddings)
            }
            return cache
        except Exception as exc:
            print(f"⚠️ Failed to load embedding cache: {exc}")
            return {}

    def build_indexes(self, force_rebuild: bool = False,
                      min_quality: float = 0.5,
                      batch_size: int = 64,
                      num_workers: int = 4,
                      chunk_size: int = 1024,
                      limit: int = 0):
        """
        Build all indexes.
        
        Args:
            force_rebuild: Force rebuild even if exists
            min_quality: Minimum alignment quality for inclusion
            num_workers: CPU tokenization workers for encode pipeline
            chunk_size: Number of samples per encode pass (higher = better throughput)
            limit: Max number of pairs to process (0 = no limit)
        """
        data_signature, data_stats = self._compute_data_signature()
        metadata = self._load_metadata()
        rebuild_required = self._needs_rebuild(
            force_rebuild, metadata, data_signature, min_quality
        )

        if rebuild_required:
            # Load and process data
            self.processor.load_all_data()
            self.processor.print_statistics()

            print("\n" + "=" * 60)
            print("Building Smart Sentence Index")
            print("=" * 60)

            pairs = self.processor.all_pairs
            if min_quality > 0:
                original_count = len(pairs)
                pairs = [p for p in pairs if p.alignment_score >= min_quality]
                print(
                    f"📊 Quality filter: {len(pairs)}/{original_count} pairs (>= {min_quality})"
                )
            
            if limit > 0:
                print(f"⚠️ Limiting to {limit} pairs for testing")
                pairs = pairs[:limit]

            if not pairs:
                raise ValueError("No aligned pairs available for index building.")
            
            reuse_cache = self._load_embedding_cache()
            if reuse_cache:
                print(f"⚙️ Reuse cache size: {len(reuse_cache)}")
            
            self.sentence_index.build_from_pairs(
                pairs,
                index_type="sentence",
                batch_size=batch_size,
                reuse_cache=reuse_cache,
                num_workers=num_workers,
                chunk_size=chunk_size
            )
            self.sentence_index.save(SMART_INDEX_FILE)

            metadata_payload = {
                "data_signature": data_signature,
                "data_stats": data_stats,
                "pair_count": len(pairs),
                "min_quality": min_quality,
                "model": get_model_name(),
                "built_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_metadata(metadata_payload)
            print("\n✅ Smart index rebuilt successfully!")
            
            # Use the in-memory index directly, do not reload from disk
            return
        else:
            print("📂 Smart index is up-to-date. Loading from disk.")
            self.sentence_index.load(SMART_INDEX_FILE)
            print(
                f"ℹ️ Cached pairs: {metadata.get('pair_count', len(self.sentence_index.texts))}"
            )
        
        print("\n✅ Index building complete!")
    
    def get_index(self) -> SmartEmbeddingIndex:
        """Get the sentence index."""
        if self.sentence_index.embeddings is None:
            if os.path.exists(SMART_INDEX_FILE):
                self.sentence_index.load(SMART_INDEX_FILE)
            else:
                raise ValueError("Index not found. Run build_indexes() first.")
        return self.sentence_index


def main():
    """Build indexes and test."""
    parser = argparse.ArgumentParser(description="Smart index builder.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if cache is fresh.")
    parser.add_argument("--min-quality", type=float, default=0.5,
                        help="Minimum alignment score for inclusion.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for embedding generation.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="CPU workers for tokenization/DataLoader (>=0).")
    parser.add_argument("--chunk-size", type=int, default=1024,
                        help="Number of texts processed per encode chunk.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of pairs (for testing).")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Smart Index Builder for Classical Chinese Translation")
    print("=" * 60)
    
    builder = SmartIndexBuilder()
    builder.build_indexes(
        force_rebuild=args.force,
        min_quality=args.min_quality,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        limit=args.limit
    )
    
    # Test search
    print("\n" + "=" * 60)
    print("Testing Smart Search")
    print("=" * 60)
    
    index = builder.get_index()
    
    test_queries = [
        "梳洗罢",
        "过尽千帆皆不是",
        "独倚望江楼",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 40)
        
        results = index.search(query, top_k=3, quality_boost=True)
        for i, (text, meta, score) in enumerate(results, 1):
            raw_sim = meta.get('raw_similarity', score)
            print(f"{i}. Score: {score:.4f} (raw: {raw_sim:.4f})")
            print(f"   Classical: {meta['classical']}")
            print(f"   Modern: {meta['modern'][:60]}...")


if __name__ == "__main__":
    main()

