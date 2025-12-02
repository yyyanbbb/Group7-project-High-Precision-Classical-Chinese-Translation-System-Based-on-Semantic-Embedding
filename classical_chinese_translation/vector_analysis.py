#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Space Analysis: Classical to Modern Transition

Adds cosine similarity diagnostics and clustering visualization to study
non-literal yet semantically aligned translations.
"""
import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from smart_index_builder import SmartEmbeddingIndex, SMART_INDEX_FILE
from project_config import INDEX_DIR

_INDEX_CACHE: Optional[SmartEmbeddingIndex] = None
CACHE_PREFIX = "vector_analysis_cache"


def _literal_overlap_ratio(text_a: str, text_b: str) -> float:
    """Character-level Jaccard overlap used to approximate literal similarity."""
    set_a = set(text_a)
    set_b = set(text_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _ensure_visualization_dir() -> str:
    """Return visualization directory path, creating it if necessary."""
    viz_dir = os.path.join(INDEX_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir

class VectorSpaceAnalyzer:
    def __init__(
        self,
        max_samples: int = 2000,
        batch_size: int = 16,
        reuse_embeddings: bool = True,
        index: Optional[SmartEmbeddingIndex] = None,
    ):
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.reuse_embeddings = reuse_embeddings
        self.index = self._resolve_index(index)
        self.model = self.index.load_model()

        # Extract paired vectors
        self.classical_vecs: List[np.ndarray] = []
        self.modern_vecs: np.ndarray = np.zeros((0, 0))
        self.pairs: List[Dict[str, str]] = []
        self.literal_overlap: List[float] = []
        self.length_ratios: List[float] = []
        self.alignment_scores: List[float] = []
        self.cosine_scores: Optional[np.ndarray] = None
        self.transition_vecs: Optional[np.ndarray] = None

        self._cache_dir = _ensure_visualization_dir()
        self._modern_cache = os.path.join(
            self._cache_dir, f"{CACHE_PREFIX}_modern_{max_samples}.npy"
        )

        self._extract_valid_pairs()

    def _resolve_index(self, provided: Optional[SmartEmbeddingIndex]) -> SmartEmbeddingIndex:
        """Reuse loaded indexes whenever possible to avoid redundant IO."""
        global _INDEX_CACHE
        if provided and provided.embeddings is not None:
            return provided
        if _INDEX_CACHE and _INDEX_CACHE.embeddings is not None:
            return _INDEX_CACHE
        index = provided or SmartEmbeddingIndex()
        if os.path.exists(SMART_INDEX_FILE):
            index.load(SMART_INDEX_FILE)
        else:
            raise FileNotFoundError(f"Index not found: {SMART_INDEX_FILE}")
        _INDEX_CACHE = index
        return index

    def _extract_valid_pairs(self):
        """Sample aligned pairs and pre-compute literal heuristics."""
        print("🔍 Extracting valid pairs for analysis...")
        count = 0
        for i, meta in enumerate(self.index.metadata):
            modern = meta.get("modern", "").strip()
            classical = self.index.texts[i]
            if modern and len(modern) > 5:
                self.classical_vecs.append(self.index.embeddings[i])
                literal_overlap = _literal_overlap_ratio(classical, modern)
                length_ratio = len(modern) / max(1, len(classical))
                self.literal_overlap.append(literal_overlap)
                self.length_ratios.append(length_ratio)
                self.alignment_scores.append(meta.get("alignment_score", 0.0))
                self.pairs.append(
                    {
                        "classical": classical,
                        "modern": modern,
                        "title": meta.get("title", "Unknown"),
                    }
                )
                count += 1
                if count >= self.max_samples:
                    break

        self.classical_vecs = self._normalize_vectors(np.array(self.classical_vecs))
        print(f"✅ Found {len(self.pairs)} valid pairs (Sampled from total)")

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def compute_modern_vectors(self):
        """Compute embeddings for modern translations (with caching)."""
        if (
            self.reuse_embeddings
            and os.path.exists(self._modern_cache)
            and os.path.getsize(self._modern_cache) > 0
        ):
            try:
                cached = np.load(self._modern_cache)
                if cached.shape[0] == len(self.pairs):
                    self.modern_vecs = cached
                    self._compute_pairwise_cosine()
                    print(f"♻️ Loaded cached modern embeddings: {self._modern_cache}")
                    return
            except Exception as exc:
                print(f"⚠️ Failed to reuse embeddings cache: {exc}")

        print("🔄 Computing modern text embeddings...")
        modern_texts = [p['modern'] for p in self.pairs]
        
        # Compute in smaller chunks to avoid OOM
        all_vecs = []
        chunk_size = 500
        for i in range(0, len(modern_texts), chunk_size):
            chunk = modern_texts[i : i + chunk_size]
            print(f"  Processing chunk {i//chunk_size + 1}/{(len(modern_texts)-1)//chunk_size + 1}")
            vecs = self.model.encode(
                chunk,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            all_vecs.append(vecs)
            
        if all_vecs:
            self.modern_vecs = self._normalize_vectors(np.vstack(all_vecs))
            self._compute_pairwise_cosine()
            if self.reuse_embeddings:
                try:
                    np.save(self._modern_cache, self.modern_vecs)
                    print(f"💾 Saved modern embedding cache to {self._modern_cache}")
                except Exception as exc:
                    print(f"⚠️ Failed to save embedding cache: {exc}")
            print("✅ Modern vectors computed")
        else:
            self.modern_vecs = np.zeros((0, 0))
            self.cosine_scores = None

    def _compute_pairwise_cosine(self):
        """Compute cosine similarity between each classical-modern pair."""
        if self.modern_vecs.shape[0] == self.classical_vecs.shape[0]:
            self.cosine_scores = np.sum(self.classical_vecs * self.modern_vecs, axis=1)
        else:
            self.cosine_scores = None

    def _ensure_modern_vectors(self):
        """Lazy-load modern embeddings to avoid redundant encoding passes."""
        if self.modern_vecs.size == 0 or self.modern_vecs.shape[0] == 0:
            self.compute_modern_vectors()

    def analyze_transition_vectors(self):
        """
        Analyze the vector difference (Modern - Classical).
        Returns the mean transition vector and consistency metrics.
        """
        self._ensure_modern_vectors()
            
        # Calculate transition vectors: Classical -> Modern
        # Note: We use Modern - Classical to represent the "Modernization" direction
        self.transition_vecs = self.modern_vecs - self.classical_vecs
        
        # 1. Mean Transition Vector (The "Modernization Direction")
        mean_transition = np.mean(self.transition_vecs, axis=0)
        
        # 2. Consistency Check (Cosine similarity between individual transitions and mean)
        similarities = cosine_similarity(self.transition_vecs, mean_transition.reshape(1, -1))
        avg_consistency = np.mean(similarities)
        
        print("\n📊 Transition Vector Analysis:")
        print(f"  • Direction Consistency: {avg_consistency:.4f} (-1 to 1)")
        print(f"  • Mean Vector Magnitude: {np.linalg.norm(mean_transition):.4f}")
        
        return mean_transition, avg_consistency

    def visualize_translation_paths(self, n_samples: int = 50):
        """
        Visualize Classical -> Modern paths in 2D PCA space.
        Shows how meaning shifts during translation.
        """
        self._ensure_modern_vectors()
            
        # Select random samples
        indices = np.random.choice(len(self.pairs), min(n_samples, len(self.pairs)), replace=False)
        
        sample_c = self.classical_vecs[indices]
        sample_m = self.modern_vecs[indices]
        
        # Combine for PCA to ensure same space
        combined = np.vstack([sample_c, sample_m])
        
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(combined)
        
        c_2d = reduced[:len(indices)]
        m_2d = reduced[len(indices):]
        
        # Plot
        fig = go.Figure()
        
        # Draw arrows (Classical -> Modern)
        for i in range(len(indices)):
            pair = self.pairs[indices[i]]
            
            # Arrow line
            fig.add_trace(go.Scatter(
                x=[c_2d[i, 0], m_2d[i, 0]],
                y=[c_2d[i, 1], m_2d[i, 1]],
                mode='lines',
                line=dict(color='gray', width=1),
                opacity=0.5,
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Classical Point
            fig.add_trace(go.Scatter(
                x=[c_2d[i, 0]],
                y=[c_2d[i, 1]],
                mode='markers',
                marker=dict(color='blue', size=8),
                name='Classical' if i == 0 else None,
                text=f"Classical: {pair['classical'][:30]}...",
                showlegend=(i == 0)
            ))
            
            # Modern Point
            fig.add_trace(go.Scatter(
                x=[m_2d[i, 0]],
                y=[m_2d[i, 1]],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Modern' if i == 0 else None,
                text=f"Modern: {pair['modern'][:30]}...",
                showlegend=(i == 0)
            ))

        fig.update_layout(
            title="Translation Trajectories in Semantic Space (PCA)",
            xaxis_title="PC1",
            yaxis_title="PC2",
            template="plotly_white",
            width=1000,
            height=800
        )
        
        return fig

    def cosine_similarity_report(
        self,
        high_semantic_threshold: float = 0.82,
        low_literal_threshold: float = 0.15,
        output_filename: str = "cosine_similarity_report.json",
        top_n: int = 20,
    ) -> Dict[str, List[Dict[str, float]]]:
        """Surface pairs with high semantic agreement but low literal overlap."""
        self._ensure_modern_vectors()
        if self.cosine_scores is None:
            self._compute_pairwise_cosine()

        if self.cosine_scores is None:
            raise RuntimeError("Cosine similarities unavailable; embeddings mismatch.")

        records = []
        for idx, pair in enumerate(self.pairs):
            record = {
                "classical": pair["classical"],
                "modern": pair["modern"],
                "title": pair.get("title", "Unknown"),
                "cosine_similarity": float(self.cosine_scores[idx]),
                "literal_overlap": float(self.literal_overlap[idx]),
                "alignment_score": float(self.alignment_scores[idx]),
                "length_ratio": float(self.length_ratios[idx]),
            }
            records.append(record)

        interpretive = [
            r for r in records
            if r["cosine_similarity"] >= high_semantic_threshold
            and r["literal_overlap"] <= low_literal_threshold
        ]
        interpretive.sort(key=lambda x: x["cosine_similarity"], reverse=True)

        literal_pairs = [r for r in records if r["literal_overlap"] >= 0.4]
        literal_pairs.sort(key=lambda x: x["cosine_similarity"], reverse=True)

        low_similarity = sorted(records, key=lambda x: x["cosine_similarity"])[:top_n]

        payload = {
            "thresholds": {
                "high_semantic": high_semantic_threshold,
                "low_literal": low_literal_threshold,
            },
            "interpretive_highlights": interpretive[:top_n],
            "literal_reference": literal_pairs[:top_n],
            "challenging_pairs": low_similarity,
        }

        viz_dir = _ensure_visualization_dir()
        output_path = os.path.join(viz_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        print(f"💾 Cosine similarity report saved to {output_path}")
        print(f"   🌟 Interpretive pairs captured: {len(interpretive[:top_n])}")
        return payload

    def visualize_cluster_map(
        self,
        n_clusters: int = 4,
        output_filename: str = "translation_cluster_map.html",
    ) -> Dict[str, Dict[str, float]]:
        """Cluster translation strategies and visualize them in 2D."""
        self._ensure_modern_vectors()
        if self.cosine_scores is None:
            self._compute_pairwise_cosine()

        features = np.column_stack(
            [
                np.array(self.literal_overlap),
                np.array(self.alignment_scores),
                np.array(self.cosine_scores),
                np.array(self.length_ratios),
            ]
        )
        features = np.nan_to_num(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(features)

        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(features)

        data_records = []
        for idx, pair in enumerate(self.pairs):
            data_records.append(
                {
                    "pc1": coords[idx, 0],
                    "pc2": coords[idx, 1],
                    "cluster": f"Cluster {cluster_ids[idx]}",
                    "literal_overlap": self.literal_overlap[idx],
                    "cosine_similarity": float(self.cosine_scores[idx]),
                    "alignment_score": self.alignment_scores[idx],
                    "length_ratio": self.length_ratios[idx],
                    "classical": pair["classical"],
                    "modern": pair["modern"],
                }
            )

        fig = px.scatter(
            data_records,
            x="pc1",
            y="pc2",
            color="cluster",
            size="alignment_score",
            hover_data={
                "classical": True,
                "modern": True,
                "literal_overlap": True,
                "cosine_similarity": True,
                "length_ratio": False,
            },
            title="Translation Strategy Clusters (Literal vs Interpretive Landscape)",
        )
        fig.update_layout(template="plotly_white", width=1000, height=800)

        viz_dir = _ensure_visualization_dir()
        output_path = os.path.join(viz_dir, output_filename)
        fig.write_html(output_path)

        cluster_summary: Dict[str, Dict[str, float]] = {}
        for cid in range(n_clusters):
            mask = cluster_ids == cid
            if not np.any(mask):
                continue
            cluster_summary[f"Cluster {cid}"] = {
                "count": int(np.sum(mask)),
                "mean_literal": float(np.mean(features[mask, 0])),
                "mean_alignment": float(np.mean(features[mask, 1])),
                "mean_cosine": float(np.mean(features[mask, 2])),
                "mean_length_ratio": float(np.mean(features[mask, 3])),
            }

        summary_path = os.path.join(viz_dir, output_filename.replace(".html", "_summary.json"))
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(cluster_summary, fh, ensure_ascii=False, indent=2)

        print(f"💾 Cluster map saved to {output_path}")
        print(f"   📊 Cluster summary saved to {summary_path}")
        return cluster_summary

    def describe_modernization_direction(
        self,
        top_n: int = 10,
        output_filename: str = "modernization_direction.json",
        mean_transition: Optional[np.ndarray] = None,
        avg_consistency: Optional[float] = None,
    ) -> Dict[str, object]:
        """Summarize the meaning of the Modern - Classical vector direction."""
        if mean_transition is None or avg_consistency is None:
            mean_transition, avg_consistency = self.analyze_transition_vectors()
        self._ensure_modern_vectors()
        if self.cosine_scores is None:
            self._compute_pairwise_cosine()
        if self.transition_vecs is None:
            self.transition_vecs = self.modern_vecs - self.classical_vecs
        sims = cosine_similarity(self.transition_vecs, mean_transition.reshape(1, -1)).flatten()
        top_indices = sims.argsort()[::-1][: min(top_n, len(sims))]

        anchor_examples = []
        for idx in top_indices:
            pair = self.pairs[idx]
            anchor_examples.append(
                {
                    "classical": pair["classical"],
                    "modern": pair["modern"],
                    "transition_alignment": float(sims[idx]),
                    "cosine_similarity": float(self.cosine_scores[idx]),
                    "literal_overlap": float(self.literal_overlap[idx]),
                    "alignment_score": float(self.alignment_scores[idx]),
                    "length_ratio": float(self.length_ratios[idx]),
                }
            )

        summary = {
            "direction_consistency": float(avg_consistency),
            "mean_vector_norm": float(np.linalg.norm(mean_transition)),
            "avg_length_ratio": float(np.mean(self.length_ratios)),
            "anchor_examples": anchor_examples,
        }

        viz_dir = _ensure_visualization_dir()
        output_path = os.path.join(viz_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

        print(f"💾 Modernization direction summary saved to {output_path}")
        return summary

    def vector_arithmetic_search(self, modern_query: str, top_k: int = 5):
        """
        Experimental: Search using Vector Arithmetic.
        Target = Vector(Modern_Query) - Mean_Transition_Vector
        
        Hypothesis: This might find "semantically equivalent but structurally different" classical texts.
        """
        if self.transition_vecs is None:
            self.analyze_transition_vectors()
            
        mean_transition = np.mean(self.transition_vecs, axis=0)
        
        # Encode query
        query_vec = self.model.encode(modern_query, convert_to_numpy=True, normalize_embeddings=True)
        
        # Apply arithmetic: Predicted_Classical = Modern - Transition
        # (Reversing the modernization process)
        target_vec = query_vec - mean_transition
        
        # Search in classical index
        # Normalize target for cosine similarity
        target_vec = target_vec / np.linalg.norm(target_vec)
        
        # Dot product search
        sims = np.dot(self.index.embeddings_norm, target_vec)
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'classical': self.index.texts[idx],
                'similarity': float(sims[idx]),
                'modern_ref': self.index.metadata[idx].get('modern', '')
            })
            
        return results

def parse_args():
    parser = argparse.ArgumentParser(
        description="Vector space diagnostics for the Classical→Modern embedding pipeline."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Number of aligned pairs to sample for analysis.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size when encoding modern translations.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recomputation of modern embeddings (ignore cached vectors).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=4,
        help="Number of clusters for translation strategy visualization.",
    )
    parser.add_argument(
        "--interpretive-threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold for interpretive pair detection.",
    )
    parser.add_argument(
        "--literal-threshold",
        type=float,
        default=0.15,
        help="Literal overlap threshold for interpretive pair detection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyzer = VectorSpaceAnalyzer(
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        reuse_embeddings=not args.no_cache,
    )
    
    # 1. Analyze Vector Shift
    mean_transition, avg_consistency = analyzer.analyze_transition_vectors()
    analyzer.describe_modernization_direction(
        mean_transition=mean_transition,
        avg_consistency=avg_consistency,
    )
    
    # 1.5 Cosine similarity diagnostics + clustering
    analyzer.cosine_similarity_report(
        high_semantic_threshold=args.interpretive_threshold,
        low_literal_threshold=args.literal_threshold,
    )
    analyzer.visualize_cluster_map(n_clusters=args.clusters)
    
    # 2. Visualize
    fig = analyzer.visualize_translation_paths()
    vector_paths_path = os.path.join(_ensure_visualization_dir(), "vector_paths.html")
    fig.write_html(vector_paths_path)
    print(f"\n💾 Translation trajectory plot saved to {vector_paths_path}")
    
    # 3. Test Vector Arithmetic Translation
    test_queries = [
        "我不想学习",
        "我很帅",
        "时光飞逝",
    ]
    
    print("\n🧪 Vector Arithmetic Search Test:")
    print("Target = Query_Vector - (Modern - Classical)_Mean")
    
    for q in test_queries:
        print(f"\n🔎 Query: {q}")
        results = analyzer.vector_arithmetic_search(q)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['classical']} (sim: {r['similarity']:.3f})")
            print(f"     Ref: {r['modern_ref'][:50]}...")

if __name__ == "__main__":
    main()

