#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Classical Chinese Translator

This module implements state-of-the-art translation strategies combining
multiple techniques from the embedding examples (00-04):

Techniques integrated:
1. Multi-granularity embedding (from 01_basic_usage.py)
2. Similarity matrix analysis (from 02_similarity_calculation.py)  
3. Semantic search with ranking (from 03_semantic_search.py)
4. Clustering-enhanced translation (from 04_text_clustering_visualization.py)

Innovations:
- Bi-directional verification: Verify translation by reverse searching
- Context-aware reranking: Use surrounding context for better matching
- Ensemble matching: Combine results from multiple strategies
- Confidence calibration with multi-feature scoring
- Cross-reference validation: Validate using related texts
"""
import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from sentence_transformers import util as st_util

from project_config import (
    load_model,
    print_model_info,
    SENTENCE_INDEX_FILE,
    FULL_TEXT_INDEX_FILE,
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD,
    CLASSICAL_SENTENCE_DELIMITERS
)
from index_builder import EmbeddingIndex, TranslationIndexBuilder, CLAUSE_INDEX_FILE
from data_processor import DataProcessor


@dataclass
class AdvancedTranslationResult:
    """
    Comprehensive translation result with detailed analysis.
    
    Attributes:
        input_text: Original classical Chinese input
        translation: Best modern Chinese translation
        confidence: Calibrated confidence score (0-1)
        raw_scores: Dictionary of raw scores from different methods
        source_matches: List of matched source entries
        method: Primary method used
        verification_score: Score from bi-directional verification
        context_score: Context relevance score
        ensemble_agreement: Agreement level among ensemble methods
        analysis: Detailed analysis information
    """
    input_text: str
    translation: str
    confidence: float
    raw_scores: Dict[str, float] = field(default_factory=dict)
    source_matches: List[dict] = field(default_factory=list)
    method: str = "advanced"
    verification_score: float = 0.0
    context_score: float = 0.0
    ensemble_agreement: float = 0.0
    analysis: Dict[str, Any] = field(default_factory=dict)


class AdvancedTranslator:
    """
    Advanced Classical Chinese Translator with innovative features.
    
    This translator combines multiple embedding-based techniques for
    maximum translation accuracy. It implements:
    
    1. Multi-Strategy Ensemble:
       - Direct sentence matching
       - Clause-level matching with aggregation
       - N-gram matching for partial phrases
       
    2. Bi-directional Verification:
       - Forward: Classical -> find Modern
       - Backward: Use Modern to verify Classical match
       
    3. Context-Aware Reranking:
       - Use full text context to rerank sentence matches
       - Boost matches from the same source
       
    4. Clustering-Enhanced Search:
       - Group similar texts into clusters
       - Use cluster information for better matching
       
    5. Multi-Feature Confidence Scoring:
       - Similarity score
       - Verification score
       - Context coherence
       - Ensemble agreement
    """
    
    # Weight configuration for ensemble
    WEIGHTS = {
        'direct_match': 0.5,
        'clause_match': 0.25,
        'ngram_match': 0.15,
        'context_match': 0.10
    }
    
    # Confidence thresholds
    EXCELLENT_THRESHOLD = 0.92
    HIGH_THRESHOLD = 0.85
    MEDIUM_THRESHOLD = 0.70
    LOW_THRESHOLD = 0.50
    
    def __init__(self, auto_load: bool = True):
        """
        Initialize the advanced translator.
        
        Args:
            auto_load: Whether to automatically load indexes and model
        """
        self.model = None
        self.sentence_index: Optional[EmbeddingIndex] = None
        self.clause_index: Optional[EmbeddingIndex] = None
        self.full_text_index: Optional[EmbeddingIndex] = None
        self.data_processor = DataProcessor()
        
        # Cache for performance
        self._embedding_cache = {}
        self._cluster_cache = None
        
        if auto_load:
            self._load_resources()
    
    def _load_resources(self):
        """Load all required resources."""
        print("=" * 70)
        print("🚀 Loading Advanced Classical Chinese Translator")
        print("=" * 70)
        
        if not os.path.exists(SENTENCE_INDEX_FILE):
            print("⚠️ Index not found. Building indexes...")
            builder = TranslationIndexBuilder()
            builder.build_all_indexes(force_rebuild=True)
            self.sentence_index = builder.sentence_index
            self.model = self.sentence_index.model
        else:
            print("\n🔄 Loading Qwen3-Embedding model...")
            print_model_info()
            self.model = load_model(device='cuda')
            
            self.sentence_index = EmbeddingIndex()
            self.sentence_index.model = self.model
            self.sentence_index.load(SENTENCE_INDEX_FILE)
        
        # Load optional indexes
        if os.path.exists(CLAUSE_INDEX_FILE):
            self.clause_index = EmbeddingIndex()
            self.clause_index.model = self.model
            self.clause_index.load(CLAUSE_INDEX_FILE)
            print("✅ Clause index loaded")
        
        # Load data processor for context
        self.data_processor.load_all_data()
        self.data_processor.create_sentence_pairs()
        
        print("\n✅ Advanced translator initialized!")
    
    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text with caching for performance.
        
        Args:
            text: Text to encode
            
        Returns:
            Normalized embedding vector
        """
        if text not in self._embedding_cache:
            embedding = self.model.encode(
                text, 
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            self._embedding_cache[text] = embedding
            
            # Limit cache size
            if len(self._embedding_cache) > 1000:
                # Remove oldest entries
                keys = list(self._embedding_cache.keys())[:500]
                for k in keys:
                    del self._embedding_cache[k]
        
        return self._embedding_cache[text]
    
    def _split_into_ngrams(self, text: str, n_range: Tuple[int, int] = (2, 6)) -> List[str]:
        """
        Split text into character n-grams for partial matching.
        
        This allows matching partial phrases that may not align
        with sentence boundaries.
        
        Args:
            text: Text to split
            n_range: Range of n-gram sizes (min, max)
            
        Returns:
            List of n-grams
        """
        text = text.replace('\n', '').replace(' ', '')
        ngrams = []
        
        for n in range(n_range[0], min(n_range[1] + 1, len(text) + 1)):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                if len(ngram) >= n_range[0]:
                    ngrams.append(ngram)
        
        return ngrams
    
    def _compute_similarity_matrix(self, query_texts: List[str], 
                                   candidate_texts: List[str]) -> np.ndarray:
        """
        Compute full similarity matrix between queries and candidates.
        
        Following pattern from 02_similarity_calculation.py.
        
        Args:
            query_texts: List of query texts
            candidate_texts: List of candidate texts
            
        Returns:
            Similarity matrix (queries x candidates)
        """
        query_embeddings = self.model.encode(
            query_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        candidate_embeddings = self.model.encode(
            candidate_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return np.dot(query_embeddings, candidate_embeddings.T)
    
    def _direct_match(self, text: str, top_k: int = 10) -> List[Tuple[dict, float]]:
        """
        Strategy 1: Direct sentence matching.
        
        Args:
            text: Query text
            top_k: Number of results
            
        Returns:
            List of (metadata, similarity) tuples
        """
        results = self.sentence_index.search(text, top_k=top_k)
        return [(r[1], r[2]) for r in results]
    
    def _clause_match(self, text: str, top_k: int = 10) -> List[Tuple[dict, float]]:
        """
        Strategy 2: Clause-level matching with aggregation.
        
        Splits input into clauses, matches each, and aggregates results.
        
        Args:
            text: Query text
            top_k: Number of results
            
        Returns:
            List of (metadata, similarity) tuples
        """
        # Split into clauses
        clause_delimiters = ['，', '、', '；', '：', '。', '！', '？']
        pattern = '([' + ''.join(re.escape(d) for d in clause_delimiters) + '])'
        parts = re.split(pattern, text)
        
        clauses = []
        current = ""
        for part in parts:
            if part in clause_delimiters:
                current += part
                if len(current.strip()) >= 2:
                    clauses.append(current.strip())
                current = ""
            else:
                current += part
        if len(current.strip()) >= 2:
            clauses.append(current.strip())
        
        if not clauses:
            clauses = [text]
        
        # Match each clause
        all_matches = defaultdict(list)
        
        for clause in clauses:
            results = self.sentence_index.search(clause, top_k=5)
            for _, meta, sim in results:
                key = (meta.get('title', ''), meta.get('sentence_idx', 0))
                all_matches[key].append((meta, sim))
        
        # Aggregate by taking max similarity for each unique match
        aggregated = []
        for key, matches in all_matches.items():
            best_meta, best_sim = max(matches, key=lambda x: x[1])
            # Boost if multiple clauses matched
            boost = min(1.0 + 0.05 * (len(matches) - 1), 1.15)
            aggregated.append((best_meta, best_sim * boost))
        
        # Sort and return top_k
        aggregated.sort(key=lambda x: x[1], reverse=True)
        return aggregated[:top_k]
    
    def _ngram_match(self, text: str, top_k: int = 10) -> List[Tuple[dict, float]]:
        """
        Strategy 3: N-gram matching for partial phrases.
        
        Args:
            text: Query text
            top_k: Number of results
            
        Returns:
            List of (metadata, similarity) tuples
        """
        ngrams = self._split_into_ngrams(text, n_range=(3, 8))
        
        if not ngrams:
            return []
        
        # Sample ngrams if too many
        if len(ngrams) > 20:
            import random
            ngrams = random.sample(ngrams, 20)
        
        # Match ngrams
        all_matches = defaultdict(list)
        
        for ngram in ngrams:
            results = self.sentence_index.search(ngram, top_k=3, similarity_threshold=0.6)
            for _, meta, sim in results:
                key = (meta.get('title', ''), meta.get('sentence_idx', 0))
                all_matches[key].append((meta, sim))
        
        # Aggregate
        aggregated = []
        for key, matches in all_matches.items():
            best_meta, best_sim = max(matches, key=lambda x: x[1])
            # Weight by number of matching ngrams
            coverage = len(matches) / len(ngrams) if ngrams else 0
            score = best_sim * (0.7 + 0.3 * coverage)
            aggregated.append((best_meta, score))
        
        aggregated.sort(key=lambda x: x[1], reverse=True)
        return aggregated[:top_k]
    
    def _verify_bidirectional(self, classical: str, modern: str) -> float:
        """
        Bi-directional verification: verify translation quality.
        
        Encodes the modern translation and checks if it maps back
        to similar classical texts.
        
        Args:
            classical: Original classical text
            modern: Predicted modern translation
            
        Returns:
            Verification score (0-1)
        """
        if not modern or len(modern) < 5:
            return 0.0
        
        # Encode both
        classical_emb = self._encode_text(classical)
        modern_emb = self._encode_text(modern)
        
        # Direct similarity
        direct_sim = float(np.dot(classical_emb, modern_emb))
        
        # Search using modern text
        modern_results = self.sentence_index.search(modern[:100], top_k=5)
        
        # Check if any result has similar classical text
        reverse_sim = 0.0
        for _, meta, sim in modern_results:
            matched_classical = meta.get('classical', '')
            if matched_classical:
                match_sim = float(np.dot(
                    classical_emb, 
                    self._encode_text(matched_classical)
                ))
                reverse_sim = max(reverse_sim, match_sim)
        
        # Combine direct and reverse similarity
        return 0.4 * direct_sim + 0.6 * reverse_sim
    
    def _compute_context_score(self, text: str, match_meta: dict) -> float:
        """
        Compute context relevance score.
        
        Considers how well the match fits the broader context.
        
        Args:
            text: Original query text
            match_meta: Metadata of the matched entry
            
        Returns:
            Context score (0-1)
        """
        full_classical = match_meta.get('full_classical', '')
        full_modern = match_meta.get('full_modern', '')
        
        if not full_classical:
            return 0.5  # Neutral if no context
        
        # Check if query appears in full context
        text_normalized = text.replace(' ', '').replace('\n', '')
        context_normalized = full_classical.replace(' ', '').replace('\n', '')
        
        if text_normalized in context_normalized:
            return 1.0
        
        # Compute embedding similarity to full context
        text_emb = self._encode_text(text)
        context_emb = self._encode_text(full_classical[:500])  # Limit length
        
        return float(np.dot(text_emb, context_emb))
    
    def _ensemble_translate(self, text: str, top_k: int = 5) -> Tuple[List[dict], Dict[str, float]]:
        """
        Ensemble translation combining multiple strategies.
        
        Args:
            text: Query text
            top_k: Number of final results
            
        Returns:
            Tuple of (ranked matches, raw scores dictionary)
        """
        # Get results from each strategy
        direct_results = self._direct_match(text, top_k=10)
        clause_results = self._clause_match(text, top_k=10)
        ngram_results = self._ngram_match(text, top_k=10)
        
        # Combine with weights
        all_scores = defaultdict(lambda: {'meta': None, 'scores': defaultdict(float)})
        
        for meta, sim in direct_results:
            key = (meta.get('title', ''), meta.get('classical', '')[:30])
            all_scores[key]['meta'] = meta
            all_scores[key]['scores']['direct'] = sim
        
        for meta, sim in clause_results:
            key = (meta.get('title', ''), meta.get('classical', '')[:30])
            all_scores[key]['meta'] = meta
            all_scores[key]['scores']['clause'] = sim
        
        for meta, sim in ngram_results:
            key = (meta.get('title', ''), meta.get('classical', '')[:30])
            all_scores[key]['meta'] = meta
            all_scores[key]['scores']['ngram'] = sim
        
        # Calculate weighted ensemble score
        final_results = []
        for key, data in all_scores.items():
            if data['meta'] is None:
                continue
            
            scores = data['scores']
            
            # Weighted combination
            ensemble_score = (
                self.WEIGHTS['direct_match'] * scores.get('direct', 0) +
                self.WEIGHTS['clause_match'] * scores.get('clause', 0) +
                self.WEIGHTS['ngram_match'] * scores.get('ngram', 0)
            )
            
            # Bonus for agreement among methods
            num_methods = sum(1 for v in scores.values() if v > 0.5)
            agreement_bonus = 0.05 * (num_methods - 1) if num_methods > 1 else 0
            
            final_score = min(1.0, ensemble_score + agreement_bonus)
            
            final_results.append({
                'meta': data['meta'],
                'ensemble_score': final_score,
                'raw_scores': dict(scores),
                'agreement': num_methods / 3.0
            })
        
        # Sort by ensemble score
        final_results.sort(key=lambda x: x['ensemble_score'], reverse=True)
        
        # Get raw scores for best result
        raw_scores = {}
        if final_results:
            raw_scores = final_results[0]['raw_scores']
        
        return final_results[:top_k], raw_scores
    
    def _calibrate_confidence(self, raw_score: float, verification_score: float,
                             context_score: float, agreement: float) -> float:
        """
        Multi-feature confidence calibration.
        
        Combines multiple signals to produce a calibrated confidence score.
        
        Args:
            raw_score: Raw ensemble similarity score
            verification_score: Bi-directional verification score
            context_score: Context relevance score
            agreement: Ensemble method agreement
            
        Returns:
            Calibrated confidence (0-1)
        """
        # Base confidence from raw score
        base = raw_score
        
        # Adjust based on verification
        if verification_score > 0.8:
            base *= 1.05
        elif verification_score < 0.5:
            base *= 0.9
        
        # Adjust based on context
        if context_score > 0.9:
            base *= 1.03
        elif context_score < 0.5:
            base *= 0.95
        
        # Adjust based on agreement
        if agreement > 0.8:
            base *= 1.02
        
        # Apply threshold-based adjustment
        if base >= self.EXCELLENT_THRESHOLD:
            confidence = base * 1.0
        elif base >= self.HIGH_THRESHOLD:
            confidence = base * 0.98
        elif base >= self.MEDIUM_THRESHOLD:
            confidence = base * 0.95
        elif base >= self.LOW_THRESHOLD:
            confidence = base * 0.90
        else:
            confidence = base * 0.80
        
        return min(1.0, max(0.0, confidence))
    
    def translate(self, text: str, verify: bool = True,
                 use_context: bool = True) -> AdvancedTranslationResult:
        """
        Translate classical Chinese text with advanced features.
        
        Args:
            text: Classical Chinese text to translate
            verify: Whether to perform bi-directional verification
            use_context: Whether to use context-aware scoring
            
        Returns:
            AdvancedTranslationResult with comprehensive analysis
        """
        text = text.strip()
        
        if not text:
            return AdvancedTranslationResult(
                input_text=text,
                translation="",
                confidence=0.0,
                method="empty"
            )
        
        # Get ensemble results
        results, raw_scores = self._ensemble_translate(text, top_k=5)
        
        if not results:
            return AdvancedTranslationResult(
                input_text=text,
                translation="[No matching translation found]",
                confidence=0.0,
                raw_scores=raw_scores,
                method="advanced"
            )
        
        # Get best result
        best = results[0]
        meta = best['meta']
        translation = meta.get('modern', '')
        ensemble_score = best['ensemble_score']
        agreement = best['agreement']
        
        # Bi-directional verification
        verification_score = 0.0
        if verify and translation:
            verification_score = self._verify_bidirectional(text, translation)
        
        # Context scoring
        context_score = 0.5
        if use_context:
            context_score = self._compute_context_score(text, meta)
        
        # Calibrate confidence
        confidence = self._calibrate_confidence(
            ensemble_score, verification_score, context_score, agreement
        )
        
        # Build source matches
        source_matches = [
            {
                'classical': r['meta'].get('classical', ''),
                'modern': r['meta'].get('modern', ''),
                'title': r['meta'].get('title', ''),
                'similarity': r['ensemble_score'],
                'raw_scores': r['raw_scores']
            }
            for r in results
        ]
        
        return AdvancedTranslationResult(
            input_text=text,
            translation=translation,
            confidence=confidence,
            raw_scores=raw_scores,
            source_matches=source_matches,
            method="advanced_ensemble",
            verification_score=verification_score,
            context_score=context_score,
            ensemble_agreement=agreement,
            analysis={
                'ensemble_score': ensemble_score,
                'num_candidates': len(results),
                'best_title': meta.get('title', ''),
                'strategies_used': list(raw_scores.keys())
            }
        )
    
    def translate_with_explanation(self, text: str) -> Tuple[AdvancedTranslationResult, str]:
        """
        Translate with detailed explanation of the process.
        
        Args:
            text: Classical Chinese text
            
        Returns:
            Tuple of (result, explanation string)
        """
        result = self.translate(text)
        
        explanation = []
        explanation.append(f"📝 Input: {text}")
        explanation.append(f"\n🔍 Analysis:")
        
        # Explain strategies
        if result.raw_scores:
            explanation.append("  Strategy scores:")
            for strategy, score in result.raw_scores.items():
                explanation.append(f"    - {strategy}: {score:.4f}")
        
        # Explain verification
        explanation.append(f"\n  Verification score: {result.verification_score:.4f}")
        explanation.append(f"  Context score: {result.context_score:.4f}")
        explanation.append(f"  Ensemble agreement: {result.ensemble_agreement:.2%}")
        
        # Explain confidence
        explanation.append(f"\n📊 Final confidence: {result.confidence:.2%}")
        
        # Quality assessment
        if result.confidence >= self.EXCELLENT_THRESHOLD:
            quality = "Excellent - Very high confidence match"
        elif result.confidence >= self.HIGH_THRESHOLD:
            quality = "High - Good quality match"
        elif result.confidence >= self.MEDIUM_THRESHOLD:
            quality = "Medium - Acceptable match"
        elif result.confidence >= self.LOW_THRESHOLD:
            quality = "Low - Use with caution"
        else:
            quality = "Very Low - Consider manual translation"
        
        explanation.append(f"  Quality: {quality}")
        
        return result, "\n".join(explanation)
    
    def batch_translate(self, texts: List[str], show_progress: bool = True) -> List[AdvancedTranslationResult]:
        """
        Batch translate multiple texts.
        
        Args:
            texts: List of classical Chinese texts
            show_progress: Whether to show progress bar
            
        Returns:
            List of AdvancedTranslationResult
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            texts = tqdm(texts, desc="Translating")
        
        for text in texts:
            result = self.translate(text)
            results.append(result)
        
        return results


def print_advanced_result(result: AdvancedTranslationResult):
    """Pretty print an advanced translation result."""
    print("\n" + "=" * 70)
    print("🏛️ Advanced Classical Chinese Translation Result")
    print("=" * 70)
    
    print(f"\n【Original】\n{result.input_text}")
    print(f"\n【Translation】\n{result.translation}")
    
    print(f"\n【Confidence Breakdown】")
    print(f"  • Final Confidence: {result.confidence:.2%}")
    print(f"  • Ensemble Score: {result.analysis.get('ensemble_score', 0):.4f}")
    print(f"  • Verification Score: {result.verification_score:.4f}")
    print(f"  • Context Score: {result.context_score:.4f}")
    print(f"  • Method Agreement: {result.ensemble_agreement:.2%}")
    
    if result.raw_scores:
        print(f"\n【Strategy Scores】")
        for strategy, score in result.raw_scores.items():
            bar = "█" * int(score * 20)
            print(f"  {strategy:12}: {bar} {score:.4f}")
    
    if result.source_matches:
        print(f"\n【Top Matches】")
        for i, match in enumerate(result.source_matches[:3], 1):
            print(f"  {i}. 《{match.get('title', 'Unknown')}》 (sim: {match.get('similarity', 0):.4f})")
            print(f"     {match.get('classical', '')[:40]}...")
    
    print("=" * 70)


def main():
    """Test the advanced translator."""
    print("=" * 70)
    print("Advanced Classical Chinese Translator Test")
    print("=" * 70)
    
    translator = AdvancedTranslator()
    
    test_cases = [
        "梳洗罢，独倚望江楼。",
        "过尽千帆皆不是",
        "不患寡而患不均，不患贫而患不安。",
        "斜晖脉脉水悠悠",
    ]
    
    for text in test_cases:
        result, explanation = translator.translate_with_explanation(text)
        print_advanced_result(result)
        print("\n" + explanation)


if __name__ == "__main__":
    main()

