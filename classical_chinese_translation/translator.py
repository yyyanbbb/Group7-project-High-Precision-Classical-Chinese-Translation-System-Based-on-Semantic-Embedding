#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Chinese Translator

This module provides the core translation functionality using
semantic similarity search. It implements multiple translation
strategies for improved accuracy.

Translation approaches:
1. Direct match: High-confidence exact or near-exact matches
2. Sentence-level: Match individual sentences
3. Multi-granularity fusion: Combine results from different granularities
4. Context-aware: Use surrounding context for better matching
"""
import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

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


@dataclass
class TranslationResult:
    """
    Translation result container.
    
    Attributes:
        input_text: The original classical Chinese input
        translation: The modern Chinese translation
        confidence: Confidence score (0-1)
        source_matches: List of matched source entries
        method: Translation method used
        details: Additional details about the translation
    """
    input_text: str
    translation: str
    confidence: float
    source_matches: List[dict]
    method: str
    details: dict = field(default_factory=dict)


class ClassicalChineseTranslator:
    """
    Classical Chinese to Modern Chinese translator.
    
    This translator uses semantic similarity search to find the best
    matching translations from a pre-built index. It supports multiple
    translation strategies for improved accuracy.
    
    Features:
    - Multi-granularity search (clause, sentence, full text)
    - Confidence calibration
    - Context-aware matching
    - Batch translation support
    """
    
    # Confidence calibration thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.90
    MEDIUM_CONFIDENCE_THRESHOLD = 0.75
    LOW_CONFIDENCE_THRESHOLD = 0.60
    
    def __init__(self, auto_load: bool = True):
        """
        Initialize the translator.
        
        Args:
            auto_load: Whether to automatically load indexes
        """
        self.model = None
        self.sentence_index: Optional[EmbeddingIndex] = None
        self.clause_index: Optional[EmbeddingIndex] = None
        self.full_text_index: Optional[EmbeddingIndex] = None
        self.builder: Optional[TranslationIndexBuilder] = None
        
        if auto_load:
            self.load_indexes()
    
    def load_indexes(self):
        """Load all available indexes."""
        print("=" * 60)
        print("📚 Loading Classical Chinese Translation Indexes...")
        print("=" * 60)
        
        # Check if sentence index exists (required)
        if not os.path.exists(SENTENCE_INDEX_FILE):
            print("⚠️ Sentence index not found. Building indexes...")
            self.build_indexes()
            return
        
        # Load model
        print("\n🔄 Loading Qwen3-Embedding model...")
        print_model_info()
        self.model = load_model(device='cuda')
        
        # Load sentence index (required)
        self.sentence_index = EmbeddingIndex()
        self.sentence_index.model = self.model
        self.sentence_index.load(SENTENCE_INDEX_FILE)
        
        # Load clause index (optional)
        if os.path.exists(CLAUSE_INDEX_FILE):
            self.clause_index = EmbeddingIndex()
            self.clause_index.model = self.model
            self.clause_index.load(CLAUSE_INDEX_FILE)
        else:
            print("ℹ️ Clause index not available (fine-grained matching disabled)")
            self.clause_index = None
        
        # Load full text index (optional)
        if os.path.exists(FULL_TEXT_INDEX_FILE):
            self.full_text_index = EmbeddingIndex()
            self.full_text_index.model = self.model
            self.full_text_index.load(FULL_TEXT_INDEX_FILE)
        else:
            print("ℹ️ Full text index not available")
            self.full_text_index = None
        
        print("\n✅ Indexes loaded successfully!")
    
    def build_indexes(self, force_rebuild: bool = False):
        """
        Build translation indexes.
        
        Args:
            force_rebuild: Whether to force rebuild existing indexes
        """
        self.builder = TranslationIndexBuilder()
        self.builder.build_all_indexes(force_rebuild=force_rebuild)
        
        self.sentence_index = self.builder.sentence_index
        self.model = self.sentence_index.model
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using classical Chinese delimiters.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        pattern = '([' + ''.join(re.escape(d) for d in CLASSICAL_SENTENCE_DELIMITERS) + '])'
        parts = re.split(pattern, text)
        
        sentences = []
        current = ""
        
        for part in parts:
            if part in CLASSICAL_SENTENCE_DELIMITERS:
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += part
        
        if current.strip():
            sentences.append(current.strip())
        
        return [s for s in sentences if len(s) >= 2]
    
    def _calibrate_confidence(self, similarity: float, method: str) -> float:
        """
        Calibrate confidence score based on similarity and method.
        
        Higher similarities are given stronger confidence, while
        lower similarities are penalized more heavily.
        
        Args:
            similarity: Raw similarity score
            method: Translation method used
            
        Returns:
            Calibrated confidence score
        """
        if similarity >= self.HIGH_CONFIDENCE_THRESHOLD:
            # High confidence - slight boost
            return min(1.0, similarity * 1.05)
        elif similarity >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            # Medium confidence - slight penalty
            return similarity * 0.95
        elif similarity >= self.LOW_CONFIDENCE_THRESHOLD:
            # Low confidence - moderate penalty
            return similarity * 0.85
        else:
            # Very low confidence - heavy penalty
            return similarity * 0.7
    
    def translate_sentence(self, sentence: str, top_k: int = DEFAULT_TOP_K,
                          threshold: float = SIMILARITY_THRESHOLD) -> TranslationResult:
        """
        Translate a single sentence.
        
        Uses the sentence index for direct matching.
        
        Args:
            sentence: Classical Chinese sentence
            top_k: Number of candidates to consider
            threshold: Minimum similarity threshold
            
        Returns:
            TranslationResult object
        """
        if self.sentence_index is None:
            raise ValueError("Indexes not loaded. Call load_indexes() first.")
        
        # Search for similar sentences
        results = self.sentence_index.search(sentence, top_k=top_k, similarity_threshold=threshold)
        
        if not results:
            return TranslationResult(
                input_text=sentence,
                translation="[No matching translation found]",
                confidence=0.0,
                source_matches=[],
                method="sentence",
                details={'reason': 'no_match'}
            )
        
        # Get best match
        best_text, best_meta, best_sim = results[0]
        translation = best_meta.get('modern', '')
        
        # Calibrate confidence
        confidence = self._calibrate_confidence(best_sim, "sentence")
        
        # Build source matches
        source_matches = [
            {
                'classical': r[1].get('classical', ''),
                'modern': r[1].get('modern', ''),
                'title': r[1].get('title', ''),
                'similarity': r[2]
            }
            for r in results
        ]
        
        return TranslationResult(
            input_text=sentence,
            translation=translation,
            confidence=confidence,
            source_matches=source_matches,
            method="sentence",
            details={'raw_similarity': best_sim}
        )
    
    def translate_with_fusion(self, text: str, top_k: int = DEFAULT_TOP_K,
                             threshold: float = SIMILARITY_THRESHOLD) -> TranslationResult:
        """
        Translate using multi-granularity fusion.
        
        Combines results from clause and sentence levels for better accuracy.
        
        Args:
            text: Classical Chinese text
            top_k: Number of candidates per level
            threshold: Minimum similarity threshold
            
        Returns:
            TranslationResult object
        """
        if self.sentence_index is None:
            raise ValueError("Indexes not loaded. Call load_indexes() first.")
        
        # Sentence-level search
        sentence_results = self.sentence_index.search(text, top_k=top_k, similarity_threshold=threshold)
        
        # Clause-level search (if available)
        clause_results = []
        if self.clause_index is not None:
            clause_results = self.clause_index.search(text, top_k=top_k, similarity_threshold=threshold)
        
        # Fuse results with weighted scoring
        all_results = []
        
        # Sentence results (weight: 1.0)
        for text_match, meta, sim in sentence_results:
            all_results.append({
                'text': text_match,
                'meta': meta,
                'similarity': sim,
                'weight': 1.0,
                'level': 'sentence'
            })
        
        # Clause results (weight: 0.8 - they're more specific but may miss context)
        for text_match, meta, sim in clause_results:
            all_results.append({
                'text': text_match,
                'meta': meta,
                'similarity': sim * 0.8,  # Slight penalty for clause level
                'weight': 0.8,
                'level': 'clause'
            })
        
        if not all_results:
            return TranslationResult(
                input_text=text,
                translation="[No matching translation found]",
                confidence=0.0,
                source_matches=[],
                method="fusion",
                details={'reason': 'no_match'}
            )
        
        # Sort by weighted similarity
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get best result
        best = all_results[0]
        translation = best['meta'].get('modern', '')
        confidence = self._calibrate_confidence(best['similarity'], "fusion")
        
        # Build source matches
        source_matches = [
            {
                'classical': r['meta'].get('classical', ''),
                'modern': r['meta'].get('modern', ''),
                'title': r['meta'].get('title', ''),
                'similarity': r['similarity'],
                'level': r['level']
            }
            for r in all_results[:top_k]
        ]
        
        return TranslationResult(
            input_text=text,
            translation=translation,
            confidence=confidence,
            source_matches=source_matches,
            method="fusion",
            details={'best_level': best['level'], 'raw_similarity': best['similarity']}
        )
    
    def translate_text(self, text: str, top_k: int = DEFAULT_TOP_K,
                      threshold: float = SIMILARITY_THRESHOLD) -> TranslationResult:
        """
        Translate complete text with intelligent strategy selection.
        
        For single sentences: use direct matching
        For multiple sentences: translate each and combine
        
        Args:
            text: Classical Chinese text
            top_k: Number of candidates
            threshold: Minimum similarity threshold
            
        Returns:
            TranslationResult object
        """
        if self.sentence_index is None:
            raise ValueError("Indexes not loaded. Call load_indexes() first.")
        
        # First try full text match if available and text is short
        if self.full_text_index is not None and len(text) <= 200:
            full_results = self.full_text_index.search(text, top_k=1, similarity_threshold=0.85)
            if full_results and full_results[0][2] >= 0.85:
                best_text, best_meta, best_sim = full_results[0]
                return TranslationResult(
                    input_text=text,
                    translation=best_meta.get('modern', ''),
                    confidence=self._calibrate_confidence(best_sim, "full_text"),
                    source_matches=[{
                        'classical': best_meta.get('classical', ''),
                        'modern': best_meta.get('modern', ''),
                        'title': best_meta.get('title', ''),
                        'author': best_meta.get('author', ''),
                        'similarity': best_sim
                    }],
                    method="full_text",
                    details={'raw_similarity': best_sim}
                )
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            # Single sentence - use fusion for better accuracy
            return self.translate_with_fusion(text, top_k=top_k, threshold=threshold)
        
        # Multiple sentences - translate each
        translations = []
        all_matches = []
        total_confidence = 0.0
        
        for sentence in sentences:
            result = self.translate_with_fusion(sentence, top_k=top_k, threshold=threshold)
            translations.append(result.translation)
            if result.source_matches:
                all_matches.append(result.source_matches[0])
            total_confidence += result.confidence
        
        avg_confidence = total_confidence / len(sentences) if sentences else 0.0
        combined_translation = '\n'.join(translations)
        
        return TranslationResult(
            input_text=text,
            translation=combined_translation,
            confidence=avg_confidence,
            source_matches=all_matches,
            method="sentence_by_sentence",
            details={'sentence_count': len(sentences)}
        )
    
    def translate(self, text: str, mode: str = "auto", top_k: int = DEFAULT_TOP_K,
                 threshold: float = SIMILARITY_THRESHOLD) -> TranslationResult:
        """
        Main translation entry point.
        
        Args:
            text: Classical Chinese text
            mode: Translation mode ("auto", "sentence", "fusion", "full_text")
            top_k: Number of candidates
            threshold: Minimum similarity threshold
            
        Returns:
            TranslationResult object
        """
        text = text.strip()
        
        if not text:
            return TranslationResult(
                input_text=text,
                translation="",
                confidence=0.0,
                source_matches=[],
                method="empty"
            )
        
        if mode == "sentence":
            return self.translate_sentence(text, top_k=top_k, threshold=threshold)
        elif mode == "fusion":
            return self.translate_with_fusion(text, top_k=top_k, threshold=threshold)
        elif mode == "full_text" and self.full_text_index is not None:
            results = self.full_text_index.search(text, top_k=top_k, similarity_threshold=threshold)
            if results:
                best_text, best_meta, best_sim = results[0]
                return TranslationResult(
                    input_text=text,
                    translation=best_meta.get('modern', ''),
                    confidence=self._calibrate_confidence(best_sim, "full_text"),
                    source_matches=[{
                        'classical': r[1].get('classical', ''),
                        'modern': r[1].get('modern', ''),
                        'title': r[1].get('title', ''),
                        'similarity': r[2]
                    } for r in results],
                    method="full_text"
                )
            return TranslationResult(
                input_text=text,
                translation="[No matching translation found]",
                confidence=0.0,
                source_matches=[],
                method="full_text"
            )
        else:  # auto
            return self.translate_text(text, top_k=top_k, threshold=threshold)
    
    def batch_translate(self, texts: List[str], mode: str = "auto",
                       show_progress: bool = True) -> List[TranslationResult]:
        """
        Batch translation for multiple texts.
        
        Args:
            texts: List of classical Chinese texts
            mode: Translation mode
            show_progress: Whether to show progress bar
            
        Returns:
            List of TranslationResult objects
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            texts = tqdm(texts, desc="Translating")
        
        for text in texts:
            result = self.translate(text, mode=mode)
            results.append(result)
        
        return results
    
    def get_similar_texts(self, text: str, top_k: int = 10) -> List[dict]:
        """
        Find similar classical texts.
        
        Args:
            text: Query text
            top_k: Number of results
            
        Returns:
            List of similar texts with metadata
        """
        results = self.sentence_index.search(text, top_k=top_k)
        
        return [
            {
                'classical': r[1].get('classical', ''),
                'modern': r[1].get('modern', ''),
                'title': r[1].get('title', ''),
                'similarity': r[2]
            }
            for r in results
        ]


def print_translation_result(result: TranslationResult, show_matches: bool = True):
    """
    Pretty print a translation result.
    
    Args:
        result: TranslationResult object
        show_matches: Whether to show source matches
    """
    print("\n" + "=" * 60)
    print("📜 Classical Chinese Translation Result")
    print("=" * 60)
    print(f"\n【Original】\n{result.input_text}")
    print(f"\n【Translation】\n{result.translation}")
    print(f"\n【Confidence】{result.confidence:.2%}")
    print(f"【Method】{result.method}")
    
    if show_matches and result.source_matches:
        print(f"\n【Reference Sources】")
        for i, match in enumerate(result.source_matches[:3], 1):
            title = match.get('title', 'Unknown')
            similarity = match.get('similarity', 0)
            classical = match.get('classical', '')[:50]
            print(f"  {i}. 《{title}》")
            print(f"     Original: {classical}...")
            print(f"     Similarity: {similarity:.2%}")
    
    print("=" * 60)


def main():
    """Test the translator."""
    print("=" * 60)
    print("Classical Chinese Translator Test")
    print("=" * 60)
    
    # Initialize translator
    translator = ClassicalChineseTranslator()
    
    # Test cases
    test_cases = [
        "梳洗罢，独倚望江楼。",
        "过尽千帆皆不是，斜晖脉脉水悠悠。",
        "不患寡而患不均，不患贫而患不安。",
        "白日依山尽，黄河入海流。",
    ]
    
    print("\n🧪 Running translation tests...")
    
    for text in test_cases:
        result = translator.translate(text)
        print_translation_result(result)


if __name__ == "__main__":
    main()
