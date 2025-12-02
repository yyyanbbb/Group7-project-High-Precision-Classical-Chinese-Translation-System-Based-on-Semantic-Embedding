#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precision Classical Chinese Translator

Ultimate high-accuracy translator using:
1. Smart data processing with noise filtering
2. Quality-weighted semantic search
3. Multi-strategy ensemble with voting
4. Bi-directional verification
5. Context-aware refinement
6. Confidence calibration

Goal: Maximize translation precision through intelligent matching.
"""
import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from project_config import load_model, print_model_info
from smart_index_builder import SmartIndexBuilder, SmartEmbeddingIndex
from llm_refiner import LocalLLMRefiner, RefinementResult


@dataclass
class PrecisionResult:
    """
    High-precision translation result.
    
    Contains comprehensive information about the translation
    including confidence breakdown and source tracking.
    """
    input_text: str
    translation: str
    confidence: float
    
    # Detailed scores
    semantic_score: float = 0.0
    verification_score: float = 0.0
    quality_score: float = 0.0
    ensemble_score: float = 0.0
    
    # Source information
    matched_title: str = ""
    matched_classical: str = ""
    raw_similarity: float = 0.0
    
    # All candidates for transparency
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    
    # Method information
    method: str = "precision"
    processing_notes: List[str] = field(default_factory=list)
    rewrites: List[Dict[str, str]] = field(default_factory=list)


class PrecisionTranslator:
    """
    High-precision classical Chinese translator.
    
    Uses multiple techniques to achieve maximum accuracy:
    
    1. Smart Index Search:
       - Quality-boosted semantic search
       - Pre-filtered noise-free data
       
    2. Multi-Candidate Analysis:
       - Analyze top candidates
       - Cross-reference verification
       
    3. Sentence-Level Fusion:
       - Combine results from multiple sentences
       - Context-aware selection
       
    4. Confidence Calibration:
       - Multi-factor confidence scoring
       - Uncertainty quantification
    """
    
    def __init__(self, auto_load: bool = True,
                 min_quality: float = 0.5,
                 enable_llm_refiner: bool = True):
        """
        Initialize the precision translator.
        
        Args:
            auto_load: Whether to auto-load indexes
        """
        self.model = None
        self.index: Optional[SmartEmbeddingIndex] = None
        self._embedding_cache = {}
        self._hash_to_index: Dict[str, int] = {}
        self.min_quality = min_quality
        self.llm_refiner = LocalLLMRefiner() if enable_llm_refiner else None
        
        if auto_load:
            self._load()
    
    def _load(self):
        """Load model and indexes."""
        print("=" * 60)
        print("🎯 Loading Precision Translator")
        print("=" * 60)
        
        builder = SmartIndexBuilder()
        builder.build_indexes(force_rebuild=False, min_quality=self.min_quality)
        self.index = builder.get_index()
        self.model = self.index.load_model()
        self._prepare_embedding_lookup()
        
        print("\n✅ Precision translator ready!")

    def _prepare_embedding_lookup(self):
        """Build fast lookup from text_hash to normalized embedding."""
        self._hash_to_index.clear()
        if not self.index or not getattr(self.index, "text_hashes", None):
            return
        if getattr(self.index, "embeddings_norm", None) is None:
            # Ensure normalized embeddings exist
            self.index._normalize()
        for idx, text_hash in enumerate(self.index.text_hashes):
            self._hash_to_index[text_hash] = idx
    def _apply_llm_refinement(
        self,
        sentence: str,
        best_candidate: Dict[str, Any],
        all_candidates: List[Dict[str, Any]]
    ) -> Optional[RefinementResult]:
        """Optionally refine translation via local LLM."""
        if not self.llm_refiner or not self.llm_refiner.available:
            return None

        # Take up to two supporting candidates excluding the primary one
        sorted_candidates = sorted(
            all_candidates,
            key=lambda c: c.get("score", 0),
            reverse=True
        )
        support = []
        for cand in sorted_candidates:
            if cand is best_candidate:
                continue
            support.append(cand)
            if len(support) >= 2:
                break

        return self.llm_refiner.refine_translation(
            sentence,
            best_candidate,
            support
        )

    def _generate_rewrites(
        self,
        sentence: str,
        candidates: List[Dict[str, Any]],
        styles: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Use local LLM to produce rewritten variants."""
        if not self.llm_refiner or not getattr(self.llm_refiner, "available", False):
            return []
        if not candidates:
            return []

        # Limit candidates to avoid prompt explosion
        top_candidates = candidates[:3]
        styles = styles or ["简明版（现代口语）", "文雅版（文学意译）"]

        try:
            rewrites = self.llm_refiner.rewrite_with_styles(
                original=sentence,
                candidates=top_candidates,
                styles=styles,
            )
        except Exception:
            return []
        return rewrites or []
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode with caching."""
        if text not in self._embedding_cache:
            emb = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            self._embedding_cache[text] = emb
            
            # Limit cache
            if len(self._embedding_cache) > 500:
                keys = list(self._embedding_cache.keys())[:250]
                for k in keys:
                    del self._embedding_cache[k]
        
        return self._embedding_cache[text]

    def _get_index_embedding(self, candidate: Dict[str, Any]) -> Optional[np.ndarray]:
        """Retrieve normalized embedding for a candidate via text hash."""
        text_hash = candidate.get('text_hash')
        if not text_hash:
            return None
        if text_hash not in self._hash_to_index:
            return None
        idx = self._hash_to_index[text_hash]
        embeddings_norm = getattr(self.index, "embeddings_norm", None)
        if embeddings_norm is None:
            self.index._normalize()
            embeddings_norm = self.index.embeddings_norm
        if embeddings_norm is None or idx >= embeddings_norm.shape[0]:
            return None
        return embeddings_norm[idx]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split into sentences."""
        delimiters = ['。', '！', '？', '；']
        pattern = '([' + ''.join(re.escape(d) for d in delimiters) + '])'
        parts = re.split(pattern, text)
        
        sentences = []
        current = ""
        for part in parts:
            if part in delimiters:
                current += part
                if len(current.strip()) >= 2:
                    sentences.append(current.strip())
                current = ""
            else:
                current += part
        
        if len(current.strip()) >= 2:
            sentences.append(current.strip())
        
        return sentences if sentences else [text]

    def _generate_query_variants(self, sentence: str) -> List[Tuple[str, float, str]]:
        """
        Generate query variants (original, normalized, clause-level, n-gram).
        
        Returns:
            List of tuples: (variant_text, weight, reason)
        """
        variants = []
        seen: set[str] = set()

        def add_variant(text: str, weight: float, reason: str):
            normalized = text.strip()
            if len(normalized) < 2:
                return
            key = normalized
            if key in seen:
                return
            seen.add(key)
            variants.append((normalized, weight, reason))

        # 1. Original (highest priority)
        add_variant(sentence, 1.0, "original")

        # 2. Remove light punctuation for normalization
        normalized = re.sub(r"[，、；：。！？\s]", "", sentence)
        if normalized != sentence:
            add_variant(normalized, 0.95, "punctuation_trim")

        # 3. Clause-level focus (take up to 3 clauses)
        clauses = [c.strip() for c in re.split(r"[，、；。]", sentence) if c.strip()]
        for i, clause in enumerate(clauses[:3]):
            weight = 0.9 - i * 0.05  # 0.9, 0.85, 0.8
            add_variant(clause, weight, f"clause_{i+1}")
        
        # 4. N-gram variants (for partial matching)
        chars = list(normalized)
        if len(chars) >= 4:
            # 4-gram sliding window
            for i in range(len(chars) - 3):
                ngram = ''.join(chars[i:i+4])
                add_variant(ngram, 0.7, f"4gram_{i}")
        
        if len(chars) >= 6:
            # 6-gram sliding window (more context)
            for i in range(len(chars) - 5):
                ngram = ''.join(chars[i:i+6])
                add_variant(ngram, 0.75, f"6gram_{i}")

        # 5. Leading segment (first half) for long sentences
        if len(sentence) > 15:
            half = sentence[: len(sentence) // 2].strip()
            add_variant(half, 0.8, "leading_half")
            
        # 6. Trailing segment (last half)
        if len(sentence) > 15:
            half = sentence[len(sentence) // 2:].strip()
            add_variant(half, 0.75, "trailing_half")

        return variants

    def _literal_overlap(self, text_a: str, text_b: str) -> float:
        """Simple character-level overlap ratio."""
        if not text_a or not text_b:
            return 0.0
        set_a = set(text_a)
        set_b = set(text_b)
        union = len(set_a.union(set_b))
        if union == 0:
            return 0.0
        return len(set_a.intersection(set_b)) / union
    
    def _verify_translation(self, classical: str, modern: str) -> float:
        """
        Verify translation quality using reverse matching.
        
        Checks if the modern translation, when used as a query,
        returns similar classical texts.
        
        Args:
            classical: Original classical text
            modern: Translated modern text
            
        Returns:
            Verification score (0-1)
        """
        if not modern or len(modern) < 5:
            return 0.0
        
        # Encode both
        classical_emb = self._encode(classical)
        modern_emb = self._encode(modern[:200])  # Limit length
        
        # Direct semantic similarity
        direct_sim = float(np.dot(classical_emb, modern_emb))
        
        # Reverse search
        reverse_results = self.index.search(modern[:100], top_k=3, quality_boost=False)
        
        reverse_sim = 0.0
        for text, meta, sim in reverse_results:
            # Check if matched text is similar to our classical
            matched_emb = self._encode(text)
            match_sim = float(np.dot(classical_emb, matched_emb))
            reverse_sim = max(reverse_sim, match_sim)
        
        # Combine scores
        return 0.3 * direct_sim + 0.7 * reverse_sim
    
    def _select_best_candidate(
        self,
        sentence: str,
        candidates: List[Dict[str, Any]],
        style: str = "auto",
        query_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Intelligently select the best candidate with enhanced scoring.
        
        Args:
            sentence: Input sentence
            candidates: List of candidate dicts
            style: "literal", "interpretive", or "auto"
            
        Returns:
            Tuple of (best_candidate, confidence_score)
        """
        if not candidates:
            return None, 0.0
        
        # Remove duplicates (by modern text)
        unique_candidates = []
        seen = set()
        for c in candidates:
            if c['modern'] not in seen:
                seen.add(c['modern'])
                unique_candidates.append(c)
        candidates = unique_candidates
        
        # Normalize input for exact matching
        input_normalized = re.sub(r"[，、；：。！？\s]", "", sentence)
        
        best_score = -1.0
        best_candidate = None
        style_mode = style.lower() if isinstance(style, str) else "auto"
        if style_mode not in {"literal", "interpretive", "auto"}:
            style_mode = "auto"
        
        for cand in candidates:
            # Normalize classical text
            classical_normalized = re.sub(r"[，、；：。！？\s]", "", cand['classical'])
            
            # Base score from vector similarity
            score = cand['base_score']
            # Semantic alignment against current query
            if query_embedding is None:
                query_embedding = self._encode(sentence)
            cand_embedding = self._get_index_embedding(cand)
            if cand_embedding is None:
                cand_embedding = self._encode(cand['classical'])
            semantic_alignment = float(np.dot(query_embedding, cand_embedding))
            cand['semantic_alignment'] = semantic_alignment
            score = score * 0.7 + semantic_alignment * 0.3
            
            # ===== EXACT MATCH DETECTION (HIGHEST PRIORITY) =====
            exact_match = False
            partial_match = False
            
            if input_normalized == classical_normalized:
                # Perfect exact match - massive bonus
                score += 0.5
                exact_match = True
                cand['match_type'] = 'exact'
            elif len(input_normalized) >= 4:
                # Check for substring containment
                if input_normalized in classical_normalized:
                    containment_ratio = len(input_normalized) / len(classical_normalized)
                    score += 0.25 * containment_ratio
                    partial_match = True
                    cand['match_type'] = 'input_in_classical'
                elif classical_normalized in input_normalized:
                    containment_ratio = len(classical_normalized) / len(input_normalized)
                    score += 0.2 * containment_ratio
                    partial_match = True
                    cand['match_type'] = 'classical_in_input'
            
            cand['exact_match'] = exact_match
            cand['partial_match'] = partial_match
            
            # Apply query weight influence
            weight = float(cand.get('variant_weight', 1.0))
            score *= (0.85 + 0.15 * max(0.5, min(1.5, weight)))
            
            # ===== LENGTH HEURISTIC =====
            c_len = max(1, len(sentence))
            m_len = max(1, len(cand['modern']))
            length_ratio = m_len / c_len
            if length_ratio < 0.8:
                score *= 0.75  # Too short - likely incomplete
            elif length_ratio > 5.0:
                score *= 0.85  # Too verbose
            else:
                # Optimal range 1.0 - 3.0
                score *= 0.9 + 0.1 * min(1.0, length_ratio / 3.0)
            
            # ===== ALIGNMENT QUALITY BOOST =====
            quality = cand['alignment_score']
            score *= (0.75 + 0.25 * quality)  # Quality matters significantly
            
            # ===== VERIFICATION (Skip for exact matches to save time) =====
            if not exact_match:
                verify_score = self._verify_translation(sentence, cand['modern'])
                # Blend verification with semantic alignment
                combined_semantic = 0.5 * semantic_alignment + 0.5 * verify_score
                score = score * 0.55 + combined_semantic * 0.45
                cand['verification'] = verify_score
            else:
                cand['verification'] = 1.0  # Assume perfect for exact match
            
            # ===== STYLE ADJUSTMENTS =====
            overlap = self._literal_overlap(sentence, cand['classical'])
            cand['literal_overlap'] = overlap
            
            literal_boost = 1.0 + 0.6 * overlap  # Increased literal bonus
            interpretive_boost = 1.0
            if overlap < 0.3:
                # Low literal overlap but potentially high semantic similarity
                semantic_gap = max(0.0, cand['raw_similarity'] - overlap)
                interpretive_boost = 1.05 + 0.2 * semantic_gap
            
            if style_mode == "literal":
                score *= literal_boost
                cand['preferred_style'] = "literal"
            elif style_mode == "interpretive":
                score *= interpretive_boost
                cand['preferred_style'] = "interpretive"
            else:  # auto - choose best style per candidate
                literal_score = score * literal_boost
                interpretive_score = score * interpretive_boost
                if interpretive_score > literal_score:
                    score = interpretive_score
                    cand['preferred_style'] = "interpretive"
                else:
                    score = literal_score
                    cand['preferred_style'] = "literal"
            
            cand['final_score'] = score
            
            if score > best_score:
                best_score = score
                best_candidate = cand
        
        return best_candidate, best_score

    def translate_sentence(self, sentence: str, top_k: int = 5, style: str = "auto") -> PrecisionResult:
        """
        Translate a single sentence with high precision.
        
        Args:
            sentence: Classical Chinese sentence
            top_k: Number of candidates to consider
            style: "literal" or "interpretive"
            
        Returns:
            PrecisionResult with translation and confidence
        """
        sentence = sentence.strip()
        
        if not sentence:
            return PrecisionResult(
                input_text=sentence,
                translation="",
                confidence=0.0,
                method="empty"
            )
        
        style_mode = style.lower() if isinstance(style, str) else "auto"
        if style_mode not in {"literal", "interpretive", "auto"}:
            style_mode = "auto"
        
        query_embedding = self._encode(sentence)

        # Aggregate candidates from multiple query variants
        candidate_map: Dict[str, Dict[str, Any]] = {}
        variants = self._generate_query_variants(sentence)

        for variant_text, weight, reason in variants:
            results = self.index.search(variant_text, top_k=max(top_k, 3), quality_boost=True)
            for text, meta, score in results:
                key = f"{meta.get('title','')}_{meta.get('classical','')}"
                adjusted_score = float(score) * weight
                candidate = {
                    'text': text,
                    'score': adjusted_score,
                    'modern': meta.get('modern', ''),
                    'classical': meta.get('classical', ''),
                    'title': meta.get('title', ''),
                    'alignment_score': meta.get('alignment_score', 1.0),
                    'raw_similarity': meta.get('raw_similarity', score),
                    'query_variant': variant_text,
                    'variant_reason': reason,
                    'variant_weight': weight,
                    'base_score': score, # Ensure base_score is available
                    'text_hash': meta.get('text_hash')
                }
                existing = candidate_map.get(key)
                if not existing or adjusted_score > existing['score']:
                    candidate_map[key] = candidate

        candidates = list(candidate_map.values())

        # Filter out candidates that share too few literal characters.
        min_overlap = 0.2 if len(sentence) >= 6 else 0.1
        filtered = [
            cand for cand in candidates
            if self._literal_overlap(sentence, cand['classical']) >= min_overlap
        ]
        if filtered:
            candidates = filtered

        if not candidates:
            return PrecisionResult(
                input_text=sentence,
                translation="[未找到匹配翻译]",
                confidence=0.0,
                processing_notes=["No matches found in index"]
            )
        
        # Select best
        best, confidence = self._select_best_candidate(
            sentence,
            candidates,
            style=style_mode,
            query_embedding=query_embedding,
        )
        confidence = float(min(1.0, max(0.0, confidence)))
        
        if not best:
            return PrecisionResult(
                input_text=sentence,
                translation="[未找到匹配翻译]",
                confidence=0.0,
                candidates=candidates
            )
        
        # Get verification score
        verification = self._verify_translation(sentence, best.get('modern', ''))
        
        notes = []
        translation_text = best.get('modern', '')

        refinement = self._apply_llm_refinement(sentence, best, candidates)
        if refinement and refinement.refined_text:
            translation_text = refinement.refined_text
            confidence = (confidence + refinement.confidence_score) / 2
            if refinement.reasoning:
                notes.append(f"LLM: {refinement.reasoning}")
            else:
                notes.append("LLM refined translation")

        rewrites = self._generate_rewrites(sentence, candidates)

        if best.get('preferred_style'):
            notes.append(f"Style preference: {best['preferred_style']}")
        
        return PrecisionResult(
            input_text=sentence,
            translation=translation_text,
            confidence=confidence,
            semantic_score=best.get('raw_similarity', 0),
            verification_score=verification,
            quality_score=best.get('alignment_score', 0),
            ensemble_score=confidence,
            matched_title=best.get('title', ''),
            matched_classical=best.get('classical', ''),
            raw_similarity=best.get('raw_similarity', 0),
            candidates=candidates,
            method="precision_sentence",
            processing_notes=notes,
            rewrites=rewrites,
        )
    
    def translate(self, text: str, top_k: int = 5, style: str = "auto") -> PrecisionResult:
        """
        Translate classical Chinese text (main entry point).
        
        Handles both single sentences and multi-sentence texts.
        
        Args:
            text: Classical Chinese text
            top_k: Candidates per sentence
            style: "literal" or "interpretive"
            
        Returns:
            PrecisionResult with translation
        """
        text = text.strip()
        
        if not text:
            return PrecisionResult(
                input_text=text,
                translation="",
                confidence=0.0,
                method="empty"
            )
        
        sentences = self._split_sentences(text)
        
        if len(sentences) == 1:
            return self.translate_sentence(text, top_k=top_k, style=style)
        
        # Multi-sentence translation
        translations = []
        all_candidates = []
        total_confidence = 0.0
        total_semantic = 0.0
        total_verification = 0.0
        total_quality = 0.0
        
        notes = [f"Split into {len(sentences)} sentences"]
        
        for sent in sentences:
            result = self.translate_sentence(sent, top_k=top_k, style=style)
            translations.append(result.translation)
            all_candidates.extend(result.candidates[:2])
            total_confidence += result.confidence
            total_semantic += result.semantic_score
            total_verification += result.verification_score
            total_quality += result.quality_score
            if result.processing_notes:
                notes.extend(result.processing_notes)
        
        n = len(sentences)
        combined = '\n'.join(translations)
        
        return PrecisionResult(
            input_text=text,
            translation=combined,
            confidence=total_confidence / n,
            semantic_score=total_semantic / n,
            verification_score=total_verification / n,
            quality_score=total_quality / n,
            candidates=all_candidates,
            method="precision_multi",
            processing_notes=notes
        )
    
    def translate_with_details(self, text: str, style: str = "auto") -> Tuple[PrecisionResult, str]:
        """
        Translate with detailed explanation.
        
        Args:
            text: Classical Chinese text
            style: "literal" or "interpretive"
            
        Returns:
            Tuple of (result, detailed explanation string)
        """
        result = self.translate(text, style=style)

        
        lines = []
        lines.append("=" * 60)
        lines.append("🎯 Precision Translation Analysis")
        lines.append("=" * 60)
        
        lines.append(f"\n📜 Input: {result.input_text}")
        lines.append(f"\n📝 Translation:\n{result.translation}")
        
        lines.append(f"\n📊 Confidence Breakdown:")
        lines.append(f"  • Overall Confidence: {result.confidence:.2%}")
        lines.append(f"  • Semantic Score: {result.semantic_score:.4f}")
        lines.append(f"  • Verification Score: {result.verification_score:.4f}")
        lines.append(f"  • Quality Score: {result.quality_score:.4f}")
        
        if result.matched_title:
            lines.append(f"\n📖 Source: 《{result.matched_title}》")
            lines.append(f"   Matched: {result.matched_classical}")
        
        if result.processing_notes:
            lines.append(f"\n📋 Notes:")
            for note in result.processing_notes:
                lines.append(f"  • {note}")

        if result.rewrites:
            lines.append(f"\n💡 Rewritten Variants:")
            for rewrite in result.rewrites:
                style = rewrite.get("style", "改写")
                translation = rewrite.get("translation", "")
                notes = rewrite.get("notes", "")
                lines.append(f"  [{style}] {translation}")
                if notes:
                    lines.append(f"     └ {notes}")
        
        lines.append("=" * 60)
        
        return result, "\n".join(lines)
    
    def batch_translate(self, texts: List[str], 
                       show_progress: bool = True) -> List[PrecisionResult]:
        """Batch translate multiple texts."""
        results = []
        
        if show_progress:
            from tqdm import tqdm
            texts = tqdm(texts, desc="Translating")
        
        for text in texts:
            results.append(self.translate(text))
        
        return results


def print_result(result: PrecisionResult):
    """Pretty print a precision result."""
    print("\n" + "=" * 60)
    print("🎯 Precision Translation Result")
    print("=" * 60)
    
    print(f"\n【原文】\n{result.input_text}")
    print(f"\n【译文】\n{result.translation}")
    
    print(f"\n【置信度】{result.confidence:.2%}")
    
    # Visual score bars
    print("\n【评分详情】")
    scores = [
        ("语义匹配", result.semantic_score),
        ("验证分数", result.verification_score),
        ("数据质量", result.quality_score),
    ]
    
    for name, score in scores:
        bar = "█" * int(score * 20)
        print(f"  {name}: {bar} {score:.4f}")
    
    if result.matched_title:
        print(f"\n【来源】《{result.matched_title}》")

    if result.rewrites:
        print("\n【生成式改写】")
        for rewrite in result.rewrites:
            style = rewrite.get("style", "改写")
            translation = rewrite.get("translation", "")
            notes = rewrite.get("notes", "")
            print(f"  [{style}] {translation}")
            if notes:
                print(f"     └ {notes}")
    
    print("=" * 60)


def main():
    """Test the precision translator."""
    print("=" * 60)
    print("Precision Classical Chinese Translator Test")
    print("=" * 60)
    
    translator = PrecisionTranslator()
    
    test_cases = [
        "梳洗罢，独倚望江楼。",
        "过尽千帆皆不是，斜晖脉脉水悠悠。",
        "不患寡而患不均，不患贫而患不安。",
        "肠断白蘋洲。",
    ]
    
    print("\n🧪 Running precision tests...")
    
    for text in test_cases:
        result, details = translator.translate_with_details(text)
        print(details)


if __name__ == "__main__":
    main()

