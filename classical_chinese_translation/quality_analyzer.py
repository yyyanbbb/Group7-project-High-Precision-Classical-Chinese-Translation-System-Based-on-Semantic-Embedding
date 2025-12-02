#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translation Quality Analyzer

This module provides comprehensive quality analysis for classical Chinese
translations, including:

1. Similarity Pattern Analysis (from 02_similarity_calculation.py)
2. Clustering Quality Assessment (from 04_text_clustering_visualization.py)
3. Cross-Validation Metrics
4. Error Pattern Detection
5. Coverage Analysis
6. Confidence Distribution Analysis

Innovations:
- Multi-dimensional quality scoring
- Automatic error categorization
- Translation consistency checking
- Vocabulary coverage analysis
"""
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from data_processor import DataProcessor
from index_builder import EmbeddingIndex, SENTENCE_INDEX_FILE
from translator import ClassicalChineseTranslator
from project_config import load_model, INDEX_DIR


@dataclass
class QualityMetrics:
    """
    Comprehensive quality metrics container.
    
    Attributes:
        overall_score: Overall quality score (0-100)
        retrieval_accuracy: Self-retrieval accuracy
        avg_similarity: Average similarity score
        similarity_std: Standard deviation of similarities
        cluster_quality: Cluster separation quality
        coverage: Vocabulary/phrase coverage
        consistency: Translation consistency score
        error_rate: Error rate estimate
        breakdown: Detailed score breakdown
    """
    overall_score: float = 0.0
    retrieval_accuracy: float = 0.0
    avg_similarity: float = 0.0
    similarity_std: float = 0.0
    cluster_quality: float = 0.0
    coverage: float = 0.0
    consistency: float = 0.0
    error_rate: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    
    def to_report(self) -> str:
        """Generate a formatted report string."""
        report = []
        report.append("=" * 60)
        report.append("📊 Translation Quality Report")
        report.append("=" * 60)
        report.append(f"\n🎯 Overall Score: {self.overall_score:.1f}/100")
        report.append(f"\n📈 Detailed Metrics:")
        report.append(f"  • Retrieval Accuracy: {self.retrieval_accuracy:.2%}")
        report.append(f"  • Average Similarity: {self.avg_similarity:.4f}")
        report.append(f"  • Similarity Std Dev: {self.similarity_std:.4f}")
        report.append(f"  • Cluster Quality: {self.cluster_quality:.4f}")
        report.append(f"  • Coverage: {self.coverage:.2%}")
        report.append(f"  • Consistency: {self.consistency:.2%}")
        report.append(f"  • Error Rate: {self.error_rate:.2%}")
        
        if self.breakdown:
            report.append(f"\n📋 Score Breakdown:")
            for key, value in self.breakdown.items():
                report.append(f"  • {key}: {value:.2f}")
        
        report.append("=" * 60)
        return "\n".join(report)


@dataclass  
class ErrorAnalysis:
    """
    Error analysis result container.
    
    Attributes:
        total_errors: Total number of errors found
        error_categories: Dictionary of error types and counts
        examples: List of error examples
        suggestions: List of improvement suggestions
    """
    total_errors: int = 0
    error_categories: Dict[str, int] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class QualityAnalyzer:
    """
    Comprehensive quality analyzer for classical Chinese translation system.
    
    This analyzer evaluates the translation system from multiple angles:
    
    1. Retrieval Quality:
       - Self-retrieval accuracy
       - Top-k accuracy
       - Mean reciprocal rank
       
    2. Embedding Quality:
       - Clustering metrics
       - Similarity distribution
       - Embedding space coherence
       
    3. Translation Quality:
       - Consistency checking
       - Error pattern detection
       - Coverage analysis
       
    4. System Quality:
       - Response time
       - Memory efficiency
       - Scalability metrics
    """
    
    def __init__(self, translator: ClassicalChineseTranslator = None):
        """
        Initialize the quality analyzer.
        
        Args:
            translator: Translator instance (creates one if not provided)
        """
        self.translator = translator or ClassicalChineseTranslator()
        self.data_processor = DataProcessor()
        self.index: Optional[EmbeddingIndex] = None
        
        self._load_data()
    
    def _load_data(self):
        """Load necessary data for analysis."""
        self.data_processor.load_all_data()
        self.data_processor.create_sentence_pairs()
        
        self.index = EmbeddingIndex()
        self.index.load(SENTENCE_INDEX_FILE)
    
    def analyze_retrieval_quality(self, sample_size: int = 100,
                                  verbose: bool = True) -> Dict[str, float]:
        """
        Analyze retrieval quality using self-retrieval test.
        
        Tests whether the system can correctly retrieve entries
        that are in the index.
        
        Args:
            sample_size: Number of samples to test
            verbose: Whether to show progress
            
        Returns:
            Dictionary of retrieval metrics
        """
        test_pairs = self.data_processor.sentence_pairs[:sample_size]
        
        metrics = {
            'top1_accuracy': 0,
            'top3_accuracy': 0,
            'top5_accuracy': 0,
            'mrr': 0,  # Mean Reciprocal Rank
            'avg_similarity': 0,
            'exact_match': 0
        }
        
        similarities = []
        
        iterator = tqdm(test_pairs, desc="Testing retrieval") if verbose else test_pairs
        
        for pair in iterator:
            results = self.index.search(pair.classical, top_k=5)
            
            if not results:
                continue
            
            # Check top-k accuracy
            for i, (_, meta, sim) in enumerate(results):
                if meta.get('classical', '') == pair.classical:
                    metrics['mrr'] += 1.0 / (i + 1)
                    if i == 0:
                        metrics['top1_accuracy'] += 1
                        metrics['exact_match'] += 1
                    if i < 3:
                        metrics['top3_accuracy'] += 1
                    if i < 5:
                        metrics['top5_accuracy'] += 1
                    break
            
            # Record best similarity
            if results:
                similarities.append(results[0][2])
        
        n = len(test_pairs)
        metrics['top1_accuracy'] /= n
        metrics['top3_accuracy'] /= n
        metrics['top5_accuracy'] /= n
        metrics['mrr'] /= n
        metrics['exact_match'] /= n
        metrics['avg_similarity'] = np.mean(similarities) if similarities else 0
        metrics['similarity_std'] = np.std(similarities) if similarities else 0
        
        return metrics
    
    def analyze_embedding_space(self, n_clusters: int = 10,
                               sample_size: int = 500) -> Dict[str, float]:
        """
        Analyze embedding space quality using clustering.
        
        Following pattern from 04_text_clustering_visualization.py.
        
        Args:
            n_clusters: Number of clusters for K-means
            sample_size: Number of samples to analyze
            
        Returns:
            Dictionary of embedding quality metrics
        """
        # Sample embeddings
        n = min(sample_size, len(self.index.embeddings))
        indices = np.random.choice(len(self.index.embeddings), n, replace=False)
        embeddings = self.index.embeddings[indices]
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / np.where(norms == 0, 1, norms)
        
        metrics = {}
        
        # Clustering analysis
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_norm)
            
            # Silhouette score (cluster separation)
            if len(set(labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(embeddings_norm, labels)
            else:
                metrics['silhouette_score'] = 0.0
            
            # Inertia (compactness)
            metrics['inertia'] = kmeans.inertia_ / n
        except Exception as e:
            print(f"⚠️ Clustering analysis failed: {e}")
            metrics['silhouette_score'] = 0.0
            metrics['inertia'] = 0.0
        
        # Embedding statistics
        metrics['avg_norm'] = np.mean(norms)
        metrics['norm_std'] = np.std(norms)
        
        # Pairwise similarity distribution
        sample_sims = []
        for i in range(min(100, n)):
            for j in range(i + 1, min(100, n)):
                sim = np.dot(embeddings_norm[i], embeddings_norm[j])
                sample_sims.append(sim)
        
        if sample_sims:
            metrics['avg_pairwise_sim'] = np.mean(sample_sims)
            metrics['pairwise_sim_std'] = np.std(sample_sims)
        
        return metrics
    
    def analyze_translation_consistency(self, sample_size: int = 50) -> Dict[str, float]:
        """
        Analyze translation consistency.
        
        Checks if similar inputs produce similar outputs.
        
        Args:
            sample_size: Number of pairs to analyze
            
        Returns:
            Dictionary of consistency metrics
        """
        pairs = self.data_processor.sentence_pairs[:sample_size]
        
        consistency_scores = []
        
        for i in range(len(pairs) - 1):
            pair1 = pairs[i]
            pair2 = pairs[i + 1]
            
            # Only compare if from same source
            if pair1.title != pair2.title:
                continue
            
            # Translate both
            result1 = self.translator.translate(pair1.classical, mode="sentence")
            result2 = self.translator.translate(pair2.classical, mode="sentence")
            
            # Check consistency
            if result1.source_matches and result2.source_matches:
                same_source = (result1.source_matches[0].get('title') == 
                              result2.source_matches[0].get('title'))
                consistency_scores.append(1.0 if same_source else 0.0)
        
        return {
            'consistency_rate': np.mean(consistency_scores) if consistency_scores else 0.0,
            'samples_analyzed': len(consistency_scores)
        }
    
    def analyze_coverage(self) -> Dict[str, Any]:
        """
        Analyze vocabulary and phrase coverage.
        
        Returns:
            Dictionary of coverage metrics
        """
        # Collect all classical characters
        all_chars = Counter()
        all_phrases = Counter()
        
        for pair in self.data_processor.sentence_pairs:
            text = pair.classical.replace(' ', '').replace('\n', '')
            
            # Character coverage
            for char in text:
                all_chars[char] += 1
            
            # Phrase coverage (2-4 grams)
            for n in range(2, 5):
                for i in range(len(text) - n + 1):
                    phrase = text[i:i+n]
                    all_phrases[phrase] += 1
        
        return {
            'unique_chars': len(all_chars),
            'total_chars': sum(all_chars.values()),
            'unique_phrases': len(all_phrases),
            'most_common_chars': all_chars.most_common(20),
            'most_common_phrases': all_phrases.most_common(20),
            'char_coverage_rate': len(all_chars) / 5000  # Approximate common char count
        }
    
    def detect_errors(self, sample_size: int = 100,
                     confidence_threshold: float = 0.6) -> ErrorAnalysis:
        """
        Detect and categorize translation errors.
        
        Args:
            sample_size: Number of samples to analyze
            confidence_threshold: Threshold for flagging low confidence
            
        Returns:
            ErrorAnalysis object
        """
        errors = ErrorAnalysis()
        
        test_pairs = self.data_processor.sentence_pairs[:sample_size]
        
        for pair in tqdm(test_pairs, desc="Detecting errors"):
            result = self.translator.translate(pair.classical, mode="sentence")
            
            # Categorize potential issues
            if result.confidence < confidence_threshold:
                errors.total_errors += 1
                
                # Determine error type
                if result.confidence < 0.3:
                    error_type = "no_match"
                    errors.error_categories["No Match"] = errors.error_categories.get("No Match", 0) + 1
                elif result.confidence < 0.5:
                    error_type = "low_confidence"
                    errors.error_categories["Low Confidence"] = errors.error_categories.get("Low Confidence", 0) + 1
                else:
                    error_type = "medium_confidence"
                    errors.error_categories["Medium Confidence"] = errors.error_categories.get("Medium Confidence", 0) + 1
                
                # Store example
                if len(errors.examples) < 10:
                    errors.examples.append({
                        'input': pair.classical,
                        'expected': pair.modern[:100],
                        'got': result.translation[:100],
                        'confidence': result.confidence,
                        'error_type': error_type,
                        'title': pair.title
                    })
        
        # Generate suggestions
        if errors.error_categories.get("No Match", 0) > sample_size * 0.1:
            errors.suggestions.append("Consider expanding the training data corpus")
        
        if errors.error_categories.get("Low Confidence", 0) > sample_size * 0.2:
            errors.suggestions.append("Consider fine-tuning similarity thresholds")
        
        if errors.total_errors > sample_size * 0.3:
            errors.suggestions.append("Consider improving sentence alignment in preprocessing")
        
        return errors
    
    def compute_overall_quality(self, verbose: bool = True) -> QualityMetrics:
        """
        Compute comprehensive quality metrics.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            QualityMetrics object
        """
        if verbose:
            print("\n" + "=" * 60)
            print("🔍 Computing Comprehensive Quality Metrics")
            print("=" * 60)
        
        metrics = QualityMetrics()
        
        # 1. Retrieval quality
        if verbose:
            print("\n1️⃣ Analyzing retrieval quality...")
        retrieval = self.analyze_retrieval_quality(sample_size=100, verbose=verbose)
        metrics.retrieval_accuracy = retrieval['top1_accuracy']
        metrics.avg_similarity = retrieval['avg_similarity']
        metrics.similarity_std = retrieval['similarity_std']
        
        # 2. Embedding quality
        if verbose:
            print("\n2️⃣ Analyzing embedding space...")
        embedding = self.analyze_embedding_space(sample_size=300)
        metrics.cluster_quality = embedding.get('silhouette_score', 0)
        
        # 3. Consistency
        if verbose:
            print("\n3️⃣ Analyzing consistency...")
        consistency = self.analyze_translation_consistency(sample_size=30)
        metrics.consistency = consistency['consistency_rate']
        
        # 4. Coverage
        if verbose:
            print("\n4️⃣ Analyzing coverage...")
        coverage = self.analyze_coverage()
        metrics.coverage = coverage['char_coverage_rate']
        
        # 5. Error detection
        if verbose:
            print("\n5️⃣ Detecting errors...")
        errors = self.detect_errors(sample_size=50)
        metrics.error_rate = errors.total_errors / 50
        
        # Compute overall score (weighted combination)
        metrics.breakdown = {
            'Retrieval (40%)': metrics.retrieval_accuracy * 40,
            'Similarity (20%)': min(1.0, metrics.avg_similarity) * 20,
            'Cluster Quality (15%)': max(0, metrics.cluster_quality + 0.5) * 15,
            'Consistency (15%)': metrics.consistency * 15,
            'Error Rate (10%)': (1 - metrics.error_rate) * 10
        }
        
        metrics.overall_score = sum(metrics.breakdown.values())
        
        return metrics
    
    def generate_full_report(self) -> str:
        """
        Generate a comprehensive quality report.
        
        Returns:
            Formatted report string
        """
        print("\n🔄 Generating full quality report...")
        
        # Compute all metrics
        metrics = self.compute_overall_quality(verbose=True)
        
        # Get error analysis
        errors = self.detect_errors(sample_size=50)
        
        # Build report
        report = []
        report.append(metrics.to_report())
        
        # Add error analysis
        report.append("\n📋 Error Analysis:")
        report.append(f"  Total Errors: {errors.total_errors}")
        for category, count in errors.error_categories.items():
            report.append(f"  • {category}: {count}")
        
        if errors.examples:
            report.append("\n📝 Error Examples:")
            for i, ex in enumerate(errors.examples[:3], 1):
                report.append(f"  {i}. Input: {ex['input']}")
                report.append(f"     Confidence: {ex['confidence']:.2%}")
                report.append(f"     Type: {ex['error_type']}")
        
        if errors.suggestions:
            report.append("\n💡 Improvement Suggestions:")
            for sug in errors.suggestions:
                report.append(f"  • {sug}")
        
        return "\n".join(report)


def main():
    """Run quality analysis."""
    print("=" * 60)
    print("Classical Chinese Translation Quality Analyzer")
    print("=" * 60)
    
    analyzer = QualityAnalyzer()
    report = analyzer.generate_full_report()
    print(report)


if __name__ == "__main__":
    main()

