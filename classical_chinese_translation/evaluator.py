#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Chinese Translation Evaluator

This module provides evaluation metrics and analysis tools for
assessing translation quality and system performance.

Metrics included:
- Retrieval accuracy (exact match, partial match)
- Similarity distribution analysis
- Cross-validation evaluation
- Per-source performance breakdown
"""
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

from translator import ClassicalChineseTranslator, TranslationResult
from data_processor import DataProcessor, SentencePair


@dataclass
class EvaluationMetrics:
    """
    Container for evaluation metrics.
    
    Attributes:
        total_samples: Total number of test samples
        exact_matches: Number of exact matches (sim >= 0.95)
        high_confidence_matches: Number of high confidence matches (sim >= 0.85)
        medium_confidence_matches: Number of medium confidence matches (sim >= 0.70)
        low_confidence_matches: Number of low confidence matches (sim >= 0.50)
        no_match: Number of samples with no good match
        avg_similarity: Average similarity score
        avg_confidence: Average confidence score
        similarity_distribution: Distribution of similarity scores
    """
    total_samples: int = 0
    exact_matches: int = 0
    high_confidence_matches: int = 0
    medium_confidence_matches: int = 0
    low_confidence_matches: int = 0
    no_match: int = 0
    avg_similarity: float = 0.0
    avg_confidence: float = 0.0
    similarity_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.similarity_distribution is None:
            self.similarity_distribution = {}
    
    def get_accuracy_at_threshold(self, threshold: float) -> float:
        """Calculate accuracy at a given similarity threshold."""
        if threshold >= 0.95:
            matches = self.exact_matches
        elif threshold >= 0.85:
            matches = self.exact_matches + self.high_confidence_matches
        elif threshold >= 0.70:
            matches = (self.exact_matches + self.high_confidence_matches + 
                      self.medium_confidence_matches)
        elif threshold >= 0.50:
            matches = (self.exact_matches + self.high_confidence_matches + 
                      self.medium_confidence_matches + self.low_confidence_matches)
        else:
            matches = self.total_samples - self.no_match
        
        return matches / self.total_samples if self.total_samples > 0 else 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_samples': self.total_samples,
            'exact_matches': self.exact_matches,
            'high_confidence_matches': self.high_confidence_matches,
            'medium_confidence_matches': self.medium_confidence_matches,
            'low_confidence_matches': self.low_confidence_matches,
            'no_match': self.no_match,
            'avg_similarity': self.avg_similarity,
            'avg_confidence': self.avg_confidence,
            'accuracy_at_0.95': self.get_accuracy_at_threshold(0.95),
            'accuracy_at_0.85': self.get_accuracy_at_threshold(0.85),
            'accuracy_at_0.70': self.get_accuracy_at_threshold(0.70),
            'accuracy_at_0.50': self.get_accuracy_at_threshold(0.50),
        }


class TranslationEvaluator:
    """
    Evaluator for classical Chinese translation system.
    
    Provides comprehensive evaluation including:
    - Self-retrieval accuracy (can the system find the correct translation?)
    - Similarity distribution analysis
    - Error analysis
    """
    
    def __init__(self, translator: ClassicalChineseTranslator = None):
        """
        Initialize the evaluator.
        
        Args:
            translator: Translator instance (will create one if not provided)
        """
        self.translator = translator or ClassicalChineseTranslator()
        self.data_processor = DataProcessor()
    
    def evaluate_self_retrieval(self, sample_size: int = None, 
                                verbose: bool = True) -> EvaluationMetrics:
        """
        Evaluate self-retrieval accuracy.
        
        Tests whether the system can retrieve the correct translation
        when queried with texts from the training set.
        
        Args:
            sample_size: Number of samples to evaluate (None for all)
            verbose: Whether to show progress
            
        Returns:
            EvaluationMetrics object
        """
        # Load test data
        self.data_processor.load_all_data()
        self.data_processor.create_sentence_pairs()
        
        test_pairs = self.data_processor.sentence_pairs
        
        if sample_size and sample_size < len(test_pairs):
            # Random sample
            import random
            test_pairs = random.sample(test_pairs, sample_size)
        
        metrics = EvaluationMetrics()
        metrics.total_samples = len(test_pairs)
        
        similarities = []
        confidences = []
        
        iterator = tqdm(test_pairs, desc="Evaluating") if verbose else test_pairs
        
        for pair in iterator:
            # Translate the classical text
            result = self.translator.translate(pair.classical, mode="sentence")
            
            # Get the best similarity
            best_sim = result.source_matches[0]['similarity'] if result.source_matches else 0.0
            
            similarities.append(best_sim)
            confidences.append(result.confidence)
            
            # Categorize by similarity
            if best_sim >= 0.95:
                metrics.exact_matches += 1
            elif best_sim >= 0.85:
                metrics.high_confidence_matches += 1
            elif best_sim >= 0.70:
                metrics.medium_confidence_matches += 1
            elif best_sim >= 0.50:
                metrics.low_confidence_matches += 1
            else:
                metrics.no_match += 1
        
        # Calculate averages
        metrics.avg_similarity = np.mean(similarities) if similarities else 0.0
        metrics.avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Build similarity distribution
        bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        hist, _ = np.histogram(similarities, bins=bins)
        metrics.similarity_distribution = {
            f'{bins[i]:.2f}-{bins[i+1]:.2f}': int(hist[i])
            for i in range(len(hist))
        }
        
        return metrics
    
    def analyze_errors(self, num_samples: int = 20) -> List[dict]:
        """
        Analyze translation errors.
        
        Identifies and categorizes cases where translation quality is low.
        
        Args:
            num_samples: Number of error samples to analyze
            
        Returns:
            List of error analysis results
        """
        self.data_processor.load_all_data()
        self.data_processor.create_sentence_pairs()
        
        errors = []
        
        for pair in tqdm(self.data_processor.sentence_pairs, desc="Analyzing errors"):
            result = self.translator.translate(pair.classical, mode="sentence")
            
            if result.confidence < 0.7:
                errors.append({
                    'input': pair.classical,
                    'expected_modern': pair.modern,
                    'predicted_modern': result.translation,
                    'confidence': result.confidence,
                    'similarity': result.source_matches[0]['similarity'] if result.source_matches else 0.0,
                    'title': pair.title,
                    'method': result.method
                })
            
            if len(errors) >= num_samples:
                break
        
        return errors
    
    def evaluate_by_source(self) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate performance by source (poem/prose title).
        
        Returns:
            Dictionary mapping source titles to their metrics
        """
        self.data_processor.load_all_data()
        self.data_processor.create_sentence_pairs()
        
        # Group by title
        by_title = defaultdict(list)
        for pair in self.data_processor.sentence_pairs:
            by_title[pair.title].append(pair)
        
        results = {}
        
        for title, pairs in tqdm(by_title.items(), desc="Evaluating by source"):
            metrics = EvaluationMetrics()
            metrics.total_samples = len(pairs)
            
            similarities = []
            
            for pair in pairs:
                result = self.translator.translate(pair.classical, mode="sentence")
                best_sim = result.source_matches[0]['similarity'] if result.source_matches else 0.0
                similarities.append(best_sim)
                
                if best_sim >= 0.95:
                    metrics.exact_matches += 1
                elif best_sim >= 0.85:
                    metrics.high_confidence_matches += 1
                elif best_sim >= 0.70:
                    metrics.medium_confidence_matches += 1
                elif best_sim >= 0.50:
                    metrics.low_confidence_matches += 1
                else:
                    metrics.no_match += 1
            
            metrics.avg_similarity = np.mean(similarities) if similarities else 0.0
            results[title] = metrics
        
        return results


def print_evaluation_report(metrics: EvaluationMetrics, title: str = "Evaluation Report"):
    """
    Print formatted evaluation report.
    
    Args:
        metrics: EvaluationMetrics object
        title: Report title
    """
    print("\n" + "=" * 70)
    print(f"📊 {title}")
    print("=" * 70)
    
    print(f"\n📈 Overall Statistics:")
    print(f"  Total samples: {metrics.total_samples}")
    print(f"  Average similarity: {metrics.avg_similarity:.4f}")
    print(f"  Average confidence: {metrics.avg_confidence:.4f}")
    
    print(f"\n🎯 Match Distribution:")
    print(f"  Exact matches (≥0.95):       {metrics.exact_matches:5d} ({metrics.exact_matches/metrics.total_samples*100:6.2f}%)")
    print(f"  High confidence (0.85-0.95): {metrics.high_confidence_matches:5d} ({metrics.high_confidence_matches/metrics.total_samples*100:6.2f}%)")
    print(f"  Medium confidence (0.70-0.85): {metrics.medium_confidence_matches:5d} ({metrics.medium_confidence_matches/metrics.total_samples*100:6.2f}%)")
    print(f"  Low confidence (0.50-0.70):  {metrics.low_confidence_matches:5d} ({metrics.low_confidence_matches/metrics.total_samples*100:6.2f}%)")
    print(f"  No match (<0.50):            {metrics.no_match:5d} ({metrics.no_match/metrics.total_samples*100:6.2f}%)")
    
    print(f"\n📉 Accuracy at Thresholds:")
    print(f"  Accuracy @ 0.95: {metrics.get_accuracy_at_threshold(0.95)*100:6.2f}%")
    print(f"  Accuracy @ 0.85: {metrics.get_accuracy_at_threshold(0.85)*100:6.2f}%")
    print(f"  Accuracy @ 0.70: {metrics.get_accuracy_at_threshold(0.70)*100:6.2f}%")
    print(f"  Accuracy @ 0.50: {metrics.get_accuracy_at_threshold(0.50)*100:6.2f}%")
    
    if metrics.similarity_distribution:
        print(f"\n📊 Similarity Distribution:")
        for range_str, count in sorted(metrics.similarity_distribution.items()):
            bar = "█" * int(count / metrics.total_samples * 40)
            print(f"  {range_str}: {bar} {count}")
    
    print("=" * 70)


def main():
    """Run evaluation."""
    print("=" * 70)
    print("Classical Chinese Translation System Evaluation")
    print("=" * 70)
    
    # Initialize evaluator
    evaluator = TranslationEvaluator()
    
    # Run self-retrieval evaluation
    print("\n🔍 Running self-retrieval evaluation...")
    metrics = evaluator.evaluate_self_retrieval(sample_size=100)
    print_evaluation_report(metrics, "Self-Retrieval Evaluation (100 samples)")
    
    # Analyze errors
    print("\n🔍 Analyzing low-confidence translations...")
    errors = evaluator.analyze_errors(num_samples=5)
    
    if errors:
        print("\n📋 Sample Error Cases:")
        print("-" * 70)
        for i, error in enumerate(errors[:5], 1):
            print(f"\n{i}. Title: {error['title']}")
            print(f"   Input: {error['input']}")
            print(f"   Expected: {error['expected_modern'][:50]}...")
            print(f"   Got: {error['predicted_modern'][:50]}...")
            print(f"   Confidence: {error['confidence']:.2%}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

