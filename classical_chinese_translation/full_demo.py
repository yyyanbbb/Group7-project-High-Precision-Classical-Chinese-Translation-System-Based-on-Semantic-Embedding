#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Demo: Classical Chinese Translation System

This comprehensive demo showcases all features of the translation system,
integrating techniques from 00-04 examples plus innovative enhancements.

Features demonstrated:
1. Basic Translation (from 00_input_output.py pattern)
2. Multi-granularity Analysis (from 01_basic_usage.py)
3. Similarity Computation (from 02_similarity_calculation.py)
4. Semantic Search (from 03_semantic_search.py)
5. Clustering & Visualization (from 04_text_clustering_visualization.py)
6. Advanced Translation with Ensemble Methods
7. Quality Analysis and Reporting
"""
import os
import sys
import argparse
import numpy as np

# Ensure module imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def demo_basic_translation():
    """
    Demo 1: Basic Translation Pipeline
    
    Shows the fundamental translation workflow.
    """
    from translator import ClassicalChineseTranslator, print_translation_result
    
    print("\n" + "=" * 70)
    print("📚 Demo 1: Basic Translation Pipeline")
    print("    (Similar to 00_input_output.py)")
    print("=" * 70)
    
    translator = ClassicalChineseTranslator()
    
    test_texts = [
        "梳洗罢，独倚望江楼。",
        "过尽千帆皆不是",
        "不患寡而患不均",
    ]
    
    print("\n🔤 Translating sample classical Chinese texts:")
    for text in test_texts:
        print(f"\n📜 Input: {text}")
        result = translator.translate(text)
        print(f"📝 Translation: {result.translation[:100]}...")
        print(f"📊 Confidence: {result.confidence:.2%}")


def demo_multi_granularity():
    """
    Demo 2: Multi-granularity Analysis
    
    Shows how text length affects embedding quality.
    """
    from translator import ClassicalChineseTranslator
    
    print("\n" + "=" * 70)
    print("📐 Demo 2: Multi-granularity Analysis")
    print("    (Similar to 01_basic_usage.py)")
    print("=" * 70)
    
    translator = ClassicalChineseTranslator()
    
    # Different granularities
    granularities = {
        'Character': '望',
        'Word': '望江楼',
        'Phrase': '独倚望江楼',
        'Clause': '梳洗罢，独倚望江楼',
        'Sentence': '梳洗罢，独倚望江楼。过尽千帆皆不是，斜晖脉脉水悠悠。'
    }
    
    print("\n🔍 Comparing search results across granularities:")
    print("-" * 60)
    
    for level, text in granularities.items():
        result = translator.translate(text, mode="sentence")
        sim = result.source_matches[0]['similarity'] if result.source_matches else 0
        bar = "█" * int(sim * 30)
        print(f"\n{level:12}: {text[:30]}...")
        print(f"             Similarity: {bar} {sim:.4f}")


def demo_similarity_matrix():
    """
    Demo 3: Similarity Matrix Computation
    
    Shows pairwise similarities between texts.
    """
    from index_builder import EmbeddingIndex, SENTENCE_INDEX_FILE
    
    print("\n" + "=" * 70)
    print("📊 Demo 3: Similarity Matrix Analysis")
    print("    (Similar to 02_similarity_calculation.py)")
    print("=" * 70)
    
    index = EmbeddingIndex()
    index.load(SENTENCE_INDEX_FILE)
    
    # Sample texts for comparison
    sample_texts = [
        "梳洗罢，独倚望江楼。",
        "过尽千帆皆不是",
        "斜晖脉脉水悠悠",
        "不患寡而患不均",
        "白日依山尽",
    ]
    
    print("\n🔢 Computing similarity matrix:")
    sim_matrix = index.get_similarity_matrix(sample_texts)
    
    # Print matrix header
    print("\n" + " " * 20, end="")
    for i in range(len(sample_texts)):
        print(f"Text{i+1:2d}  ", end="")
    print()
    
    # Print matrix rows
    for i, text in enumerate(sample_texts):
        text_short = text[:12] + "..." if len(text) > 12 else text
        print(f"Text{i+1:2d}: {text_short:15}", end="")
        for j in range(len(sample_texts)):
            print(f"{sim_matrix[i][j]:6.3f}  ", end="")
        print()
    
    # Highlight findings
    print("\n📈 Key Observations:")
    print(f"  • Same-poem texts (1-3) have higher similarity: {sim_matrix[0][1]:.3f}, {sim_matrix[1][2]:.3f}")
    print(f"  • Cross-poem similarity is lower: {sim_matrix[0][3]:.3f}")


def demo_semantic_search():
    """
    Demo 4: Semantic Search Engine
    
    Shows search functionality.
    """
    from translator import ClassicalChineseTranslator
    
    print("\n" + "=" * 70)
    print("🔍 Demo 4: Semantic Search")
    print("    (Similar to 03_semantic_search.py)")
    print("=" * 70)
    
    translator = ClassicalChineseTranslator()
    
    queries = [
        "思念远方的人",
        "江边等待",
        "贫富不均",
    ]
    
    print("\n📡 Semantic search for modern concepts in classical texts:")
    
    for query in queries:
        print(f"\n🔎 Query: '{query}'")
        print("-" * 40)
        
        similar = translator.get_similar_texts(query, top_k=3)
        
        for i, item in enumerate(similar, 1):
            sim = item['similarity']
            bar = "█" * int(sim * 25)
            print(f"  {i}. [{bar}] {sim:.4f}")
            print(f"     《{item['title']}》: {item['classical'][:30]}...")


def demo_clustering():
    """
    Demo 5: Text Clustering Analysis
    
    Shows clustering of classical texts.
    """
    from sklearn.cluster import KMeans
    from index_builder import EmbeddingIndex, SENTENCE_INDEX_FILE
    
    print("\n" + "=" * 70)
    print("🎯 Demo 5: Text Clustering")
    print("    (Similar to 04_text_clustering_visualization.py)")
    print("=" * 70)
    
    index = EmbeddingIndex()
    index.load(SENTENCE_INDEX_FILE)
    
    # Sample for clustering
    n_samples = min(200, len(index.embeddings))
    indices = np.random.choice(len(index.embeddings), n_samples, replace=False)
    embeddings = index.embeddings[indices]
    texts = [index.texts[i] for i in indices]
    metadata = [index.metadata[i] for i in indices]
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / np.where(norms == 0, 1, norms)
    
    # Cluster
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_norm)
    
    print(f"\n📊 Clustering {n_samples} texts into {n_clusters} groups:")
    print("-" * 50)
    
    # Analyze clusters
    from collections import Counter
    
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
        cluster_titles = [metadata[i].get('title', '') for i in cluster_indices]
        title_counts = Counter(cluster_titles).most_common(3)
        
        print(f"\n🏷️ Cluster {cluster_id + 1} ({len(cluster_indices)} texts):")
        print(f"   Top sources: {', '.join([t[0][:15] for t in title_counts])}")
        
        # Show sample text
        if cluster_indices:
            sample_idx = cluster_indices[0]
            print(f"   Sample: {texts[sample_idx][:40]}...")


def demo_advanced_translation():
    """
    Demo 6: Advanced Translation with Ensemble
    
    Shows the innovative ensemble translation approach.
    """
    from advanced_translator import AdvancedTranslator, print_advanced_result
    
    print("\n" + "=" * 70)
    print("🚀 Demo 6: Advanced Ensemble Translation")
    print("    (Innovative: Multi-strategy + Bi-directional Verification)")
    print("=" * 70)
    
    translator = AdvancedTranslator()
    
    test_texts = [
        "梳洗罢，独倚望江楼。",
        "过尽千帆皆不是，斜晖脉脉水悠悠。",
    ]
    
    print("\n⚡ Advanced translation with explanation:")
    
    for text in test_texts:
        result, explanation = translator.translate_with_explanation(text)
        print_advanced_result(result)


def demo_quality_analysis():
    """
    Demo 7: Quality Analysis
    
    Shows comprehensive quality metrics.
    """
    from quality_analyzer import QualityAnalyzer
    
    print("\n" + "=" * 70)
    print("📈 Demo 7: Translation Quality Analysis")
    print("    (Innovative: Multi-dimensional Quality Scoring)")
    print("=" * 70)
    
    analyzer = QualityAnalyzer()
    
    # Quick analysis (limited samples for demo)
    print("\n🔬 Running quick quality analysis...")
    
    # Retrieval quality
    print("\n1️⃣ Retrieval Quality:")
    retrieval = analyzer.analyze_retrieval_quality(sample_size=30, verbose=False)
    print(f"   • Top-1 Accuracy: {retrieval['top1_accuracy']:.2%}")
    print(f"   • Average Similarity: {retrieval['avg_similarity']:.4f}")
    print(f"   • MRR: {retrieval['mrr']:.4f}")
    
    # Embedding quality
    print("\n2️⃣ Embedding Space Quality:")
    embedding = analyzer.analyze_embedding_space(sample_size=100)
    print(f"   • Cluster Separation: {embedding.get('silhouette_score', 0):.4f}")
    print(f"   • Avg Pairwise Similarity: {embedding.get('avg_pairwise_sim', 0):.4f}")
    
    # Coverage
    print("\n3️⃣ Coverage Analysis:")
    coverage = analyzer.analyze_coverage()
    print(f"   • Unique Characters: {coverage['unique_chars']}")
    print(f"   • Unique Phrases: {coverage['unique_phrases']}")


def run_interactive_mode():
    """
    Interactive translation mode.
    """
    from advanced_translator import AdvancedTranslator, print_advanced_result
    
    print("\n" + "=" * 70)
    print("🎮 Interactive Translation Mode")
    print("=" * 70)
    print("\nCommands:")
    print("  • Enter classical Chinese text to translate")
    print("  • Type 'quit' or 'exit' to exit")
    print("=" * 70)
    
    translator = AdvancedTranslator()
    
    while True:
        try:
            user_input = input("\n📜 Enter classical Chinese: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            result = translator.translate(user_input)
            print_advanced_result(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Classical Chinese Translation System - Full Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--demo',
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help='Run specific demo (1-7), or omit to run all'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run interactive translation mode'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🏛️  Classical Chinese Translation System")
    print("    Full Feature Demonstration")
    print("    Based on Qwen3-Embedding-4B Model")
    print("=" * 70)
    
    if args.interactive:
        run_interactive_mode()
        return
    
    demos = {
        1: demo_basic_translation,
        2: demo_multi_granularity,
        3: demo_similarity_matrix,
        4: demo_semantic_search,
        5: demo_clustering,
        6: demo_advanced_translation,
        7: demo_quality_analysis,
    }
    
    if args.demo:
        demos[args.demo]()
    else:
        # Run all demos
        for demo_num, demo_func in demos.items():
            try:
                demo_func()
            except Exception as e:
                print(f"⚠️ Demo {demo_num} failed: {e}")
                continue
    
    print("\n" + "=" * 70)
    print("✅ Demo completed!")
    print("=" * 70)
    print("\n💡 Tips:")
    print("  • Use --demo N to run a specific demo (1-7)")
    print("  • Use --interactive for interactive translation")
    print("  • Check visualizer.py for visualization features")
    print("  • Check quality_analyzer.py for detailed quality reports")


if __name__ == "__main__":
    main()

