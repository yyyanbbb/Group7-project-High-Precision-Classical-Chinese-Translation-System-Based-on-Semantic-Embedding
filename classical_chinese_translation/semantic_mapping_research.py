#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Mapping Research: Literal vs. Interpretive Translation

This module analyzes how different translation strategies map in vector space.
It helps distinguish "Literal" (High Lexical Overlap) vs "Interpretive" (Low Overlap, High Semantic Sim) translations.
"""
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
from tqdm import tqdm
import jieba

from smart_index_builder import SmartEmbeddingIndex, SMART_INDEX_FILE

def compute_jaccard_similarity(str1: str, str2: str) -> float:
    """Compute character-level Jaccard similarity."""
    s1 = set(str1)
    s2 = set(str2)
    if not s1 or not s2:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def analyze_semantic_mapping():
    print("📚 Loading index...")
    if not os.path.exists(SMART_INDEX_FILE):
        print("⚠️ Index not found. Please wait for index building to complete.")
        return

    index = SmartEmbeddingIndex()
    index.load(SMART_INDEX_FILE)
    
    print("🔍 Analyzing pairs...")
    
    data_points = []
    
    # Analyze each pair
    # Note: We need modern vectors. For this script, we'll use the pre-computed alignment scores
    # and re-compute similarities if vectors are available.
    
    max_samples = 5000  # Limit to avoid memory issues
    count = 0
    
    for i, meta in enumerate(tqdm(index.metadata)):
        classical = meta.get('classical', '')
        modern = meta.get('modern', '')
        
        if not classical or not modern:
            continue
        
        # Simple sampling
        if i % (len(index.metadata) // max_samples + 1) != 0:
            continue
            
        # 1. Literal Similarity (Character overlap)
        literal_sim = compute_jaccard_similarity(classical, modern)
        
        # 2. Semantic Score (from alignment)
        semantic_score = meta.get('alignment_score', 0.0)
        
        # Classify translation style
        style = "Unknown"
        if literal_sim > 0.3:
            style = "Literal (直译)"
        elif semantic_score > 0.85 and literal_sim < 0.1:
            style = "Interpretive (意译)"
        elif semantic_score < 0.6:
            style = "Low Quality"
        else:
            style = "Balanced"
            
        data_points.append({
            'classical': classical[:50],
            'modern': modern[:50],
            'literal_sim': literal_sim,
            'semantic_score': semantic_score,
            'style': style,
            'length_ratio': len(modern) / max(1, len(classical))
        })
        
        count += 1
        if count >= max_samples:
            break
    
    print(f"✅ Analyzed {len(data_points)} pairs (Sampled)")
    
    # Visualization 1: Scatter Plot (Literal vs Semantic)
    fig = px.scatter(
        data_points,
        x='literal_sim',
        y='semantic_score',
        color='style',
        hover_data=['classical', 'modern'],
        title="Translation Style Analysis: Literal Overlap vs Semantic Quality",
        labels={
            'literal_sim': 'Literal Similarity (Character Overlap)',
            'semantic_score': 'Semantic Alignment Score'
        },
        template='plotly_white'
    )
    
    # Add regions
    fig.add_hline(y=0.85, line_dash="dash", line_color="green", annotation_text="High Semantic Quality")
    fig.add_vline(x=0.1, line_dash="dash", line_color="red", annotation_text="Low Literal Overlap")
    
    output_file = "translation_style_analysis.html"
    fig.write_html(output_file)
    print(f"💾 Visualization saved to {output_file}")
    
    # Print examples of Interpretive Translations
    print("\n🌟 Top Interpretive Translations (High Semantic, Low Literal):")
    interpretive = [d for d in data_points if d['style'] == "Interpretive (意译)"]
    interpretive.sort(key=lambda x: x['semantic_score'], reverse=True)
    
    for i, item in enumerate(interpretive[:10], 1):
        print(f"\n{i}. Score: {item['semantic_score']:.3f} | Overlap: {item['literal_sim']:.3f}")
        print(f"   古: {item['classical']}")
        print(f"   今: {item['modern']}")

if __name__ == "__main__":
    analyze_semantic_mapping()

