#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Chinese Translation Visualizer

This module provides visualization tools for analyzing the embedding space
and translation quality. Inspired by 04_text_clustering_visualization.py.

Features:
- t-SNE/UMAP dimensionality reduction
- Interactive similarity heatmaps
- Translation quality distribution plots
- Source text clustering visualization
"""
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

from data_processor import DataProcessor
from index_builder import EmbeddingIndex, SENTENCE_INDEX_FILE
from project_config import load_model, INDEX_DIR

# Try to import UMAP (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not installed. Install with: pip install umap-learn")


class TranslationVisualizer:
    """
    Visualization tool for classical Chinese translation embeddings.
    
    Provides multiple visualization types for analyzing the embedding
    space and understanding translation behavior.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = output_dir or os.path.join(INDEX_DIR, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.data_processor = DataProcessor()
        self.index: Optional[EmbeddingIndex] = None
        self.model = None
        
    def load_data(self):
        """Load index and data."""
        print("📚 Loading data for visualization...")
        
        # Load index
        self.index = EmbeddingIndex()
        self.index.load(SENTENCE_INDEX_FILE)
        
        # Load data processor
        self.data_processor.load_all_data()
        self.data_processor.create_sentence_pairs()
        
        print(f"✅ Loaded {len(self.index.texts)} indexed entries")
    
    def visualize_embedding_space_2d(self, method: str = "tsne",
                                     perplexity: int = 30,
                                     n_samples: int = 500,
                                     save_path: str = None) -> go.Figure:
        """
        Create 2D visualization of the embedding space.
        
        Args:
            method: Dimensionality reduction method ("tsne" or "umap")
            perplexity: t-SNE perplexity parameter
            n_samples: Number of samples to visualize
            save_path: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        if self.index is None:
            self.load_data()
        
        # Sample if needed
        n = min(n_samples, len(self.index.embeddings))
        indices = np.random.choice(len(self.index.embeddings), n, replace=False)
        
        embeddings = self.index.embeddings[indices]
        texts = [self.index.texts[i] for i in indices]
        metadata = [self.index.metadata[i] for i in indices]
        
        print(f"📊 Reducing {n} embeddings to 2D using {method.upper()}...")
        
        # Dimensionality reduction
        if method == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
        else:
            tsne = TSNE(n_components=2, perplexity=min(perplexity, n-1), 
                       random_state=42, max_iter=1000)
            coords = tsne.fit_transform(embeddings)
        
        # Extract titles for coloring
        titles = [m.get('title', 'Unknown') for m in metadata]
        unique_titles = list(set(titles))
        
        # Create figure
        fig = go.Figure()
        
        # Color palette
        colors = px.colors.qualitative.Set3[:len(unique_titles)]
        
        for i, title in enumerate(unique_titles):
            mask = [t == title for t in titles]
            indices_mask = [j for j, m in enumerate(mask) if m]
            
            hover_texts = [
                f"<b>{title}</b><br>" +
                f"Text: {texts[j][:80]}...<br>" +
                f"Translation: {metadata[j].get('modern', '')[:80]}..."
                for j in indices_mask
            ]
            
            fig.add_trace(go.Scatter(
                x=coords[indices_mask, 0],
                y=coords[indices_mask, 1],
                mode='markers',
                name=title[:20] + "..." if len(title) > 20 else title,
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text=f"Classical Chinese Text Embedding Space ({method.upper()} 2D)",
                font=dict(size=18)
            ),
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            width=1200,
            height=800,
            template='plotly_white',
            legend=dict(
                title="Source",
                font=dict(size=10),
                itemsizing='constant'
            )
        )
        
        # Save
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"embedding_space_{method}_2d.html")
        
        self._save_figure(fig, save_path)
        
        return fig
    
    def visualize_similarity_heatmap(self, sample_texts: List[str] = None,
                                     n_samples: int = 20,
                                     save_path: str = None) -> go.Figure:
        """
        Create a similarity heatmap for sample texts.
        
        Args:
            sample_texts: Specific texts to compare (uses random if None)
            n_samples: Number of samples if sample_texts is None
            save_path: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        if self.index is None:
            self.load_data()
        
        # Get sample texts
        if sample_texts is None:
            indices = np.random.choice(len(self.index.texts), 
                                       min(n_samples, len(self.index.texts)), 
                                       replace=False)
            sample_texts = [self.index.texts[i] for i in indices]
        
        # Get similarity matrix
        similarity_matrix = self.index.get_similarity_matrix(sample_texts)
        
        # Truncate texts for display
        display_texts = [t[:15] + "..." if len(t) > 15 else t for t in sample_texts]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=display_texts,
            y=display_texts,
            colorscale='RdYlGn',
            text=np.round(similarity_matrix, 3),
            texttemplate='%{text}',
            textfont=dict(size=8),
            hovertemplate='Text1: %{y}<br>Text2: %{x}<br>Similarity: %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Classical Chinese Text Similarity Heatmap",
            xaxis_title="Text",
            yaxis_title="Text",
            width=1000,
            height=900,
            template='plotly_white'
        )
        
        # Save
        if save_path is None:
            save_path = os.path.join(self.output_dir, "similarity_heatmap.html")
        
        self._save_figure(fig, save_path)
        
        return fig
    
    def visualize_text_length_distribution(self, save_path: str = None) -> go.Figure:
        """
        Visualize the distribution of text lengths.
        
        Args:
            save_path: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        if self.data_processor.sentence_pairs is None:
            self.load_data()
        
        # Collect lengths
        classical_lengths = [len(sp.classical) for sp in self.data_processor.sentence_pairs]
        modern_lengths = [len(sp.modern) for sp in self.data_processor.sentence_pairs]
        
        # Create subplots
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("Classical Text Lengths", "Modern Translation Lengths"))
        
        fig.add_trace(
            go.Histogram(x=classical_lengths, name="Classical", 
                        marker_color='#636EFA', nbinsx=30),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=modern_lengths, name="Modern", 
                        marker_color='#EF553B', nbinsx=30),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Text Length Distribution",
            showlegend=True,
            width=1200,
            height=500,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Character Count", row=1, col=1)
        fig.update_xaxes(title_text="Character Count", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        # Save
        if save_path is None:
            save_path = os.path.join(self.output_dir, "text_length_distribution.html")
        
        self._save_figure(fig, save_path)
        
        return fig
    
    def visualize_source_distribution(self, save_path: str = None) -> go.Figure:
        """
        Visualize the distribution of texts by source.
        
        Args:
            save_path: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        if self.data_processor.text_pairs is None:
            self.load_data()
        
        # Count by title
        from collections import Counter
        title_counts = Counter(sp.title for sp in self.data_processor.sentence_pairs)
        
        # Sort by count
        sorted_items = sorted(title_counts.items(), key=lambda x: x[1], reverse=True)
        titles = [item[0][:25] + "..." if len(item[0]) > 25 else item[0] for item in sorted_items[:20]]
        counts = [item[1] for item in sorted_items[:20]]
        
        fig = go.Figure(data=[
            go.Bar(
                x=counts,
                y=titles,
                orientation='h',
                marker_color='#636EFA'
            )
        ])
        
        fig.update_layout(
            title="Sentence Pairs by Source (Top 20)",
            xaxis_title="Number of Sentence Pairs",
            yaxis_title="Source",
            height=600,
            width=1000,
            template='plotly_white',
            yaxis=dict(autorange="reversed")
        )
        
        # Save
        if save_path is None:
            save_path = os.path.join(self.output_dir, "source_distribution.html")
        
        self._save_figure(fig, save_path)
        
        return fig
    
    def _save_figure(self, fig: go.Figure, filepath: str):
        """
        Save figure to HTML file.
        
        Args:
            fig: Plotly figure
            filepath: Output path
        """
        try:
            html_str = fig.to_html(
                include_plotlyjs='cdn',
                config={'displayModeBar': True, 'responsive': True},
                include_mathjax=False
            )
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_str)
            print(f"✅ Saved visualization to: {filepath}")
        except Exception as e:
            print(f"⚠️ Failed to save visualization: {e}")
    
    def generate_all_visualizations(self):
        """Generate all available visualizations."""
        print("=" * 60)
        print("Generating All Visualizations")
        print("=" * 60)
        
        self.load_data()
        
        print("\n1. Embedding space (t-SNE 2D)...")
        self.visualize_embedding_space_2d(method="tsne", n_samples=300)
        
        print("\n2. Similarity heatmap...")
        self.visualize_similarity_heatmap(n_samples=15)
        
        print("\n3. Text length distribution...")
        self.visualize_text_length_distribution()
        
        print("\n4. Source distribution...")
        self.visualize_source_distribution()
        
        print(f"\n✅ All visualizations saved to: {self.output_dir}")


def main():
    """Generate visualizations."""
    print("=" * 60)
    print("Classical Chinese Translation Visualizer")
    print("=" * 60)
    
    visualizer = TranslationVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()

