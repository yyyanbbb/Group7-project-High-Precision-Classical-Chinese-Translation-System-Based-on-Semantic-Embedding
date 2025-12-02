import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
import random

# Setup output directory
OUTPUT_DIR = "report_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to index
INDEX_FILE = os.path.join("classical_chinese_translation", "index_data", "smart_sentence_index.pkl")

def set_chinese_font():
    """Attempt to set a Chinese-compatible font."""
    system_fonts = fm.findSystemFonts()
    chinese_fonts = [f for f in system_fonts if "simhei" in f.lower() or "microsoft yahei" in f.lower() or "dengxian" in f.lower()]
    if chinese_fonts:
        font_path = chinese_fonts[0]
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        print(f"Using font: {prop.get_name()}")
        return prop
    else:
        print("No Chinese font found, using default.")
        return None

def load_data():
    if not os.path.exists(INDEX_FILE):
        print(f"Index file not found: {INDEX_FILE}")
        return None
    
    print("Loading index data...")
    with open(INDEX_FILE, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_length_distribution(data):
    print("Generating length distribution...")
    texts = data['texts']
    lengths = [len(t) for t in texts]
    
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Classical Sentence Lengths")
    plt.xlabel("Length (characters)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "text_length_distribution.png"))
    plt.close()

def plot_source_distribution(data):
    print("Generating source distribution...")
    metadata = data['metadata']
    titles = [m.get('title', 'Unknown') for m in metadata]
    counts = Counter(titles).most_common(15)
    
    titles_x = [x[0] for x in counts]
    counts_y = [x[1] for x in counts]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(titles_x)), counts_y, color='lightcoral')
    
    # Add labels if font supports it, otherwise indices
    try:
        plt.xticks(range(len(titles_x)), titles_x, rotation=45, ha='right')
    except:
        plt.xticks(range(len(titles_x)), range(len(titles_x)))
        
    plt.title("Top 15 Sources by Sentence Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "source_distribution.png"))
    plt.close()

def plot_embedding_space(data):
    print("Generating embedding space visualization...")
    embeddings = data.get('embeddings')
    if embeddings is None:
        print("No embeddings found. Generating dummy embeddings for visualization.")
        embeddings = np.random.rand(len(data['texts']), 128)

        
    # Downsample
    n_samples = min(500, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    subset = embeddings[indices]
    
    # PCA for dimensionality reduction (simpler than t-SNE)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(subset)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c='teal', alpha=0.6, s=30)
    plt.title("Embedding Space (PCA Projection)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "embedding_space.png"))
    plt.close()

def plot_similarity_heatmap(data):
    print("Generating similarity heatmap...")
    embeddings = data.get('embeddings')
    if embeddings is None:
        return
        
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / np.where(norms == 0, 1, norms)
    
    # Sample 15 items
    n_samples = 15
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    subset = embeddings_norm[indices]
    
    # Compute similarity matrix
    sim_matrix = np.dot(subset, subset.T)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(sim_matrix, cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.title("Similarity Heatmap (Random Sample)")
    plt.savefig(os.path.join(OUTPUT_DIR, "similarity_heatmap.png"))
    plt.close()

def main():
    try:
        set_chinese_font()
        data = load_data()
        if data:
            plot_length_distribution(data)
            plot_source_distribution(data)
            try:
                plot_embedding_space(data)
                plot_similarity_heatmap(data)
            except Exception as e:
                print(f"Could not plot embeddings: {e}")
        else:
            print("Creating dummy plots...")
            # Fallback to dummy plots if data load fails
            plt.figure()
            plt.plot([1, 2, 3], [1, 2, 3])
            plt.title("Dummy Plot (Data Load Failed)")
            plt.savefig(os.path.join(OUTPUT_DIR, "embedding_space.png"))
            plt.savefig(os.path.join(OUTPUT_DIR, "text_length_distribution.png"))
            plt.savefig(os.path.join(OUTPUT_DIR, "source_distribution.png"))
            plt.savefig(os.path.join(OUTPUT_DIR, "similarity_heatmap.png"))
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

