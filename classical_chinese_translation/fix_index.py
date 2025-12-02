#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Index Repair Tool

Fixes corrupted index files by removing None/invalid entries.
"""
import os
import pickle
import numpy as np
from smart_index_builder import SmartEmbeddingIndex, SMART_INDEX_FILE

def fix_index():
    print(f"🔧 Repairing index: {SMART_INDEX_FILE}")
    
    if not os.path.exists(SMART_INDEX_FILE):
        print("❌ Index file not found.")
        return

    try:
        with open(SMART_INDEX_FILE, 'rb') as f:
            data = pickle.load(f)
        
        embeddings = data.get('embeddings')
        texts = data.get('texts', [])
        metadata = data.get('metadata', [])
        text_hashes = data.get('text_hashes', [])
        
        print(f"📊 Loaded raw data. Texts: {len(texts)}")
        
        # Fix embeddings
        if embeddings is None:
            print("❌ No embeddings found.")
            return
            
        valid_indices = []
        valid_embeddings = []
        
        # Check if it's a list or numpy array
        if isinstance(embeddings, np.ndarray) and embeddings.dtype != object:
            print("✅ Embeddings look healthy (numpy array).")
            # Just re-normalize to be safe
            index = SmartEmbeddingIndex()
            index.embeddings = embeddings
            index.texts = texts
            index.metadata = metadata
            index.text_hashes = text_hashes
            index.index_type = data.get('index_type', 'sentence')
            index._normalize()
            index.save(SMART_INDEX_FILE)
            return

        print("⚠️ Embeddings need repair (list or object array).")
        
        # Iterate and filter
        iterable = embeddings if isinstance(embeddings, list) else list(embeddings)
        for i, emb in enumerate(iterable):
            if emb is not None and isinstance(emb, (np.ndarray, list)):
                try:
                    # Try to convert to float array
                    valid_emb = np.array(emb, dtype=np.float32)
                    if valid_emb.size > 0:
                        valid_embeddings.append(valid_emb)
                        valid_indices.append(i)
                except:
                    pass
        
        print(f"✅ Recovered {len(valid_embeddings)}/{len(texts)} valid entries.")
        
        if not valid_embeddings:
            print("❌ No valid embeddings recovered.")
            return
            
        # Rebuild arrays
        new_embeddings = np.vstack(valid_embeddings)
        new_texts = [texts[i] for i in valid_indices]
        new_metadata = [metadata[i] for i in valid_indices]
        new_hashes = [text_hashes[i] for i in valid_indices] if text_hashes else []
        
        # Save
        index = SmartEmbeddingIndex()
        index.embeddings = new_embeddings
        index.texts = new_texts
        index.metadata = new_metadata
        index.text_hashes = new_hashes
        index.index_type = data.get('index_type', 'sentence')
        
        index._normalize()
        index.save(SMART_INDEX_FILE)
        print("🎉 Index repair complete!")
        
    except Exception as e:
        print(f"❌ Repair failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_index()

