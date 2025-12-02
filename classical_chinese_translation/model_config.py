#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-Embedding Model Configuration

This is the core configuration file for model loading and management.
"""
import os
from sentence_transformers import SentenceTransformer

# ============================================================================
# Model Configuration
# ============================================================================

MODEL_SIZE = "4B"  # 4B suitable for 12GB VRAM, 8B requires 16GB+
USE_MODELSCOPE = True  # Use ModelScope mirror for faster downloads in China

MODELSCOPE_MODELS = {
    "Qwen/Qwen3-Embedding-4B": "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B"
}

MODEL_CONFIGS = {
    "4B": {
        "name": "Qwen/Qwen3-Embedding-4B",
        "params": "4B",
        "model_size": "~8GB",
        "memory_required": "8-10GB",
        "c_mteb_score": 72.27,
        "mteb_score": 69.45,
        "description": "Medium size, excellent performance"
    },
    "8B": {
        "name": "Qwen/Qwen3-Embedding-8B",
        "params": "8B",
        "model_size": "~16GB",
        "memory_required": "16GB+",
        "c_mteb_score": 73.84,
        "mteb_score": 70.58,
        "description": "Best performance, requires more resources"
    }
}


def get_model_name():
    """Get configured model name."""
    if MODEL_SIZE not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model size: {MODEL_SIZE}")
    return MODEL_CONFIGS[MODEL_SIZE]["name"]


def get_model_info():
    """Get model information."""
    if MODEL_SIZE not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model size: {MODEL_SIZE}")
    return MODEL_CONFIGS[MODEL_SIZE]


def load_model(device='cuda', **kwargs):
    """
    Load Qwen3-Embedding model.
    
    Args:
        device: 'cuda' or 'cpu'
        **kwargs: Additional parameters
        
    Returns:
        SentenceTransformer model
    """
    import torch
    import warnings
    
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available, switching to CPU")
            device = 'cpu'
        else:
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            print(f"✅ GPU detected: {gpu_name} (CUDA {capability[0]}.{capability[1]})")
            warnings.filterwarnings('ignore', message='.*CUDA.*')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision('medium')
    
    model_name = get_model_name()
    local_cache = os.path.expanduser(f"~/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-{MODEL_SIZE}")
    
    # Try local cache first
    if os.path.exists(local_cache):
        required = ["config.json", "tokenizer.json"]
        model_files = [f for f in os.listdir(local_cache) if f.startswith("model-") and f.endswith(".safetensors")]
        
        if len(model_files) >= 2 and all(os.path.exists(os.path.join(local_cache, f)) for f in required):
            print(f"✅ Loading from local cache: {local_cache}")
            try:
                model = SentenceTransformer(local_cache, device=device, trust_remote_code=True, **kwargs)
                print(f"✅ Model loaded! (device: {device})")
                return model
            except Exception as e:
                print(f"⚠️ Local loading failed: {e}")
    
    # Try ModelScope
    if USE_MODELSCOPE:
        try:
            print(f"🚀 Downloading from ModelScope: {model_name}")
            from modelscope import snapshot_download
            modelscope_name = MODELSCOPE_MODELS.get(model_name, model_name)
            cache_dir = snapshot_download(modelscope_name, cache_dir=None, resume_download=True)
            
            model = SentenceTransformer(cache_dir, device=device, trust_remote_code=True, **kwargs)
            if device == 'cuda':
                model = model.half()
            print(f"✅ Model loaded! (device: {device})")
            return model
        except Exception as e:
            print(f"⚠️ ModelScope failed: {e}")
    
    # Fallback to HuggingFace
    print(f"Loading from HuggingFace: {model_name}")
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True, **kwargs)
    if device == 'cuda':
        model = model.half()
    print(f"✅ Model loaded! (device: {device})")
    return model


def print_model_info():
    """Print model configuration."""
    info = get_model_info()
    print("=" * 60)
    print("Model Configuration")
    print("=" * 60)
    print(f"Model: {info['name']}")
    print(f"Parameters: {info['params']}")
    print(f"Memory: {info['memory_required']}")
    print(f"C-MTEB Score: {info['c_mteb_score']}")
    print("=" * 60)

