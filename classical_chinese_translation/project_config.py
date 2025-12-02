#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Chinese Translation Project Configuration

Central configuration for paths, parameters, and model settings.
"""
import os

# Import model configuration
from model_config import (
    MODEL_SIZE,
    USE_MODELSCOPE,
    MODELSCOPE_MODELS,
    MODEL_CONFIGS,
    get_model_name,
    get_model_info,
    load_model,
    print_model_info
)

# ============================================================================
# Project Paths
# ============================================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "wedsite_crawling", "诗文数据")
INDEX_DIR = os.path.join(PROJECT_DIR, "index_data")

# Ensure directories exist
os.makedirs(INDEX_DIR, exist_ok=True)

# ============================================================================
# Index File Paths
# ============================================================================

SENTENCE_INDEX_FILE = os.path.join(INDEX_DIR, "sentence_index.pkl")
CLAUSE_INDEX_FILE = os.path.join(INDEX_DIR, "clause_index.pkl")
# Backward compatibility alias for legacy modules expecting PARAGRAPH_INDEX_FILE
PARAGRAPH_INDEX_FILE = CLAUSE_INDEX_FILE
FULL_TEXT_INDEX_FILE = os.path.join(INDEX_DIR, "full_text_index.pkl")

# ============================================================================
# Translation Parameters
# ============================================================================

DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.5

# Sentence delimiters
CLASSICAL_SENTENCE_DELIMITERS = ['。', '！', '？', '；']
MODERN_SENTENCE_DELIMITERS = ['。', '！', '？', '；']
CLAUSE_DELIMITERS = ['，', '、', '：']

# ============================================================================
# Quality Parameters
# ============================================================================

MIN_ALIGNMENT_QUALITY = 0.5  # Minimum quality for index inclusion
HIGH_QUALITY_THRESHOLD = 0.8  # Threshold for "high quality" classification


if __name__ == "__main__":
    print("=" * 60)
    print("Classical Chinese Translation - Configuration")
    print("=" * 60)
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Index Directory: {INDEX_DIR}")
    print(f"Model: {get_model_name()}")
    print("=" * 60)
