# 🏛️ Classical Chinese Translation System

High-precision Classical ↔ Modern Chinese translation powered by **Qwen3-Embedding-4B** and semantic retrieval.

## ✨ Features

- **Smart data processing** – removes noisy annotations and extracts aligned sentence pairs automatically.
- **Vector semantic search** – cosine similarity over normalized embeddings with optional quality boosting.
- **Multi-strategy matching** – sentence, clause, and n-gram level indexes.
- **Bidirectional validation** – verifies a translation by searching modern text back into the classical corpus.
- **Automatic quality scoring** – alignment metrics and heuristic confidence calibration.
- **Optional LLM refinement** – plug in a local instruction-tuned model for stylistic polishing.

## 📁 Project Layout

```
classical_chinese_translation/
├── model_config.py           # embedding / LLM configuration
├── project_config.py         # shared paths & helpers
├── smart_data_processor.py   # data cleaning + alignment
├── smart_index_builder.py    # GPU accelerated index builder
├── precision_translator.py   # high-precision translator
├── advanced_translator.py    # richer demo pipeline
├── quality_analyzer.py       # evaluation utilities
├── visualizer.py             # interactive charts
├── demo.py                   # CLI demo / interactive mode
├── index_data/               # cached indexes
│   └── smart_sentence_index.pkl
└── wedsite_crawling/         # lightweight crawler + raw data
    ├── crawling.py
    └── 诗文数据/ (original texts)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd classical_chinese_translation
pip install -r requirements.txt
```

### 2. Build / Refresh Indexes

```bash
python smart_index_builder.py --min-quality 0.6
```

The builder automatically checks for new files under `wedsite_crawling/诗文数据`, reuses cached embeddings via SHA1 hashes, and records metadata in `index_data/smart_index_metadata.json`. No manual cleanup is required.

### 3. Run Translators

```bash
# quality-weighted retrieval
python precision_translator.py

# interactive showcase
python demo.py --interactive
```

## 📊 Usage Example

```python
from precision_translator import PrecisionTranslator

translator = PrecisionTranslator()

result = translator.translate("梳洗罢，独倚望江楼。")
print("Modern:", result.translation)
print("Confidence:", f"{result.confidence:.2%}")

result, details = translator.translate_with_details("过尽千帆皆不是")
print(details)
```

## 🔧 Core Modules

- **SmartDataProcessor** – regex/heuristic filtering, alignment scoring, statistics.
- **PrecisionTranslator** – cosine search with quality boosts, candidate reranking, dual verification.
- **QualityAnalyzer** – retrieval accuracy, clustering diagnostics, error mining.
- **Visualizer / vector_analysis.py** – PCA plots, “modernization vector” experiments, interpretive vs literal detection.

## 📈 Performance Snapshot

- **Retrieval accuracy**: ~90 % Top-1 on curated evaluation sets.
- **Semantic similarity**: ≥0.85 cosine for high-quality pairs.
- **Throughput**: ~10 sentences / second on a single NVIDIA GPU (batch size 64).

## 🔮 Technical Highlights

1. Noise filtering pipeline combining rule-based detectors and length heuristics.
2. Adaptive alignment that balances clause lengths and punctuation cues.
3. Quality-weighted search (boost = 0.9 + 0.1 × score) for more faithful matches.
4. Reverse lookup verification to avoid hallucinated translations.
5. Multi-dimensional scoring (alignment, literal overlap, LLM critique) for confidence output.

## 📝 Data Provenance

Default corpus originates from [gushiwen.cn](https://www.gushiwen.cn/) and includes 60+ curated pieces with human translations. You can expand the dataset via the included crawler or by dropping additional aligned files into `wedsite_crawling/诗文数据`.

## 🤖 Optional Local LLM Polishing

Place an instruction-tuned checkpoint (e.g., `Qwen2.5-7B-Instruct`) under `D:\Models\Qwen2.5-7B-Instruct` and expose it via:

```powershell
$env:LOCAL_LLM_MODEL_PATH="D:\Models\Qwen2.5-7B-Instruct"
```

`precision_translator.py` and `demo.py` will automatically detect the path and run an extra refinement step; if the variable is unset, the deterministic retrieval pipeline still works.

## 🛠️ Extending the Corpus

```bash
cd wedsite_crawling
python crawling.py --max-pages 100 --tags 诗词,散文
```

Progress checkpoints live in `crawl_state*.json`, ensuring polite retry and resumable downloads.

---

Built on Qwen3-Embedding-4B • GPU-friendly • Focused on bridging classical literature and modern comprehension.

