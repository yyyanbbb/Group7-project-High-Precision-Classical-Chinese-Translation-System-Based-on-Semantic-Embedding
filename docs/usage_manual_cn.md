## 古今义理映射项目使用说明书

### 1. 项目概览
本仓库围绕“古文⇌今译”向量化研究，核心流程包括：  
1. 数据抓取与清洗（`wedsite_crawling/`）；  
2. 语句对齐与质量评估（`smart_data_processor.py`）；  
3. 索引与向量缓存（`smart_index_builder.py`、`vector_analysis.py`）；  
4. 语义-字面差异可视化（`translation_style_analysis.html`、`vector_paths.html` 等）。  
所有产出文件默认写入 `classical_chinese_translation/index_data/` 以及 `reports/` 目录。

### 2. 环境与依赖
- Python ≥ 3.10，建议使用 Conda 建立独立环境。  
- GPU（NVIDIA）可显著加速嵌入生成；若仅做可视化复现，CPU 亦可。  
- 安装依赖：`pip install -r classical_chinese_translation/requirements.txt`。  
- 需要访问 Plotly 输出的 HTML 时，确保浏览器可直接打开本地文件。

### 3. 数据获取与少量爬虫
1. 站点配置在 `wedsite_crawling/crawling.py`，默认来源为“古诗文网”。  
2. 断点信息记录在 `wedsite_crawling/crawl_state*.json`，避免重复抓取。  
3. 触发方式：
   ```bash
   cd classical_chinese_translation/wedsite_crawling
   python crawling.py --max-pages 50 --tags 诗词,散文
   ```
4. 抓取完成的原始文本存放在 `wedsite_crawling/诗文数据/`；若需新语料，可复制同目录结构并在数据处理阶段指定路径。

### 4. 构建索引（已预构建，可复用）
> **注意**：仓库已包含最新的 `smart_sentence_index.pkl`，如非必要勿重复训练，只需复用缓存即可。

1. 首次或数据更新时运行：
   ```bash
   cd classical_chinese_translation
   python smart_index_builder.py --min-quality 0.6 --batch-size 64
   ```
2. 程序会：
   - 自动计算数据指纹；若内容未变则跳过重建；
   - 复用历史嵌入（依据 `text_hash`）避免重复 GPU 计算；
   - 生成 `smart_index_metadata.json` 记录数据规模、模型版本等。
3. 若仅想加载现有索引，在代码中调用：
   ```python
   from smart_index_builder import SmartIndexBuilder
   builder = SmartIndexBuilder()
   builder.build_indexes(force_rebuild=False)
   index = builder.get_index()
   ```

### 5. 语义-字面分析与可视化
| 工具 | 命令 | 主要输出 | 说明 |
| --- | --- | --- | --- |
| `semantic_mapping_research.py` | `python classical_chinese_translation/semantic_mapping_research.py` | `translation_style_analysis.html` | 通过 Jaccard 字面重叠与对齐得分区分“直译 / 意译 / 低质量”样本。截图中前 10 例全部满足 `cos=1.0` 且 `Overlap<0.1`，展示“世间水 vs 我不想学习”类句对。 |
| `vector_analysis.py` | `python classical_chinese_translation/vector_analysis.py --no-cache`（重算时） | `vector_paths.html`、`translation_cluster_map.html`、`cosine_similarity_report.json`、`modernization_direction.json` | 研究“Modern - Classical”向量，输出转换轨迹、聚类（默认 4~5 组意译簇）以及“现代化向量”代表样本。命令行中还会展示 `Vector Arithmetic Search Test`，可验证“我不想学习”“我很帅”等现代高频表达对应的古文等价表述。 |
| `reports/figures/*.png` | — | `embedding_space.png`、`source_distribution.png`、`text_length_distribution.png` | 由 `generate_report_images.py` 生成的统计图，可直接插入文档或幻灯片。 |

### 6. 向量算术与现代化方向
- `vector_analysis.py` 中的 `vector_arithmetic_search` 默认公式：  
  \[
  \text{target} = \text{ModernQuery} - (\overline{\text{Modern}} - \overline{\text{Classical}})
  \]
  可借此寻找“非字面但语义相关”的古文候选。  
- `modernization_direction.json` 提供代表性样本：  
  - **Most aligned**：格式化、口语化程度高（如“孔子东游”→现代人物对白）；  
  - **Least aligned**：保留诗意或隐喻（“采菊东篱下”仍呈现高度文学性）。  

### 7. 快速复现流程
1. **准备**：确保 `index_data/` 现有文件未被删除。若需测试新语料，可在 `smart_index_builder.py` 中设置 `limit`。  
2. **分析命令**：
   ```bash
   python classical_chinese_translation/vector_analysis.py \
       --max-samples 3000 --batch-size 32 --clusters 5 \
       --interpretive-threshold 0.8 --literal-threshold 0.12
   ```
   运行后 `reports/` 会刷新 JSON/HTML。
3. **查看结果**：双击打开 `translation_style_analysis.html`、`translation_cluster_map.html`、`vector_paths.html`。  
4. **撰写报告**：本次提交附带 `reports/modernization_report.tex` 与 `modernization_report.zip`，可直接用于论文撰写。

### 8. 常见问题
1. **显存不足**：降低 `--batch-size` 或在 `project_config.load_model` 中切换到 CPU。  
2. **重复计算**：`vector_analysis.py` 默认开启嵌入缓存，可用 `--no-cache` 强制重算。  
3. **爬虫限流**：`wedsite_crawling/crawling.py` 内的 `REQUEST_INTERVAL` 可调，务必遵守目标站点 robots 规范。  
4. **Plotly 打不开**：若浏览器提示脚本受阻，可手动托到 VSCode Live Server 或使用 `python -m http.server` 临时托管。

---
如需扩展到“现代汉语 - 古代汉语 = 口语化”之类的向量解释，可结合 `cosine_similarity_report.json` 中的 interpretive 高分样本进行人工标注，或在 `vector_analysis.py` 里增加主题模型以解耦文化因素。祝研究顺利！

