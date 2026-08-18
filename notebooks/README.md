# Notebooks

## `PubMedBERT_BC5CDR_MERGED_FINAL.ipynb` *(Latest & Recommended)*

This is the **complete, self-contained, end-to-end pipeline** for the capstone project. It runs the full 24-run experimental grid (3 fine-tuning methods × 4 dataset sizes × 2 seeds), computes all held-out test evaluations, exports all publication-ready figures & Excel tables, launches an **interactive Gradio Web Demo with PDF analysis**, and automates model deployment to the **Hugging Face Hub**.

---

## 📋 Comprehensive Cell-by-Cell Architecture

| Cell # | Section / Component | Purpose & Implementation Details |
|:---:|:---|:---|
| **0** | **Title & Paper Metadata** | Markdown overview, author attribution, citation metadata, and research objectives. |
| **1** | **Package Installation** | Installs `transformers`, `peft`, `bitsandbytes`, `seqeval`, `accelerate`, and dependencies. |
| **2** | **Imports & Environment** | Loads PyTorch, Hugging Face, scientific computing stack; configures logging & device setup. |
| **3** | **Global Configuration & Drive Mount** | Sets dataset hyperparameters, learning rates, seed lists `[42, 123]`, training size grid `[500, 1000, 3000, ALL]`, and auto-mounts Google Drive for persistent checkpointing. |
| **4** | **BC5CDR Dataset Loading** | Ingests `tner/bc5cdr` via the Parquet branch workaround (`revision='refs/convert/parquet'`) to bypass deprecated loading scripts with explicit BIO label mapping (`['O', 'B-Chemical', 'B-Disease', 'I-Disease', 'I-Chemical']`). |
| **5** | **Tokenizer & Token Alignment** | Initializes PubMedBERT WordPiece tokenizer, aligns BIO entity tags with subword tokens using first-subword labeling, and caps sequence length at 128 tokens. |
| **6** | **Evaluation Metrics** | Implements entity-level span F1, Precision, Recall, and Accuracy scoring via `seqeval`. |
| **7** | **Data Collator & Model Factories** | Sets up dynamic batch collation and creates modular factories for **Full Fine-Tuning**, **LoRA ($r=8$)**, and **QLoRA (4-bit NF4)** with `llm_int8_skip_modules=['classifier']` to prevent classifier initialization crashes. |
| **8** | **Hardware & Resource Profiler** | Integrates CUDA memory tracking (`torch.cuda.max_memory_allocated`) and wall-clock training timer utilities. |
| **9** | **Experiment Engine (`run_experiment`)** | Encapsulates single-run execution with per-epoch evaluation, checkpointing, early stopping (patience=2), and `DONE.json` crash-resilience sentinels. |
| **10** | **Master 24-Run Training Loop** | Iterates over all 24 configurations (3 methods × 4 sizes × 2 seeds). Automatically resumes from the last completed run if interrupted. |
| **11** | **Results Compilation & Best Models** | Aggregates all run metadata into a unified pandas DataFrame (`res_df`) and highlights top performers. |
| **12** | **Held-Out Test Set Backfill** | Iteratively reloads the 24 checkpointed models and computes true held-out test set metrics (5,865 sentences) with incremental persistence. |
| **13** | **Per-Entity Breakdown** | Evaluates fine-grained performance on **Chemical** (5,384 test entities) vs **Disease** (4,424 test entities) categories. Results cached to disk (`entity_reports_cache.pkl`). |
| **14** | **Figure 1: Data Efficiency Curves** | Plots test F1 vs training set size ($N \in \{500, 1000, 3000, 5228\}$) demonstrating adapter superiority in low-data regimes. |
| **15** | **Figure 2: Final Results Bar Chart** | Renders comparative bar plots for F1, Precision, Recall, and Accuracy at full dataset size ($N=5,228$). |
| **16** | **Figure 3: Training Efficiency** | Visualizes training wall-clock time, peak GPU VRAM allocation, and trainable parameter counts. |
| **17** | **Figure 4: Loss Curves (Self-Healing)** | Plots train/validation cross-entropy loss convergence across epochs with automated `trainer_state.json` reconstruction. |
| **18** | **Table I: Master Results Export** | Formats and outputs master statistical summary tables (mean ± std across seeds). |
| **19** | **Figure 5: Multi-Metric Radar Chart** | Generates 5-axis radar chart (F1, Precision, Recall, Parameter Efficiency, Speed Score). |
| **20** | **Executive Summary** | Prints terminal summary of findings, memory reductions, and data-efficiency metrics. |
| **21** | **Demo Dependencies** | Installs `gradio` and `PyMuPDF` (`fitz`) for web UI and PDF parsing. |
| **22** | **Inference Engine & Span Formatter** | Builds cached inference pipeline supporting on-the-fly model switching, BIO subword span decoding, confidence scoring, and HTML entity formatting. |
| **23** | **Interactive Gradio Web App** | Launches interactive web application with real-time entity highlighting, PDF document analysis, entity count statistics, and curated clinical examples. |
| **24** | **Hugging Face Hub Auto-Publisher** | Selects best-performing seed models for Full FT, LoRA, and QLoRA, and uploads them to the Hugging Face Model Hub repository (`Venkatsaikiran/pubmedbert-bc5cdr-ner`). |

---

## ⚡ Hardware & Execution Profile (RTX 3050 / Google Colab T4)

| Method | Full Data Train Time | Peak VRAM | Trainable Parameters |
|:---|:---:|:---:|:---:|
| **Full Fine-Tuning** | ~9.26 min | 2.241 GB | 108.9M (100%) |
| **LoRA ($r=8$)** | ~16.88 min | 0.805 GB | **5.3M (4.65%)** |
| **QLoRA (4-bit NF4)** | ~21.43 min | **0.496 GB** | **5.3M (4.65%)** |
| **Total Pipeline (24 runs)** | **~4–5 hours** | **< 2.3 GB** | Fully Crash-Resilient & Resumable |

---

## 🌐 Live Demonstration & Model Weights

- **Hugging Face Spaces Demo**: [https://huggingface.co/spaces/Venkatsaikiran/pubmedbert-ner-demo](https://huggingface.co/spaces/Venkatsaikiran/pubmedbert-ner-demo)
- **Hugging Face Model Repository**: [https://huggingface.co/Venkatsaikiran/pubmedbert-bc5cdr-ner](https://huggingface.co/Venkatsaikiran/pubmedbert-bc5cdr-ner)
