# Project Brief: Data-Efficient & Resource-Aware Fine-Tuning of PubMedBERT for Biomedical NER

> **Purpose of this document:** This is a complete technical brief of a finished research project, written to give an AI coding agent ("Antigravity") everything it needs to understand the project end-to-end and turn it into a polished, well-organized GitHub repository (README, folder structure, code, docs, figures). Everything below is grounded in the actual completed Jupyter notebook and its real output — no numbers are invented.

---

## 1. One-line summary

A controlled, reproducible comparison of **Full Fine-Tuning vs LoRA vs QLoRA** for fine-tuning **PubMedBERT** on the **BC5CDR biomedical Named Entity Recognition (NER)** task, with a second, original contribution measuring how each method's performance scales with training-data size (500 → 1,000 → 3,000 → all ~5,228 samples).

## 2. Motivation / Problem Statement

Full fine-tuning of large pretrained language models is expensive in both compute time and GPU memory, which is a real barrier for researchers/clinics with limited hardware (e.g. a single consumer GPU with 4GB VRAM). Parameter-efficient fine-tuning methods — **LoRA** (Low-Rank Adaptation) and **QLoRA** (4-bit-quantized LoRA) — promise similar accuracy while training a tiny fraction of the parameters and using much less memory.

This project asks two concrete research questions:

1. **Q1 (Accuracy/Efficiency tradeoff):** On a real biomedical NER task, do LoRA/QLoRA match Full fine-tuning's accuracy while using dramatically fewer trainable parameters and less GPU memory?
2. **Q2 (Data efficiency — the project's unique contribution):** Which method is more *data-efficient* — i.e., which one performs best when only a small amount of labeled training data (500, 1,000, or 3,000 sentences) is available, which is the realistic scenario for a small hospital or lab that cannot label thousands of documents?

## 3. Dataset

**BC5CDR** (BioCreative V Chemical-Disease Relation corpus), loaded from HuggingFace (`tner/bc5cdr`, loaded via the dataset's auto-converted Parquet branch to avoid deprecated loading-script issues — see Engineering Notes).

- **Task:** Token-level sequence labeling (NER) using the standard BIO tagging scheme.
- **Entity types:** Chemical and Disease (2 entity types, 5 BIO labels total: `O`, `B-Chemical`, `I-Chemical`, `B-Disease`, `I-Disease`).
- **Splits (real counts from the loaded dataset):**
  - Train: 5,228 sentences
  - Validation: 5,330 sentences
  - Test: 5,865 sentences
- Example: `Naloxone reverses the antihypertensive effect of clonidine .` → `Naloxone`=B-Chemical, `clonidine`=B-Chemical, all other tokens=O.

## 4. Base Model

**PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`) — a BERT-base model pretrained from scratch on PubMed abstracts, chosen because it's domain-specific to biomedical text rather than a general-domain BERT. The classification head (`classifier`) is newly initialized for this NER task — it does not exist in the pretrained checkpoint.

## 5. The Three Fine-Tuning Methods Compared

| Method | Description | Trainable Params | What's frozen |
|---|---|---|---|
| **Full Fine-Tuning** | Every weight in the model is updated | 108.9M (100%) | Nothing |
| **LoRA** (r=8) | Low-rank adapter matrices injected into attention layers; only adapters + new classifier head are trained | 5.3M (~4.65%) | The entire PubMedBERT backbone |
| **QLoRA** | Same as LoRA, but the frozen backbone is loaded in **4-bit NF4 quantization** (via `bitsandbytes`) before adapters are attached — trades extra compute for much lower memory | 5.3M (~4.65%) | The entire (quantized) backbone |

**Critical implementation detail:** the new `classifier` head is explicitly excluded from 4-bit quantization (`llm_int8_skip_modules=['classifier']` in `BitsAndBytesConfig`). Without this, the freshly-initialized, never-quantized classifier layer crashes on its first forward pass inside `bitsandbytes` (`AssertionError: module.weight.shape[1] == 1`) — this was a real bug encountered and fixed during development.

## 6. Experimental Design — The Unique Contribution

This is not just "3 methods, 1 run each." The core of the project is a **24-run grid**:

```
3 methods (Full, LoRA, QLoRA)
× 4 data sizes (500, 1,000, 3,000, ALL ≈5,228)
× 2 random seeds (42, 123)
= 24 total training runs
```

Averaging over 2 seeds removes "lucky/unlucky initialization" noise from the comparison. The data-size sweep directly answers Q2 above — e.g., it can show that LoRA degrades far less than Full fine-tuning when only 500 labeled examples are available.

## 7. Evaluation Methodology (designed for rigor, not just convenience)

- **Validation set** is used only for early stopping and best-checkpoint selection during training (`load_best_model_at_end=True`, `metric_for_best_model='f1'`, `EarlyStoppingCallback(patience=2)`).
- **Held-out test set is evaluated separately and is what's reported as the final/headline result** — this is a deliberate fix made during development after noticing an early version of the pipeline was only ever reporting validation scores (a methodological flaw that would have been flagged by any reviewer, since validation-set performance is biased by being used for model selection).
- **Metrics:** entity-level F1, Precision, Recall (via `seqeval`, the standard library for NER evaluation — it scores whole entity spans, not individual tokens) plus token-level Accuracy.
- **Per-entity-type breakdown:** Chemical and Disease are scored separately, not just as one combined NER score, since a method could be strong on one entity type and weak on the other.
- **Efficiency metrics measured automatically for every run:** wall-clock training time, peak GPU VRAM (via `torch.cuda.max_memory_allocated`), trainable parameter count, total parameter count, and training throughput (samples/sec).

## 8. Final Results (real, from the completed 24-run experiment)

### 8.1 Headline comparison — held-out TEST set, full training data, averaged over 2 seeds

| Method | Test F1 | Test Precision | Test Recall | Test Accuracy | Trainable Params | Peak VRAM |
|---|---|---|---|---|---|---|
| **QLoRA** | **0.8964** ± 0.0013 | 0.8771 | 0.9166 | 0.9782 | 5.3M (4.65%) | ~0.50 GB |
| **LoRA** | 0.8955 ± 0.0011 | 0.8768 | 0.9150 | 0.9783 | 5.3M (4.65%) | ~0.81 GB |
| **Full** | 0.8913 ± 0.0040 | 0.8714 | 0.9120 | 0.9784 | 108.9M (100%) | ~2.24 GB |

**Takeaway:** QLoRA and LoRA *slightly exceed* Full fine-tuning's test F1 while training ~20x fewer parameters and using 3–4x less GPU memory. Validation-vs-test score gaps were small (0.7–1.3 F1 points across all 6 full-data runs), indicating no serious overfitting to the validation set.

### 8.2 Data-efficiency study — Test F1 by training set size

| Method | 500 samples | 1,000 samples | 3,000 samples | ALL (~5,228) |
|---|---|---|---|---|
| **Full** | 0.7098 | 0.8230 | 0.8758 | 0.8913 |
| **LoRA** | **0.8020** | **0.8552** | 0.8856 | 0.8955 |
| **QLoRA** | 0.7853 | 0.8529 | **0.8878** | **0.8964** |

**Takeaway — the project's headline finding:** at low data (500 samples), Full fine-tuning is dramatically worse than LoRA (gap of ~9 F1 points), supporting the claim that parameter-efficient methods are meaningfully more data-efficient, which matters most for exactly the resource-constrained settings (small labs, rare diseases, low-resource languages) where this matters in practice.

### 8.3 Per-entity-type breakdown — TEST set, full data

| Method | Chemical F1 | Disease F1 |
|---|---|---|
| Full | 0.9360 | 0.8439 |
| LoRA | 0.9370 | 0.8483 |
| QLoRA | **0.9383** | **0.8485** |

**Takeaway:** Disease entities are consistently harder to recognize than Chemical entities across all three methods (likely because disease names are more linguistically varied — compound terms, abbreviations, synonyms — than chemical names). This holds true regardless of fine-tuning method, suggesting it's a property of the task/dataset, not the method.

## 9. System / Pipeline Architecture

```
BC5CDR Dataset (HuggingFace, Parquet)
        │
        ▼
Data Preprocessing (tokenize + BIO label alignment to PubMedBERT subwords)
        │
        ├──► Data-Size Sampler (deterministic subsampling: 500 / 1K / 3K / ALL)
        │
        ▼
PubMedBERT Backbone
        │
   ┌────┼────────────┐
   ▼    ▼            ▼
 Full  LoRA        QLoRA
  FT   (r=8)      (4-bit NF4 + LoRA)
   │    │            │
   └────┴────────────┘
        │
        ▼
Training Phase (multi-seed × multi-size grid, early stopping, per-epoch checkpointing)
        │
        ▼
Evaluation Module (Validation → model selection only; Held-out Test → final reported scores)
        │
        ▼
Results & Insights (data-size curves, efficiency frontier, per-entity breakdown, master tables)

Cross-cutting: Seed Control Module (42, 123) + Checkpoint & Results Logging
(crash-resilient — see Engineering Notes — Google-Drive-backed when run on Colab)
```

## 10. Engineering Notes — Reliability Features (notable, non-obvious design decisions)

This project was built with real-world interruption in mind (laptop sleep, power loss, Colab session disconnects), not just "run once in one sitting." Key design decisions:

- **Per-epoch checkpointing + automatic resume:** every run saves a checkpoint after every epoch. If interrupted mid-run, simply re-running the training cell resumes from the last saved epoch — no wasted GPU time.
- **`DONE.json` per run + master `all_results.pkl`:** a run is only marked complete after it fully finishes; the master results file is re-saved after *every single run* (not just at the end of all 24), so a crash never throws away more than the one run currently in progress.
- **Google Drive auto-mount on Colab:** Colab's local disk (`/content/...`) is wiped on every disconnect. The notebook auto-detects Colab and redirects all checkpoints/results to Google Drive instead, including a one-time automatic migration of any results that were already sitting on local disk before this fix was added.
- **`log_history` recovery from disk:** an early bug caused training-loss curves to be silently empty for already-completed runs, because the lightweight `DONE.json` cache intentionally excludes the full per-step log to keep it small. Fixed by reconstructing `log_history` directly from the canonical `trainer_state.json` written by HuggingFace `Trainer` inside each checkpoint folder — a "self-healing" step that repairs old cached results without needing to retrain anything.
- **Held-out test-set "backfill":** since an early version of the pipeline never evaluated on the test set, a dedicated cell reloads each of the 24 already-trained models from their saved checkpoints (no retraining) and runs them once against the test set, with progress saved incrementally so a Colab disconnect mid-backfill doesn't lose completed work.
- **Dataset loading workaround:** `tner/bc5cdr` (and most older biomedical NER datasets on HuggingFace) use a now-deprecated Python loading-script format. The fix is to load the dataset's auto-converted Parquet branch directly (`revision='refs/convert/parquet'`), which bypasses the broken script entirely. Because that conversion also strips the dataset's `ClassLabel` metadata (turning label IDs into plain integers with no names attached), the BIO label names are supplied explicitly as a fallback (`['O','B-Chemical','B-Disease','I-Disease','I-Chemical']`), confirmed against the dataset's own example data.
- **`Trainer` API version compatibility:** newer `transformers` versions renamed the `Trainer(tokenizer=...)` constructor argument to `Trainer(processing_class=...)`. The code tries the new name first and falls back to the old one, so it runs correctly regardless of which `transformers` version is installed.

## 11. Tech Stack

- **Model/Training:** `transformers`, `peft` (for LoRA/QLoRA), `bitsandbytes` (4-bit quantization), `accelerate`, PyTorch
- **Data:** `datasets` (HuggingFace), `seqeval` (entity-level NER metrics)
- **Analysis:** `pandas`, `numpy`, `scikit-learn`, `scipy`
- **Visualization:** `matplotlib`, `seaborn`
- **Environment:** Designed to run in Jupyter/Colab; tuned defaults assume a small consumer GPU (originally an RTX 3050, 4GB VRAM) but scales fine to larger GPUs
- **Reproducibility:** fixed seeds (42, 123) applied via `transformers.set_seed()`, deterministic data subsampling for the data-size study

## 12. Repository Structure — what should be built from this

Suggested structure for the GitHub repo (the notebook itself is currently one long sequential file — splitting into modules and a clean README would make this genuinely presentable):

```
pubmedbert-bc5cdr-peft-comparison/
├── README.md                      # project overview, results table, how to run
├── requirements.txt                # pinned dependency versions
├── notebooks/
│   └── PubMedBERT_BC5CDR_Capstone_Project.ipynb   # the full pipeline (as-is, works end to end)
├── results/
│   ├── all_results.pkl             # raw results cache (24 runs)
│   ├── table1_results.csv
│   ├── table2_efficiency.csv
│   └── table_per_entity_breakdown.csv
├── figures/
│   ├── fig1_data_size_vs_metrics.png
│   ├── fig2_final_results_bar.png
│   ├── fig3_efficiency.png
│   ├── fig4_loss_curves.png
│   ├── fig5_radar.png
│   └── fig_per_entity_f1.png
├── docs/
│   └── PROJECT_BRIEF.md            # this document
└── LICENSE
```

## 13. How to Run

1. Open the notebook in Jupyter (local) or Google Colab.
2. Run cells top to bottom. Cell 1 installs all dependencies (restart kernel once after first install).
3. If running on Colab, Cell 3 will prompt to mount Google Drive — approve it so results persist across sessions.
4. The main training loop (24 runs) is fully resumable — if interrupted at any point (crash, disconnect, manual stop), simply re-run the training cell and it continues from the last completed run and the last saved checkpoint within an in-progress run.
5. Once training is complete, run the backfill cell once to compute true held-out test-set scores for all 24 runs.
6. Remaining cells generate all figures/tables for the final report/paper.

## 14. Limitations (worth stating explicitly in the README)

- Only **2 random seeds** were used; this is sufficient to show consistent trends but is a thin basis for strong statistical significance claims. Disclosed honestly rather than implying stronger confidence than the data supports.
- Results are specific to BC5CDR and PubMedBERT; generalization to other biomedical NER datasets or other base models is not directly tested here.
- LoRA rank (r=8) and learning rates were chosen via reasonable defaults rather than an exhaustive hyperparameter search.

## 15. Suggested Citation / Framing for the Paper Title

*"Data-Efficient and Resource-Aware Fine-Tuning of PubMedBERT for Biomedical Named Entity Recognition: A Comparative Study of Full Fine-Tuning, LoRA, and QLoRA on BC5CDR"*
