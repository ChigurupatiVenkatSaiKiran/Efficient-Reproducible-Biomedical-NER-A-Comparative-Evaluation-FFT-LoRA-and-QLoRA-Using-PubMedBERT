<div align="center">

<img src="figures/Architecture_FINAL.svg" width="600" alt="System Architecture"/>

# 🧬 Data-Efficient & Resource-Aware Fine-Tuning of PubMedBERT for Biomedical NER

### *A Controlled Comparison of Full Fine-Tuning · LoRA · QLoRA on BC5CDR*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-FFD21E?style=for-the-badge)](https://huggingface.co)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%7C%20QLoRA-00A651?style=for-the-badge)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Open In Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

**Chigurupati Venkat Sai Kiran** · Capstone Project · 2025–26

</div>

---

## 📌 One-Line Summary

> A reproducible, 24-run experimental study showing that **LoRA and QLoRA match — and slightly exceed — Full fine-tuning** of PubMedBERT on biomedical NER (BC5CDR), while using **~20× fewer trainable parameters** and **3–4× less GPU memory**, with the additional finding that PEFT methods are dramatically **more data-efficient** at low training sizes (≤500 samples).

---

## 🗂️ Table of Contents

- [Motivation](#-motivation--problem-statement)
- [Key Results](#-key-results)
- [Experimental Design](#-experimental-design--the-unique-contribution)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Methods](#-the-three-methods)
- [Results Deep Dive](#-results-deep-dive)
- [Engineering Notes](#-engineering-notes--reliability-features)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Tech Stack](#-tech-stack)
- [Limitations](#-limitations)
- [Citation](#-citation)

---

## 💡 Motivation & Problem Statement

Full fine-tuning of large pre-trained language models demands significant compute and GPU memory — a real barrier for researchers and clinicians operating on **consumer-grade hardware** (e.g., a single 4 GB VRAM GPU). Parameter-efficient fine-tuning (PEFT) methods like **LoRA** and **QLoRA** promise similar accuracy while updating only a tiny fraction of parameters.

This project answers **two concrete research questions**:

| # | Research Question |
|---|---|
| **Q1** | On a real biomedical NER task, do LoRA/QLoRA match Full fine-tuning's accuracy while using dramatically fewer trainable parameters and less GPU memory? |
| **Q2** *(unique contribution)* | Which method is more **data-efficient** — i.e., which performs best when only 500–3,000 labeled training examples are available, as in a small hospital or rare-disease lab? |

---

## 🏆 Key Results

### Headline — Held-out TEST Set · Full Training Data · Averaged over 2 Seeds

| Method | Test F1 | Precision | Recall | Accuracy | Trainable Params | Peak VRAM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **QLoRA** | **0.8964** ± 0.0013 | 0.8771 | 0.9166 | 0.9782 | 5.3M (4.65%) | **~0.50 GB** |
| 🥈 **LoRA** | 0.8955 ± 0.0011 | 0.8768 | 0.9150 | 0.9783 | 5.3M (4.65%) | ~0.81 GB |
| 🥉 **Full FT** | 0.8913 ± 0.0040 | 0.8714 | 0.9120 | 0.9784 | 108.9M (100%) | ~2.24 GB |

> **QLoRA and LoRA slightly exceed Full fine-tuning's test F1 while training ~20× fewer parameters and using 3–4× less GPU memory.**

---

### 🔑 The Project's Headline Finding — Data Efficiency

| Method | 500 samples | 1,000 samples | 3,000 samples | ALL (~5,228) |
|:---|:---:|:---:|:---:|:---:|
| Full FT | 0.7098 | 0.8230 | 0.8758 | 0.8913 |
| **LoRA** | **0.8020** | **0.8552** | 0.8856 | 0.8955 |
| **QLoRA** | 0.7853 | 0.8529 | **0.8878** | **0.8964** |

> ⚡ At just **500 samples**, LoRA outperforms Full fine-tuning by **~9 F1 points** — a dramatic gap that is practically significant for any lab with limited labeled data.

---

## 🔬 Experimental Design — The Unique Contribution

This is not just "3 methods, 1 run each." The core of the project is a **24-run controlled grid**:

```
3 methods (Full, LoRA, QLoRA)
  × 4 data sizes (500, 1,000, 3,000, ALL ≈ 5,228)
  × 2 random seeds (42, 123)
= 24 total training runs
```

Averaging over 2 seeds removes lucky/unlucky initialization noise. The data-size sweep directly answers Q2 — measuring *how fast* each method's performance degrades as labeled data shrinks.

---

## 🏗️ System Architecture

<div align="center">
<img src="figures/Architecture_FINAL.svg" width="560" alt="Pipeline Architecture"/>
</div>

```
BC5CDR Dataset (HuggingFace, Parquet)
        │
        ▼
Data Preprocessing (tokenize + BIO label alignment to PubMedBERT subwords)
        │
        ├──► Data-Size Sampler (500 / 1K / 3K / ALL — deterministic)
        │
        ▼
PubMedBERT Backbone (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
        │
   ┌────┼────────────┐
   ▼    ▼            ▼
 Full  LoRA        QLoRA
  FT   (r=8)      (4-bit NF4 + LoRA)
   │    │            │
   └────┴────────────┘
        │
        ▼
Training (multi-seed × multi-size grid, early stopping, per-epoch checkpoints)
        │
        ▼
Evaluation (Validation → model selection only; Held-out Test → final scores)
        │
        ▼
Results & Insights (data-size curves, efficiency frontier, entity breakdown)

Cross-cutting: Seed Control (42, 123) + Crash-Resilient Checkpoint & Results Logging
```

---

## 📊 Dataset

**[BC5CDR](https://huggingface.co/datasets/tner/bc5cdr)** — BioCreative V Chemical-Disease Relation corpus

| Split | Sentences |
|:---|:---:|
| Train | 5,228 |
| Validation | 5,330 |
| Test | 5,865 |

- **Task:** Token-level NER using the BIO tagging scheme
- **Entity types:** Chemical · Disease (5 BIO labels: `O`, `B-Chemical`, `I-Chemical`, `B-Disease`, `I-Disease`)
- **Example:** `Naloxone reverses the antihypertensive effect of clonidine.`
  - `Naloxone` → `B-Chemical`, `clonidine` → `B-Chemical`

> **Engineering note:** The dataset is loaded from its auto-converted Parquet branch (`revision='refs/convert/parquet'`) to bypass the deprecated Python loading-script format. Because this strips `ClassLabel` metadata, BIO label names are supplied explicitly as a fallback — confirmed against the dataset's own example data.

---

## ⚙️ The Three Methods

| Method | Description | Trainable Params | Frozen |
|:---|:---|:---:|:---|
| **Full Fine-Tuning** | Every weight in the 110M-param model is updated | 108.9M (100%) | Nothing |
| **LoRA** (r=8) | Low-rank adapter matrices injected into attention layers; only adapters + new classifier head trained | 5.3M (~4.65%) | Entire PubMedBERT backbone |
| **QLoRA** | Same as LoRA, but the frozen backbone is loaded in **4-bit NF4 quantization** via `bitsandbytes` | 5.3M (~4.65%) | Entire (quantized) backbone |

> **Critical implementation detail:** the new `classifier` head is explicitly excluded from 4-bit quantization (`llm_int8_skip_modules=['classifier']` in `BitsAndBytesConfig`). Without this, the freshly-initialized classifier layer crashes on its first forward pass inside `bitsandbytes` — a real bug encountered and fixed during development.

---

## 📈 Results Deep Dive

### Figure 1 — Data Size vs. Metrics (F1, Precision, Recall, Accuracy)

<div align="center">
<img src="figures/fig1_data_size_vs_metrics.png" alt="Fig 1: Data Size vs Metrics" width="900"/>
</div>

*At 500 samples, Full FT lags by ~9 F1 points. LoRA and QLoRA converge rapidly and maintain the lead across all data sizes.*

---

### Figure 2 — Final Results: F1 / Precision / Recall / Accuracy (Full Dataset)

<div align="center">
<img src="figures/fig2_final_results_bar.png" alt="Fig 2: Final Results Bar Chart" width="700"/>
</div>

---

### Figure 3 — Training Efficiency: Time · GPU Memory · Parameter Count

<div align="center">
<img src="figures/fig3_efficiency.png" alt="Fig 3: Efficiency Comparison" width="700"/>
</div>

| Method | Avg Train Time | Peak VRAM | Trainable Params |
|:---|:---:|:---:|:---:|
| Full FT | 5.3 min | 2.24 GB | 108.9M |
| LoRA | 8.9 min | 0.78 GB | 5.3M |
| QLoRA | 13.2 min | **0.50 GB** | 5.3M |

> QLoRA uses the least memory (4× less than Full FT) at the cost of slightly longer training time due to quantization overhead.

---

### Figure 4 — Training & Validation Loss Curves (Full Dataset)

<div align="center">
<img src="figures/fig4_loss_curves.png" alt="Fig 4: Loss Curves" width="800"/>
</div>

*Both seeds converge smoothly for all three methods. Train/val gap is small — no significant overfitting observed.*

---

### Figure 5 — Multi-Metric Radar Chart (Full Dataset)

<div align="center">
<img src="figures/fig5_radar.png" alt="Fig 5: Radar Chart" width="500"/>
</div>

*LoRA and QLoRA dominate the radar on Param Efficiency. All three methods are comparable on F1, Precision, and Recall.*

---

### Figure 6 — Per-Entity-Type F1: Chemical vs. Disease

<div align="center">
<img src="figures/fig_per_entity_f1.png" alt="Fig 6: Per Entity F1" width="600"/>
</div>

| Method | Chemical F1 | Disease F1 |
|:---|:---:|:---:|
| Full FT | 0.9360 | 0.8439 |
| LoRA | 0.9370 | 0.8483 |
| **QLoRA** | **0.9383** | **0.8485** |

> Disease entities are consistently ~9 F1 points harder to recognize than Chemical entities across all methods — likely due to greater linguistic variation (compound terms, abbreviations, synonyms). This is a property of the dataset, not the fine-tuning method.

---

## 🛡️ Engineering Notes — Reliability Features

This project was built for real-world interruption (laptop sleep, power loss, Colab disconnects), not just a single clean run.

| Feature | Description |
|:---|:---|
| **Per-epoch checkpointing + auto-resume** | Every run saves after every epoch. Re-running the training cell resumes from the last completed checkpoint — no wasted GPU time. |
| **`DONE.json` + `all_results.pkl`** | A run is only marked complete after fully finishing. The master results file is re-saved after *every single run*, so a crash never loses more than the current in-progress run. |
| **Google Drive auto-mount on Colab** | Detects Colab and redirects all checkpoints/results to Drive. Includes one-time migration of results already on local disk. |
| **`log_history` recovery from disk** | Training-loss curves are reconstructed directly from `trainer_state.json` inside each checkpoint folder, repairing any runs where loss curves were silently empty without retraining. |
| **Held-out test-set backfill** | A dedicated cell reloads all 24 trained models from checkpoints (no retraining) and evaluates them on the held-out test set, with progress saved incrementally. |
| **Dataset loading workaround** | Loads `tner/bc5cdr` from its auto-converted Parquet branch to bypass the deprecated loading-script format. |
| **`Trainer` API version compatibility** | Tries the new `processing_class=` argument first, falls back to `tokenizer=`, so it runs correctly across `transformers` versions. |

---

## 📁 Repository Structure

```
pubmedbert-bc5cdr-peft-comparison/
│
├── README.md                              ← You are here
├── requirements.txt                       ← Pinned dependencies
├── LICENSE                                ← MIT License
│
├── notebooks/
│   └── PubMedBERT_BC5CDR_Capstone_Project.ipynb   ← Full pipeline (works end-to-end)
│
├── figures/
│   ├── Architecture_FINAL.svg             ← System architecture diagram
│   ├── fig1_data_size_vs_metrics.png      ← Data efficiency curves
│   ├── fig2_final_results_bar.png         ← F1/Precision/Recall/Accuracy bars
│   ├── fig3_efficiency.png                ← Time / VRAM / Param count
│   ├── fig4_loss_curves.png               ← Train & val loss per method
│   ├── fig5_radar.png                     ← Multi-metric radar chart
│   └── fig_per_entity_f1.png             ← Chemical vs Disease F1
│
├── results/
│   ├── table1_results.xls                 ← Headline comparison table
│   ├── table2_efficiency.xls              ← Efficiency metrics table
│   └── table_per_entity_breakdown.xls    ← Per-entity F1 breakdown
│
└── docs/
    └── PROJECT_BRIEF.md                   ← Full technical brief
```

---

## 🚀 How to Run

### Option A — Google Colab (Recommended, free GPU)

1. Upload `notebooks/PubMedBERT_BC5CDR_Capstone_Project.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Set **Runtime → Change runtime type → T4 GPU**
3. Run cells top-to-bottom. **Cell 1** installs all dependencies — **restart the kernel once** after first install.
4. **Cell 3** will prompt to mount Google Drive — approve it so results persist across sessions.
5. The 24-run training loop is **fully resumable** — if interrupted, re-run the training cell and it continues from the last checkpoint.
6. Once training finishes, run the **backfill cell** to compute held-out test-set scores for all 24 runs.
7. Remaining cells generate all figures and tables.

### Option B — Local Jupyter

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/pubmedbert-bc5cdr-peft-comparison.git
cd pubmedbert-bc5cdr-peft-comparison

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook notebooks/PubMedBERT_BC5CDR_Capstone_Project.ipynb
```

> **Hardware requirement:** A CUDA-capable GPU is required. The notebook was developed on an **RTX 3050 (4GB VRAM)**. QLoRA runs comfortably in under 0.5 GB VRAM — making it feasible on almost any modern GPU.

---

## 🧰 Tech Stack

| Category | Libraries |
|:---|:---|
| **Model / Training** | `transformers`, `peft`, `bitsandbytes`, `accelerate`, `torch` |
| **Data** | `datasets` (HuggingFace), `seqeval` |
| **Analysis** | `pandas`, `numpy`, `scikit-learn`, `scipy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Environment** | Jupyter / Google Colab |

---

## ⚠️ Limitations

- Only **2 random seeds** were used — sufficient to show consistent trends, but a thin basis for strong statistical significance claims. Disclosed honestly.
- Results are **BC5CDR-specific** — generalization to other biomedical NER datasets or base models is not directly tested.
- LoRA rank (`r=8`) and learning rates were chosen via reasonable defaults rather than exhaustive hyperparameter search.

---

## 📖 Evaluation Methodology

- **Validation set:** used *only* for early stopping and best-checkpoint selection (not for final reported scores)
- **Held-out test set:** evaluated separately — this is what's reported as the final/headline result
- **Metrics:** entity-level F1, Precision, Recall (`seqeval` — scores whole entity spans, not individual tokens), plus token-level Accuracy
- **Per-entity breakdown:** Chemical and Disease scored separately

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@misc{kiran2025pubmedbert,
  title   = {Data-Efficient and Resource-Aware Fine-Tuning of PubMedBERT for
             Biomedical Named Entity Recognition: A Comparative Study of
             Full Fine-Tuning, LoRA, and QLoRA on BC5CDR},
  author  = {Chigurupati, Venkat Sai Kiran},
  year    = {2025},
  note    = {Capstone Project, 25MAI1006},
  url     = {https://github.com/YOUR_USERNAME/pubmedbert-bc5cdr-peft-comparison}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ by **Chigurupati Venkat Sai Kiran**

*"The best model isn't the biggest — it's the one that fits your constraints."*

</div>
