# Notebook

## `PubMedBERT_BC5CDR_Capstone_Project.ipynb`

This is the **complete, self-contained pipeline** for the capstone project. It runs end-to-end and produces all results, figures, and tables in the repository.

## Cell-by-Cell Overview

| Section | Description |
|:---|:---|
| **Cell 1 — Install** | Installs all dependencies (`transformers`, `peft`, `bitsandbytes`, `seqeval`, etc.). Restart kernel once after first install. |
| **Cell 2 — Imports** | Imports all libraries and sets global configuration constants. |
| **Cell 3 — Google Drive Mount** | Auto-detects Colab and mounts Google Drive. All checkpoints and results are saved there to survive session disconnects. |
| **Cell 4 — Dataset Loading** | Loads BC5CDR from HuggingFace via Parquet branch workaround. Inspects splits and label schema. |
| **Cell 5 — Preprocessing** | Tokenizes sentences with PubMedBERT tokenizer. Aligns BIO labels to subword tokens (first-subword labeling). |
| **Cell 6 — Model Factory** | Factory function that creates Full / LoRA / QLoRA model variants with consistent hyperparameters. Includes the `classifier`-exclusion fix for QLoRA. |
| **Cell 7 — Training Grid** | **Main training loop.** Runs all 24 combinations (3 methods × 4 sizes × 2 seeds). Fully resumable — re-run this cell after any interruption to continue from the last checkpoint. |
| **Cell 8 — Test Backfill** | Reloads all 24 trained models from disk (no retraining) and evaluates on the held-out test set. Run once after training is complete. |
| **Cell 9 — Results Assembly** | Aggregates all 24 runs into a master DataFrame. Computes mean ± std across seeds. |
| **Cell 10 — Figure 1** | Data-size curves (F1, Precision, Recall, Accuracy). |
| **Cell 11 — Figure 2** | Bar chart — full-data final results. |
| **Cell 12 — Figure 3** | Efficiency comparison (time, VRAM, params). |
| **Cell 13 — Figure 4** | Training & validation loss curves. |
| **Cell 14 — Figure 5** | Multi-metric radar chart. |
| **Cell 15 — Figure 6** | Per-entity-type F1 (Chemical vs Disease). |
| **Cell 16 — Tables** | Exports all results tables to XLS. |

## Estimated Runtimes (RTX 3050, 4GB VRAM)

| Method | Per run (full data) | All 24 runs |
|:---|:---:|:---:|
| Full FT | ~5–6 min | — |
| LoRA | ~8–10 min | — |
| QLoRA | ~12–14 min | — |
| **Total** | — | **~4–5 hours** |

> The loop is fully resumable, so you can run it across multiple Colab sessions.
