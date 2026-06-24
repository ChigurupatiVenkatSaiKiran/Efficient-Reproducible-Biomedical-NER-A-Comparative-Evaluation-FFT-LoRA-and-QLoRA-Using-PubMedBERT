# Contributing

Thank you for your interest in this project! This is a capstone research project by **Chigurupati Venkat Sai Kiran**.

## How You Can Contribute

### 🐛 Bug Reports
If you find a bug in the notebook (e.g., a cell that fails on a specific `transformers` version), please [open an issue](../../issues) with:
- Your Python / `transformers` / `peft` / `bitsandbytes` versions
- The full error traceback
- Whether you are running locally or on Colab

### 💡 Feature Suggestions
Potential extensions worth exploring:
- Additional base models (BioBERT, ClinicalBERT, GatorTron)
- Additional datasets (NCBI Disease, BioNLP shared task corpora)
- Higher LoRA ranks (r=16, r=32) or alternative target modules
- QLoRA with 8-bit quantization comparison
- More seeds (5–10) for stronger statistical confidence

### 📊 Reproducing Results
If you reproduce results on different hardware or a different dataset, please open an issue or PR sharing your numbers — cross-hardware reproducibility data is genuinely useful.

## Code Style
- The main notebook is the single source of truth — keep it runnable top-to-bottom
- Add a comment whenever a non-obvious workaround is required (see the existing engineering notes in the notebook)

## Contact
Open a GitHub Issue for any questions. This is an academic project, not a production codebase.
