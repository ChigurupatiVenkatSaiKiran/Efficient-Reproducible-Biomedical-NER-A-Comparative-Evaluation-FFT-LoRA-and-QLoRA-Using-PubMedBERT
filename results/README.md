# Results Tables

This directory contains the quantitative output tables from the 24-run experiment.

## Files

| File | Description |
|:---|:---|
| `table1_results.xls` | Headline comparison — Test F1, Precision, Recall, Accuracy for all 3 methods at all 4 data sizes, averaged over 2 seeds |
| `table2_efficiency.xls` | Efficiency metrics — training time, peak VRAM, trainable parameter count, total parameters, throughput (samples/sec) |
| `table_per_entity_breakdown.xls` | Per-entity-type F1 — Chemical vs Disease, all 3 methods, full training data |

## How Results Were Generated

All results come from the 24-run grid:
```
3 methods × 4 data sizes × 2 seeds = 24 runs
```

- Validation set → used only for early stopping and checkpoint selection
- **Held-out test set → what is reported in all tables** (evaluated once after training via the backfill cell)
- Metric library: `seqeval` (entity-level span F1) + token-level accuracy

## Raw Results Cache

The full results cache (`all_results.pkl`) is **not committed to git** due to file size.
It is stored in Google Drive when running on Colab and can be reconstructed by running the full notebook.
