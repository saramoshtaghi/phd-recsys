# PerturbRec — Model 3: Transformer + Knowledge Graph (Retrained)

Same transformer as Model 2, but trained with a **knowledge-graph regularization loss** that pulls structurally-related books together in the item-embedding space. The KG is built from book metadata (deterministic: author/genre/decade). This is the "does LLM-extracted structure help?" row of the 2-layer × 3-model table.

## Structure

```
transformer_kg/
├── config_kg.py              # shared config (KG_LAMBDA, etc.)
├── build_kg.py               # deterministic KG builder
├── build_kg_llm.py           # LLM KG builder (optional, needs api_key.txt)
├── merge_kg.py               # merge deterministic + LLM KGs
├── model_kg.py               # shared KG-augmented model
├── train_kg.py               # training script (retrains per injection) — remove after run completes
├── run_all_kg.py             # orchestrator — remove after run completes
├── README.md
├── kg/                      # KG edge files
│   ├── book_kg_deterministic.csv
│   └── book_kg_merged.csv
├── result/
│   └── embeddings/          # 21 KG-augmented .pt files (shared input to both layers)
├── layer1/
│   ├── notebook.ipynb          # per-genre item/user centroid shift
│   └── result/                # Layer 1 outputs
└── layer2/
    ├── notebook.ipynb          # KNN-shift (K=15): infiltration + displacement
    └── result/                # Layer 2 outputs
```

## Two layers

| Layer | Question | Tool | Metric |
|-------|-----------|------|--------|
| Layer 1 | Did the numbers move? | Centroid distance | item / user centroid shift per genre |
| Layer 2 | Did identity flip? | KNN-shift (K=15) | infiltration (fake users) + displacement (real users) |

## The KG loss

```
total_loss = MLM_loss + KG_LAMBDA * KG_loss
KG_loss   = mean over sampled edges (i,j) of (1 - cosine_sim(item_emb[i], item_emb[j]))
```

Books connected in the KG (same author / same genre / same decade) get pulled together. Hypothesis: the KG-regularized model resists infiltration/displacement better than Models 1 and 2.

## Notebooks

- **`layer1/notebook.ipynb`** — per-genre item/user embedding centroid shift from ORIGINAL to each injection. Tests whether KG regularization makes item embeddings more stable. Output: `layer1/result/centroid_shift.csv` + 2 figures.
- **`layer2/notebook.ipynb`** — KNN-neighborhood-shift (K=15), self-contained KNN per model. Measures infiltration (fake users) and displacement (real users vs ORIGINAL baseline). Output: `layer2/result/knn_shift_kg_summary.csv` + 2 figures.

## Run

```bash
jupyter nbconvert --to notebook --execute layer1/notebook.ipynb
jupyter nbconvert --to notebook --execute layer2/notebook.ipynb
```

## Retraining methodology (for reproducibility)

Train a fresh KG-augmented transformer on each dataset (ORIGINAL + each injection file) via MLM + KG regularization. 15 epochs, AdamW, lr=1e-4, KG_LAMBDA=0.1. Save user (CLS) + item embeddings per model as `.pt` in `result/embeddings/`. Each model's embedding space is independent; only scalar measurements are compared across models.

Note: `train_kg.py` and `run_all_kg.py` remain in the folder until the ongoing training run completes, then they will be removed to match Model 2's clean layout.
