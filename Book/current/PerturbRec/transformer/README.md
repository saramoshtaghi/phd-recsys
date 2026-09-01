# PerturbRec — Model 2: Transformer Recommender (Retrained)

A transformer trained fresh on each injection (ORIGINAL + N fake users), apples-to-apples with SVD/NMF. Each retrained model has its own embedding space, so Layer 2 builds a self-contained KNN per model and only scalar fractions are compared across models.

## Structure

```
transformer/
├── config.py              # shared (Model 3 imports data.py)
├── data.py                # shared (Model 3 imports this)
├── model.py               # shared model definition
├── README.md              # this file
├── result/
│   └── embeddings/        # 21 retrained .pt files (shared input to both layers)
├── layer1/
│   ├── notebook.ipynb       # per-genre item/user centroid shift
│   └── result/             # Layer 1 outputs (CSV + figures)
└── layer2/
    ├── notebook.ipynb       # KNN-shift (K=15): infiltration + displacement
    └── result/             # Layer 2 outputs (CSV + figures)
```

## Two layers

| Layer | Question | Tool | Metric |
|-------|-----------|------|--------|
| Layer 1 | Did the numbers move? | Centroid distance | item / user centroid shift per genre |
| Layer 2 | Did identity flip? | KNN-shift (K=15) | infiltration (fake users) + displacement (real users) |

## Notebooks

- **`layer1/notebook.ipynb`** — per-genre item/user embedding centroid shift from ORIGINAL to each injection. Analogous to SVD's bi/bu shift. Output: `layer1/result/centroid_shift.csv` + 2 figures.
- **`layer2/notebook.ipynb`** — KNN-neighborhood-shift (K=15) with self-contained KNN per model. Measures infiltration (fake users) and displacement (real users vs ORIGINAL baseline). Output: `layer2/result/knn_shift_retrained_summary.csv` + 2 figures.

## Run

```bash
jupyter nbconvert --to notebook --execute layer1/notebook.ipynb
jupyter nbconvert --to notebook --execute layer2/notebook.ipynb
```

## Retraining methodology (for reproducibility)

Train a fresh transformer on each dataset (ORIGINAL + each injection file) via MLM self-supervision: mask 15% of ratings, predict the true rating bin. 15 epochs, AdamW, lr=1e-4. Save user (CLS) + item embeddings per model as `.pt` in `result/embeddings/`. Each model's embedding space is independent; only scalar measurements are compared across models.
