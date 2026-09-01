# PerturbRec

## Perturbation-based Behavioral Validation for Recommender Systems

### The problem

Standard recommender evaluation checks one thing: did the model predict the right rating / suggest the right item? That's a **metric** check (RMSE, Recall@k, NDCG).

It misses a second layer: **did the user's identity actually flip?**

If you inject a few fake users who love drama, a real romance-fan's "people like you" neighborhood can switch to drama-fans — even when the rating predictions barely change. The system looks fine on paper but is behaviorally unstable.

Nobody in recsys checks this second layer.

### Why it matters

**1. Manipulation.** If 100 fake accounts can shift a real user's taste neighborhood, anyone can buy that — a publisher pumps in fake 5-star ratings for a book, and suddenly it's recommended to people whose "similar readers" never actually liked it. The system is gamed, and the metrics say nothing happened.

**2. Trust.** "People like you also bought..." is the core of how these systems work. If that identity is fragile, the recommendation isn't based on understanding you — it's based on whatever noise was last injected. Users feel this (recommendations suddenly feel off) but platforms can't see it because their dashboards only show RMSE going down.

**3. Safety for AI deployment.** We're moving toward LLM-based recommenders being deployed at scale. If we don't have a way to test "is this model's identity structure stable under pressure" *before* shipping, we're releasing models whose failure mode is invisible to standard eval. Layer-2 is that missing safety check.

Short version: you can't fix or regulate what you can't measure, and right now nobody measures behavioral robustness.

### The idea

Borrow a validation method from single-cell biology: **poke the system, see if identity flips.**

- In biology: knock out a gene, check if a sick cell's nearest neighbors become healthy cells.
- In recsys: inject fake users, check if a real user's nearest-neighbor taste cluster reclassifies.

Same logic, different domain. No biology required in the recsys paper — just the methodology.

### Two layers of measurement

| Layer | What it measures | How |
|-------|-----------------|-----|
| **Layer 1 — Metric** (existing) | Did the numbers move? | Delta b_i, Delta b_u, Delta Recall@k, Delta NDCG |
| **Layer 2 — Behavioral** (new) | Did identity flip? | `knn_shift`: fraction of K-nearest-neighbors whose genre-label reclassifies after perturbation |

**The finding:** these two layers dissociate. A model can have great metrics and a flipping identity, or vice versa. Layer 2 exposes a failure mode Layer 1 can't see.

### Three models, one poke

All three trained on the same book data, hit with the same perturbation suite, validated with the same KNN-shift test.

1. **SVD / NMF** (classical) — already built (`SVD_0420.py`, `NMF_0420.py`). User embeddings = `model.pu`.
2. **Transformer recommender** (LLM-style) — a sequence transformer with MLM + user-embedding head. Items = tokens, ratings = values, user = "sentence." Trained from scratch on book data.
3. **KG-augmented transformer** — model #2 plus an LLM-extracted book knowledge graph (GPT reads synopses -> extracts book relationships -> graph fed as regularization / extra context).

### The perturbation suite

- **User injection:** fake biasing users who rate a target genre highly (the existing `f_<genre>_<n>u_pos5_neg1_*.csv` sweep, Drama x {2,4,6,25,50,...,2000} users).
- **Item knockout (optional add-on):** delete a user's ratings on a genre — the symmetric analog of gene knockout.

### The expected result

- Layer 1 (metrics) will look similar across the three models.
- Layer 2 (`knn_shift`) will differ — and model #3 (KG-augmented) should be the most behaviorally stable, because the knowledge graph provides real-world structure that fake users can't easily break.

That dissociation is the paper's contribution. The LLM knowledge graph earns its spot by improving the *behavioral* layer, not the metric layer.

### What gets reused vs what's new

**Reuse as-is:**
- SVD/NMF injection pipeline -> already done.
- KNN-shift validation logic (`query_neighbors` + `knn_shift` + scatter plot) -> model-agnostic, works on any embedding.
- LLM knowledge-graph extraction pipeline (GPT reads text -> structured relationships -> graph) -> swap PubMed abstracts for book synopses.

**New work:**
- Train transformer recommender on book data (real but bounded — the training pattern is familiar).
- Extract book KG from synopses (a few hundred texts through the existing prompt).
- Wire KNN-shift onto each model's user embeddings (small).
- Run experiments + write.

### Where it publishes

- **RecSys / CIKM** — core recsys + validation framework.
- **KDD / TOIS / TKDE** — journal extension with the KG + cross-domain angle.

### One-line thesis framing

Perturbation-based behavioral validation for latent-factor models: a transferable methodology that exposes identity-instability standard metrics miss, demonstrated in recommender systems with LLM-derived knowledge graphs as robustness-inducing side information.
