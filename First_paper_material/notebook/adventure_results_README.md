# Adventure controlled experiment: completed sweep

Counts: 0, 2, 4, 6, 25, 50, 100, 200, 500, 1000, 2000, 2500, 3000, 3500.
53,424 original users; 9,415 books; 5,815,569 original ratings; 1,789 Adventure and 7,626 non-Adventure books.
Each synthetic user rates every Adventure book 5 and every other book 1. At 3500, the injected profiles add 32,952,500 ratings (85.0% of augmented observations), corresponding to 6.55% of original users.
Surprise biased SVD/NMF: 8 factors, 180 epochs, seed 42; identical common-entity initial factors within model.
No clipping of scores, factors, or injection range. Rankings use raw scores and unseen books.

SVD: mean Adventure books in Top-15: 3.4652 at baseline, 4.3265 at 3500; raw maximum 4.6435 at 100; maximum among checkpoints passing the factor-sign check 4.6435 at 100. No monotonicity assumed.
NMF: mean Adventure books in Top-15: 3.5889 at baseline, 12.7202 at 3500; raw maximum 14.2666 at 2500; maximum among checkpoints passing the factor-sign check 13.3176 at 500. No monotonicity assumed.

Non-negative factor check: NMF n=1000 (min pu=-628992.283529374, min qi=-88.89836111012345); NMF n=2000 (min pu=-1063882.181398028, min qi=-4.846196999408499); NMF n=2500 (min pu=-1.913146700784208e-06, min qi=1.0839319442998748e-15)

Interpretation and limits:
- Any NMF run with negative factors is a numerical sign-constraint failure, not evidence about correctly constrained NMF. Such runs are preserved and explicitly flagged; interpretation must distinguish them from valid checkpoints.
- All-training means weight observed rating pairs; adding dense fake profiles changes that mixture. Synthetic-only views isolate which records are measured, not a separate model.
- Red MAE measures reconstruction of training ratings; it is not held-out recommendation accuracy.
- Real-unseen components average available unseen candidates per user, then users. They describe score components; actual Top-K counts are separately computed from complete rankings.
- Within-user item ranking depends on book bias plus user–book interaction. Global mean and user bias are identical offsets across a user’s candidates.
- Component swaps are conditional replacements in fitted models, with a symmetric split of the interaction between replacements. They do not identify a unique causal training mechanism.
- Neighbors are the 20 closest other real users, compared with baseline. Recommendation genre counts are measured on the neighbors’ own current Top-15 lists, not actual human taste changes. Overlapping genre labels can sum above 15.
- Heatmap text gives absolute mean books. Color gives increase/decrease from baseline; non-target rows are visually muted only.
- Display titles are joined by book_id from the original Goodbooks books.csv, including titles missing from the cleaned original_title field. Genre labels and all numerical results remain those of the cleaned experiment data.
- Recommendation Jaccard is estimated using the same 10,000 randomly sampled distinct user pairs at every dose; it is not the exhaustive average over all pairs.
- UMAP is a joint projection of aligned before/after vectors at 3500, not evidence of causal movement.
- One controlled training seed; no universal ranking of algorithms is claimed.