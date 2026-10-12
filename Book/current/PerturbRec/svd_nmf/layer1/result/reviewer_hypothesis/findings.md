# Reviewer hypothesis — results, 2026-10-12

Forty saved controlled cases analyzed: NMF/SVD × Drama/Thriller × 2, 4, 6, 25, 50, 100, 200, 500, 1000, 2000 synthetic users. No retraining or checkpoint modification. Both paired baselines match exactly. All 40 recomputed Top-15 counts match the existing PerturbRec tables exactly. Top-25 and Top-35 decompositions are also computed.

## Verdict

The sign/bias hypothesis receives partial, descriptive support, but its claim that all downward compensation is forced into item bias is not supported. Signed SVD interactions contribute negative compensation in dense synthetic non-target pairs. In the NMF cases that retain nonnegative factors, interaction contributions cannot be negative and signed offsets support fitting low ratings; both user bias and item bias contribute. Reducing a positive interaction is also possible.

## Example: 500 fake users, Thriller injection, non-target books rated 1

| Model | Global mean | Fake-user bias | Item bias | Interaction | Raw prediction |
|---|---:|---:|---:|---:|---:|
| NMF | 2.917927 | −1.039074 | −1.120518 | 0.000075 | 0.758410 |
| SVD | 2.917927 | 0.003990 | 0.133206 | −2.060469 | 0.994654 |

Each row averages the same population of dense fake-user/non-target-book pairs. Components sum to the mean prediction. They do not express causal percentages. The average NMF prediction below 1 also shows imperfect fitting of these synthetic ratings; these are unclipped raw scores.

## Recommendation exposure versus fitting ratings

For real users, global mean and user bias cancel when ranking candidate books. We therefore separately measured item-bias and interaction differences, and swapped fitted components between baseline and augmented models. A symmetric two-component allocation assigns ranking nonlinearity across the two components; it is a conditional score decomposition, not a retraining intervention.

Example, NMF–Thriller with 500 synthetic users, Top-15: mean target count changes from 1.729859 to 10.839248 (+9.109389). The conditional item-bias allocation is +12.029041 and the embedding allocation is −2.919652. For NMF–Drama at 2,000 users, both allocations are positive (+5.625983 bias, +3.062584 embedding). Thus the component pattern depends on dose and target; there is no universal bias-only mechanism.

## Critical checkpoint audit: NMF sign constraint fails in two saved cases

NMF–Thriller at 1,000 and 2,000 synthetic users has mixed-sign raw factors:
- 1,000: 49.20% of user-factor entries and 49.03% of item-factor entries are negative.
- 2,000: 48.96% of user-factor entries and 47.37% of item-factor entries are negative.

These are not centered or UMAP representations. run_controlled.py saves model.pu and model.qi directly. Mixed signs occur within each dimension, and negative interaction scores occur as well; this is not merely a common per-dimension sign flip. Among fake/non-target pairs, interactions below −1e−10 occur in 14.36% and 25.78% of pairs, respectively. Their mean interactions are nevertheless near zero, so frequency must not be mistaken for contribution magnitude.

The installed Surprise 1.1.4 NMF code accumulates the biased prediction in multiplicative-update denominators without applying a nonnegative projection. This is a possible sign-loss pathway. The first offending epoch was not traced, so its exact trigger is not established. See implementation_audit.json for installed-source hash and line references.

These two cases cannot be used to support a mechanism premised on strictly nonnegative factors. Their raw exposure outputs are retained as diagnostics and flagged on the corresponding figure. A future constrained implementation or traced rerun is required before claiming these cases characterize genuine nonnegative factorization. No model or old result has been deleted or altered.

## Outputs

Each model/genre folder contains four CSV tables and one PNG four-panel figure. factor_constraint_audit.csv records all factor audits; validation.json contains the source-data hash, all cases, and cross-checks. Figures use linear axes. The earlier failed audit and successful resumed completion are preserved in layer1/log/reviewer_hypothesis.log.
