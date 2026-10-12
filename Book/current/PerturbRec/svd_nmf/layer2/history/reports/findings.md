# Personalization and concentration after injection

This analysis is about recommended titles, not observed changes in human taste. No retraining was performed.

## Results at 500 synthetic users

|Model|Target|Shared titles before → after (out of 20)|Distinct recommended titles before → after|Effective target catalog before → after|Different-history fixed-two target Jaccard before → after|
|---|---|---|---|---|---|
|NMF|Drama|1.737 → 3.563|1177 → 1068|68.9 → 29.1|5.56% → 22.63%|
|NMF|Thriller|1.737 → 8.749|1177 → 1326|26.7 → 22.9|10.29% → 39.42%|
|SVD|Drama|0.860 → 0.837|2579 → 3108|139.8 → 177.9|2.12% → 1.68%|
|SVD|Thriller|0.860 → 0.833|2579 → 3066|73.5 → 96.4|2.34% → 3.13%|

## Method and limits

- Catalog/exposure metrics cover every real user. Mean pair intersection is exact for all distinct user pairs, using item exposure frequencies. Jaccard metrics use the same 100,000 randomly selected distinct user pairs at every condition (sampling seed 20261012).
- History similarity uses cosine of original per-user genre frequencies, with each rated book allocating one unit fractionally across its tags. Lowest/highest quartiles define different/similar history groups; this is a history proxy, not measured human preferences.
- Raw ranking score is pu @ qi.T + bi. Global mean and user bias are omitted because they are constant across candidates for a user. Previously rated books are excluded; ties favor smaller book IDs. Recommendation target counts are validated exactly against the earlier cross-genre arrays.
- Effective target catalog = 1 / sum(p_i²), where p_i is a title’s fraction of all target recommendations. This normalizes exposure volume but is still affected by which users receive target recommendations.
- The fixed-two control uses the top-ranked TWO target books within each Top-20 list, only users who have at least two target books in every condition. The final figures use a cohort eligible in BOTH models, with a separate fixed sample of 100,000 pairs per genre; history strata keep the original global quartile thresholds. It fixes users, pair identities, and target list length, but selects a subset and cannot establish an effect for excluded users. It does not test a separately trained genre-share-matched intervention.
- Increased title overlap is evidence of output homogenization; it is not by itself proof of poorer relevance, utility, or human satisfaction. No held-out accuracy or behavioral outcomes are measured.
- This is one fixed-initialization training seed and one dense injection protocol (target 5, other books 1). Pair sample sizes do not constitute independent training repetitions; no training-level significance is claimed. Analysis is restricted to <=500.
- There is no expectation that every model or genre must move in the same direction. Report declines and null/mixed findings as well as increases.

The final results table and fixed-two figure use the cross-model shared cohort. Model-specific controls remain available separately. The Thriller shared cohort covers 3,815 real users (7.14%); Drama covers 31,048 (58.12%). These selected-cohort findings should not be generalized to all users.
