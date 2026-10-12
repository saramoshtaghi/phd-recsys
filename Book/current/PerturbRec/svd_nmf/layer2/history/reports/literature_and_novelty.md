# Research positioning: preliminary literature check

## Prior work already covers parts of this question

1. Chaney, Stewart, and Engelhardt, *How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility* (RecSys 2018). Their simulations examine feedback involving recommendation-confounded training data and resulting homogenization. This establishes that homogenization itself is not a new research topic. Our intervention adds synthetic profiles to fixed original histories rather than observing a recommendation-consumption feedback loop. [Author paper](https://arxiv.org/abs/1710.11214).

2. *An Empirical Analysis of Collaborative Recommender Systems Robustness to Shilling Attacks* (OHARS 2021). The paper evaluates genuine/spam-containing data using RMSE, NDCG, prediction shift and hit ratio, and also examines mainstream/non-mainstream user groups. Its findings caution that average metrics may hide subgroup effects. User-group analysis under profile manipulation is therefore not new by itself. Our candidate extension examines same-title recommendation overlap and fixed-size within-genre overlap, rather than those outcome measures. [Paper](https://ceur-ws.org/Vol-3012/OHARS2021-paper4.pdf).

3. Wang et al., *Poisoning Attacks against Recommender Systems: A Survey* (2024). Reviews attack methodologies and provides taxonomy and benchmarking resources. It is background for comparing injection protocols and choosing stronger controls; it does not establish novelty of our specific analysis. [Author survey](https://arxiv.org/abs/2401.01527).

## Candidate contribution — conditional on measured evidence

A controlled dose-response characterization of **title-level output homogenization under genre-oriented dense profile injection**, comparing NMF and SVD while separating genre-volume changes from overlap among a fixed number of target titles for the same real users.

The strongest evidence would be increased same-title overlap among users whose original histories differ, plus increased overlap in the fixed-two target control, and a distinct response pattern across the two models. Catalog concentration is corroborating evidence, not a substitute for the pair-level control.

If only full-list overlap grows while the fixed-two statistic stays unchanged or falls, report that the apparent homogenization may be explained by changes in genre allocation or item exposure without evidence of the proposed within-target convergence. If coverage expands, do not claim catalog diversity uniformly declines.

This is a preliminary source-grounded comparison, not an exhaustive systematic review or a verified claim of being first. Strengthening publication claims would require several training seeds, additional datasets/protocols, stronger genre-share-matched controls, and relevance/utility evaluation before calling the effect harmful.
