# What drives shared Thriller titles at 500 injected users?

## Four score combinations

Baseline = G0 + bi0; embedding only = G1 + bi0; item bias only = G0 + bi1; full after = G1 + bi1. G = pu @ qi.T. The global mean and user bias cancel within each user’s raw-score ranking. Previously rated books are excluded.

| model | condition | n_fake | fixed_target_pair_count | fixed_target_jaccard_percent | exact_mean_shared_titles | mean_target_count | target_effective_catalog | target_top10_exposure_percent | catalog_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nmf | baseline | 0 | 26148 | 10.293967671204935 | 1.7371628406166213 | 2.3117512728361787 | 26.67625073698805 | 47.99640494562885 | 1177 |
| nmf | embedding_only | 500 | 26148 | 70.09713936056293 | 9.5834863043348 | 0.0 | nan | nan | 788 |
| nmf | item_bias_only | 500 | 26148 | 8.68390189179542 | 5.114007831577096 | 19.92759808325846 | 77.65536975381288 | 25.011083850266573 | 496 |
| nmf | full_after | 500 | 26148 | 39.41665391871908 | 8.748740942038278 | 13.9995882000599 | 22.90519744146299 | 50.62787967600553 | 1326 |
| svd | baseline | 0 | 26148 | 2.339248381010657 | 0.8601799280384382 | 2.1617437855645405 | 73.54245250151496 | 27.173150689676078 | 2579 |
| svd | embedding_only | 500 | 26148 | 2.575085411248789 | 1.2542809960838694 | 0.5451482479784366 | 75.12366920578091 | 26.5657189946436 | 2632 |
| svd | item_bias_only | 500 | 26148 | 2.5087960838305032 | 0.910561899842054 | 9.1480607966457 | 153.77239575619194 | 15.825022609805902 | 2557 |
| svd | full_after | 500 | 26148 | 3.1296211309979087 | 0.8325063747412489 | 3.047937256663672 | 96.3542387636602 | 22.46227730251239 | 3066 |

## Symmetric allocation of metric changes

| model | metric | embedding_allocation | item_bias_allocation | total_change |
| --- | --- | --- | --- | --- |
| nmf | exact_mean_shared_titles | 5.74052828708968 | 1.2710498143319766 | 7.011578101421657 |
| nmf | fixed_target_jaccard_percent | 45.26796185814082 | -16.145275610626687 | 29.122686247514142 |
| nmf | mean_target_count | -4.119880578017369 | 15.80771750524109 | 11.68783692722372 |
| svd | exact_mean_shared_titles | 0.158022771472313 | -0.1856963247695023 | -0.0276735532971893 |
| svd | fixed_target_jaccard_percent | 0.4283310387027685 | 0.3620417112844829 | 0.7903727499872515 |
| svd | mean_target_count | -3.858359538784067 | 4.744553009883198 | 0.8861934710991313 |

## Personalized score heterogeneity within Thriller

| model | condition | target_personalized_score_rms | target_common_score_std | common_to_personalized_ratio |
| --- | --- | --- | --- | --- |
| nmf | baseline | 0.3132001818066026 | 0.2642056823357728 | 0.8435681001581179 |
| nmf | full_after | 0.043274992887062 | 0.3651653139701644 | 8.438252431910584 |
| svd | baseline | 0.5173426121229399 | 0.2840467707601414 | 0.5490496319151869 |
| svd | full_after | 0.4879776196162649 | 0.2755639589968718 | 0.5647061420840763 |

Within-target personalized score RMS is the root mean square of the double-centered interaction matrix: sqrt(trace(Cov(pu) Cov(qi_target))). It removes user-constant and item-common score terms and is invariant to equivalent factor reparameterization. Common-score spread is std_i(mean_u G_ui + bi_i) within target books. These are diagnostics of score variation, not measured preference or accuracy.

## Limits

- Hybrid scores are not retrained models. Symmetric allocations sum exactly to the observed output-metric change but are not estimates of causal effects of changing the training algorithm.
- Bias-only and embedding-only combinations are descriptive substitutions at the saved fitted parameters. They can be implausible standalone training outcomes.
- The target control fixes original pairs and exactly two target titles. In hybrid conditions those titles are selected from all unseen Thriller books, even if fewer than two appear in a hybrid Top-20; the baseline/full conditions exactly reproduce the original within-Top-20 control.
- Relative book-score decomposition centers each user’s score by their mean score over all unseen candidates. Raw score increases alone do not establish rank improvements; observed recommendation reach and the hybrid rerankings provide separate evidence.
- One training seed and one dense target=5 / other=1 protocol; only Thriller at 500 evaluated here. No additional training was run.

Dense intervention volume: 500 synthetic profiles × 9,415 known books = 4,707,500 injected rating rows, versus 5,815,569 original rows (80.95%). This is a large rating-volume intervention despite a small user-count fraction; findings do not imply the same effect for 500 sparse profiles.
