# Layer 1: recommendation outcomes

- notebook/NMF_result_analysis_0420.ipynb and SVD_result_analysis_0420.ipynb: adapted to controlled results; the bias-analysis cells are executed through 3000 users with figures embedded in the existing notebooks.
- tables/controlled_recommendation_summary.csv: current top-20 target-genre counts, 40 conditions; extracted unchanged from the controlled score-analysis summary.
- result/summary/recommendation_target_share.png: current recommendation overview.
- result/NMF and result/SVD: destinations for notebook bar_figures and details_figures on execution.
- Existing result/nmf, result/svd and *_0420.py are historical Layer-1 artifacts, unchanged.

The notebooks retain the original mean-count metric and plotting code. Inputs now use current controlled top-20 counts and saved model parameters. User-bias genre groups use rated-book history, as in the original Layer-1 code. Synthetic users have dense ratings across all known genres.

Shared model arrays remain under ../layer2/embeddings. Neighbor analyses, UMAP, recommendation-score decomposition, cross-genre embedding analysis and sensitivity prediction are Layer-2 extensions and remain there. No training was performed during organization.

## Controlled bi / bu / mu update through 3000 users

All existing `result/nmf/{bi,bu,mio,merged}` and `result/svd/{bi,bu,mio,merged}` CSV files now use saved controlled models and the clean 9415-book catalog. No historical files outside these requested PerturbRec folders were removed. Baseline and all 12 injection sizes are covered for Drama and Thriller. Existing 2000-user CSV filenames contain three dose blocks (2000, 2500, 3000); always group by the explicit `n_fake` column rather than infer the dose from the filename. `injected_genre` identifies the scenario and `training_protocol` identifies the controlled source. No new CSV filenames or standalone bias PNGs were created in these result folders.

The existing NMF/SVD analysis notebooks contain ten embedded bias charts each (five metric families per injection genre). `bi` groups book biases by book tags; `bu` groups real-user biases by any genre in rated-book history, with synthetic users separate. History groups overlap. Dense fake users belong to every genre group, so synthetic-group curves coincide. `mio` contains observed genre-rating means, sample standard deviations, rating counts and the global training mean. `mu + bi` is only part of a predicted rating. `mu` and a user's `bu` are constant across that user's candidate books. Other-genre bu curves average genre-group means, matching the original notebook. All curves include baseline. No models were retrained for this extraction.
