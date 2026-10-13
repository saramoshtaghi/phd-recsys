# Layer 2 — Recommendation homogenization

Question: Does dense synthetic-user injection change only the recommended genre mix, or also make different users receive the same book titles?

## Scope

- Saved controlled NMF/SVD models; Drama/Thriller; baseline and 2, 4, 6, 25, 50, 100, 200, 500 injected users.
- Original real users only; Top-20 unseen books, raw ranking scores, stable ties by book ID.
- No new training. Unknown genre omitted by the established clean-data loader.
- All output is under this directory. Existing Layer 1/2 figures are untouched.

## Directory structure

- `notebook/analyze_personalization.py`: reproducible analysis.
- `data/<model>/*_top20.npz`: compact arrays of actual book indices, with user/item ID mappings.
- `tables/catalog_metrics.csv`: all-user coverage, concentration, exact pair overlap.
- `tables/pair_metrics.csv`: sampled-pair Jaccard and history strata.
- `tables/fixed_target_control.csv`: count-controlled target-only analysis with model-specific eligible cohorts.
- `tables/cross_model_fixed_target_control.csv`: primary comparison using the same eligible real users and pairs across both models and every dose.
- `notebook/match_model_cohorts.py`: compute the shared cohort and redraw the final figures.
- `tables/<model>_fixed_user_pairs.csv.gz`: the fixed pairs and original history similarity.
- `result/<genre>/personalization_6panel.png`: widescreen model comparisons.
- `reports/findings.md`: measured findings and limitations.
- `reports/literature_and_novelty.md`: research positioning, with sources.
- `log/manifest.json`, `log/analysis.log`: validation/status and execution log.

## Reading the measures

A user receives a set R_u of 20 titles. Shared titles for two users = |R_u intersection R_v|. Jaccard = intersection / union. Same-title overlap is distinct from merely recommending books with the same genre.

Let c_i be the number of real users recommended book i and N the real-user count. Exact mean shared titles across all unordered distinct user pairs = sum_i c_i(c_i-1) / [N(N-1)]. This avoids enumerating over one billion pairs.

Coverage is the number of distinct titles recommended to any user. Effective catalog = 1 / sum_i p_i², with p_i = c_i / sum_j c_j. It distinguishes evenly spread exposure from repeatedly recommending a few titles. Target-only effective catalog uses just target-tagged titles.

Pair metrics use the same 100,000 sampled pairs at each condition. Baseline history cosine is calculated from per-user fractional genre frequencies of previously rated books. Lowest-quartile pairs are relatively different-history users; highest-quartile pairs relatively similar-history users. These are descriptive proxies, not true preference measurements.

**Count control:** retain the same users who have at least TWO target titles in all nine conditions in BOTH models. The primary control and final figures use this cross-model shared cohort; model-specific exploratory controls are also retained. Compare their top-ranked TWO target titles, using the same pairs. If this overlap rises, simply recommending more target titles cannot explain the increase in this particular fixed-size statistic. This is a selected cohort, not a causal intervention or a result for every user.

No accuracy/utility degradation, multi-seed stability, or first-ever research novelty should be claimed from this experiment alone.

## Reproduce

Run `python notebook/analyze_personalization.py`, then `python notebook/match_model_cohorts.py`. The second command computes the same eligible cohort across both models, samples 100,000 pairs within it for each genre, and updates the final plots/report.

See `reports/summary_fa.md` for the Persian explanation and `log/validation.json` for validation.

## Score mechanism follow-up (Thriller, 500)

See `reports/title_mechanism_fa.md` and `reports/title_mechanism.md`; `tables/title_mechanism` contains full per-title scores, component allocations and invariant score heterogeneity. Figures are in `result/Thriller/title_mechanism/{NMF,SVD}/mechanism_4panel.png`. Reproduce with `python notebook/analyze_title_mechanism.py`. This script reads saved factors directly and supports the reorganized materials layout.

Hybrid recommendation arrays are stored separately in `data/title_mechanism`, not in the tables directory.
