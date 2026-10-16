# Top-35 target exposure and union-based repetition

## Reading the figures

- Bar height: mean target-genre count per real user (out of 35). The label above is this mean and its percentage of 35.
- Three values below a bar, top to bottom: distinct target titles (union U), union-based repeated-occurrence percentage, and the percentage of real users receiving the most frequent target title in that condition.
- Repeat is 100 × (T − U) / T, where T is all target recommendation occurrences over all real users. This metric approaches 100% even at baseline in a large user population. It is NOT a pairwise overlap/Jaccard statistic and is not evidence of homogenization by itself.
- Reach is 100 × max_i frequency_i / N. It differs from the most frequent book’s share of all target recommendation slots, which is separately retained in the CSV.
- Source factors are reused; no training. Same real users, original unseen-book eligibility, raw dot-product + item-bias rankings, book-ID-stable ties.
- Hatched NMF Thriller bars at 1000, 2000, 2500 and 3000 reproduce historical checkpoint outputs with negative factors. Do not treat them as validated nonnegative NMF results.

## Validation

- 52 cases matched available earlier Top-15 means exactly.
- 16 cases matched available earlier extension Top-35 means exactly.
- Each user receives exactly 35 distinct unseen titles; frequency totals equal 53,424 × 35.
- All 52 model/genre/dose combinations are present.

## Reproduce

`python notebook/build_top35_target_union.py` then `python notebook/plot_top35_target_union.py --metric union`. Alternative annotations `--metric top1` or `--metric top10` are available; all metrics are already stored.