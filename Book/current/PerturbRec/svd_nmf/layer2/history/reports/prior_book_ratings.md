# Original ratings versus recommendation exposure

Observed rating counts, observed mean ratings, and recommendation frequency are different quantities. Zero baseline Top-20 exposure does not imply zero ratings or dislike. The pooled mean shown in the CSV is an arithmetic input-data diagnostic, not the NMF prediction.

## Flickan som lekte med elden (ID 98)
- Original rating rows: 7,563; observed mean: 4.101/5.
- Ratings 4–5: 78.82%; ratings 1–2: 3.68%.
- Relative to target books: count percentile 99.7; mean-rating percentile 81.2.
- Real users receiving it in Top-20: 0.0000% before → 76.99% after.
- Learned book bias: -0.5682 → 2.8867; mean user–book dot score: 1.2780 → 0.5910.

## Luftslottet som sprängdes (ID 140)
- Original rating rows: 6,702; observed mean: 4.077/5.
- Ratings 4–5: 77.28%; ratings 1–2: 4.52%.
- Relative to target books: count percentile 99.6; mean-rating percentile 77.8.
- Real users receiving it in Top-20: 0.0000% before → 75.72% after.
- Learned book bias: -0.5678 → 2.8824; mean user–book dot score: 1.2531 → 0.5269.

## The Stand (ID 168)
- Original rating rows: 5,973; observed mean: 4.256/5.
- Ratings 4–5: 81.10%; ratings 1–2: 4.65%.
- Relative to target books: count percentile 99.3; mean-rating percentile 94.3.
- Real users receiving it in Top-20: 0.0000% before → 72.84% after.
- Learned book bias: -0.0258 → 2.9016; mean user–book dot score: 0.8834 → 0.3915.

## The Da Vinci Code (ID 26)
- Original rating rows: 13,072; observed mean: 3.621/5.
- Ratings 4–5: 58.89%; ratings 1–2: 15.24%.
- Relative to target books: count percentile 99.9; mean-rating percentile 14.3.
- Real users receiving it in Top-20: 0.0019% before → 72.83% after.
- Learned book bias: -1.0670 → 2.8469; mean user–book dot score: 1.4021 → 0.9688.

## Män som hatar kvinnor (ID 16)
- Original rating rows: 14,382; observed mean: 3.995/5.
- Ratings 4–5: 74.00%; ratings 1–2: 8.69%.
- Relative to target books: count percentile 99.9; mean-rating percentile 64.6.
- Real users receiving it in Top-20: 0.0000% before → 69.92% after.
- Learned book bias: -0.3922 → 2.8788; mean user–book dot score: 0.9761 → 0.8385.

## The Catcher in the Rye (ID 8)
- Original rating rows: 14,472; observed mean: 3.710/5.
- Ratings 4–5: 61.64%; ratings 1–2: 14.86%.
- Relative to target books: count percentile 100.0; mean-rating percentile 23.5.
- Real users receiving it in Top-20: 0.0000% before → 69.35% after.
- Learned book bias: -0.6062 → 2.8487; mean user–book dot score: 0.8380 → 0.9296.

## The Shining (ID 72)
- Original rating rows: 6,861; observed mean: 4.120/5.
- Ratings 4–5: 77.83%; ratings 1–2: 4.53%.
- Relative to target books: count percentile 99.6; mean-rating percentile 83.5.
- Real users receiving it in Top-20: 0.0000% before → 69.34% after.
- Learned book bias: -0.0533 → 2.9074; mean user–book dot score: 0.7118 → 0.3460.

## Gone Girl (ID 30)
- Original rating rows: 12,105; observed mean: 3.907/5.
- Ratings 4–5: 71.65%; ratings 1–2: 8.01%.
- Relative to target books: count percentile 99.8; mean-rating percentile 50.7.
- Real users receiving it in Top-20: 0.0000% before → 68.98% after.
- Learned book bias: -0.1826 → 2.8914; mean user–book dot score: 0.7196 → 0.5370.

## Jurassic Park (ID 84)
- Original rating rows: 6,601; observed mean: 3.915/5.
- Ratings 4–5: 68.78%; ratings 1–2: 6.17%.
- Relative to target books: count percentile 99.5; mean-rating percentile 51.9.
- Real users receiving it in Top-20: 0.0000% before → 66.88% after.
- Learned book bias: -0.3269 → 2.9090; mean user–book dot score: 1.0188 → 0.2988.

## The Joy Luck Club (ID 118)
- Original rating rows: 6,212; observed mean: 3.881/5.
- Ratings 4–5: 67.76%; ratings 1–2: 7.36%.
- Relative to target books: count percentile 99.4; mean-rating percentile 46.7.
- Real users receiving it in Top-20: 0.0000% before → 65.92% after.
- Learned book bias: -0.6786 → 2.8891; mean user–book dot score: 1.2247 → 0.2973.

These are descriptive measurements. A rating-data mean is not a recommendation score; item selection depends on the full learned ranking among unseen candidates. High recommendation reach after injection does not establish prior popularity or prior dislike.