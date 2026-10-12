# Reviewer hypothesis: sign constraint and bias compensation

Saved controlled SVD/NMF checkpoints are analyzed without retraining. Original
real users/items are aligned by raw IDs; fake users are identified as IDs absent
from the original data. Unknown labels are excluded by requiring the clean catalog.

1. Dense synthetic target-5 and other-1 pairs: signed interaction scores, biases,
   offset, prediction, residual, and fraction with a negative interaction.
2. Original observed pairs: exact before/after prediction decomposition into
   global mean, user bias, item bias, and interaction. Low ratings (<=2) on
   non-target items are separately reported, without treating them as a new test set.
3. Unseen candidate pairs: each user's mean target-vs-non-target raw score margin
   is split into item-bias and interaction parts. Mean/user weighting is used.
4. Top-15/25/35 after excluding previously rated items: baseline, fully augmented,
   augmented embeddings with baseline item bias, and baseline embeddings with
   augmented item bias. Sorted raw book IDs provide stable tie resolution.
   Symmetric two-component allocation sums to the total exposure change, including
   nonlinearity; it is not a causal proof because components were jointly trained.

IMPORTANT: factor_constraint_audit.csv checks actual raw checkpoint factors.
Some biased Surprise NMF runs can violate nonnegativity. Such cases are retained
as raw checkpoint diagnostics, flagged explicitly, and must not be interpreted as
valid evidence about a strictly nonnegative factorization. Saved arrays are never
clipped or changed.

NMF's intended nonnegative interaction is an algebraic constraint, not empirical confirmation
of the reviewer's full explanation. Evidence for bias compensation requires actual
bias/offset changes alongside low-rating fit. A decrease in a positive interaction
can also lower predictions. No component percentage is reported when opposite signs
or a near-zero net change would make that percentage misleading. Larger SVD negative
interaction usage and stronger NMF bias compensation would support, but not prove,
the proposed mechanism. A trained-model constraint/optimizer ablation is needed for
exclusive causal attribution. Raw predictions are not clipped before decomposition.

Charts use linear axes and PNG output only; counts stop at 2,000.
