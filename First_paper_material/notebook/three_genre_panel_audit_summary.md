# Full audit of the six figures in first_reviewer

Scope: thriller, advanture and drama; two PNGs in each folder; baseline and 2, 4, 6, 25, 50, 100, 200 and 500 injected users.

54 genre/model/dose cases were independently recalculated from saved raw checkpoints. All original observed ratings and all injected full-catalog ratings were scored. Unseen-book component means were independently computed by aggregating candidate vectors using sparse histories before the dot product. Complete unseen lists for 12 fixed users per case/group were also checked.

Result: zero numeric mismatches. Maximum absolute difference: 5.329070518200751e-15. All six PNGs matched fresh renders pixel for pixel, with zero changed pixels. All six existing PNGs have distinct hashes.

Only the baseline checkpoint is intentionally shared among genres within each model. No augmented checkpoint duplicates another genre. The target catalogs contain 1,606 Thriller, 1,789 Adventure and 3,006 Drama books.

Why some values appear identical:
- Baseline global mean is the same original-data mean, 3.918111.
- Global mean is identical between SVD and NMF trained on the same augmented data.
- Every synthetic profile gives target books 5 and non-target books 1, for all three genres. Well-fitted SVD synthetic predictions therefore approach the same training targets.
- The two candidate groups share the user-bias term for the same users. This is part of the score formula.
- Figure axes use comparable scales, and nearby means may coincide after rounding.

NMF, 500 injected users, target books:

All-user training:
- Thriller: 4.305413745 = mu 2.917926795 + user bias -0.889880019 + book bias 2.215841974 + dot product 0.061524995
- Adventure: 4.454587883 = mu 2.952707523 + user bias -0.854296258 + book bias 2.186285639 + dot product 0.169890980
- Drama: 4.270832323 = mu 3.184008867 + user bias -0.841479435 + book bias 0.663386323 + dot product 1.264916569

Synthetic-only training:
- Thriller: 3.864462194 = mu 2.917926795 + user bias -1.039073691 + book bias 1.985607003 + dot product 0.000002087
- Adventure: 3.810202390 = mu 2.952707523 + user bias -1.018538840 + book bias 1.876023196 + dot product 0.000010511
- Drama: 4.782086238 = mu 3.184008867 + user bias -1.180299004 + book bias 0.764879675 + dot product 2.013496701

Prediction on real users and all unrated target books:
- Thriller: 4.267422526 = mu 2.917926795 + user bias -0.703152838 + book bias 1.981184218 + dot product 0.071464351
- Adventure: 4.268625144 = mu 2.952707523 + user bias -0.696480217 + book bias 1.868975512 + dot product 0.143422326
- Drama: 3.931484429 = mu 3.184008867 + user bias -0.572810859 + book bias 0.767057692 + dot product 0.553228731

For real-unseen prediction, Thriller and Adventure round to 4.27 at two decimal places despite different component values. Drama rounds to 3.93. These figures measure mean raw fitted/predicted scores and components, rather than the number of genre books in Top-15 recommendations.

No figures or model parameters were changed during the audit; reports and reproducible validation code were added.