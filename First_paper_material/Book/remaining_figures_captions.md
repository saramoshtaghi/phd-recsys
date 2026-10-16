# Figure captions

## dataset_overview.png
Original dataset characteristics: (a) distinct user and item IDs on a logarithmic scale, and (b) the empirical distribution of observed ratings. Counts refer to the full rating catalog. The movie catalog includes one item with an unknown release decade, which is omitted from the category-distribution figure.

## book_thriller_bi_updated.png
Mean learned book bias under Thriller injection for NMF and SVD. Coloured lines show the target genre and grey lines show individual non-target genre means. Books with multiple labels can contribute to multiple genre means. Injection counts range from baseline to 500 users.

## score_offsets_three_domains.png
Global mean plus category-mean item bias under injection targeting Thriller books, Shopping applications and movies released in the 1990s. NMF and SVD occupy the top and bottom rows. Coloured lines show the target category, grey lines show individual non-target categories, and dashed lines show the global mean. Book injections extend to 500 users, mobile injections to 300, and movie injections to 50. Mobile parameter curves use the original app categories. The global mean is a common offset within each model and does not directly reorder a user's candidate items.
