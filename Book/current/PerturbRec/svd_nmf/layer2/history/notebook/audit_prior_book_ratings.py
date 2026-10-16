"""Audit original observed ratings of the NMF Thriller titles, without retraining."""
from pathlib import Path
import argparse
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf/layer2'))
p.add_argument('--source',type=Path,default=Path('/home/moshtasa/Research/phd-recsys/Book/Archive/data/genre_clean.csv'))
a=p.parse_args();h=a.root/'history';t=h/'tables/title_mechanism'
lead=pd.read_csv(t/'nmf_leading_thriller_titles.csv');catalog=pd.read_csv(t/'nmf_book_scores.csv');parts=[]
for d in pd.read_csv(a.source,usecols=['book_id','rating'],chunksize=1000000):
 d['high']=d.rating>=4;d['low']=d.rating<=2
 parts.append(d.groupby('book_id').agg(original_rating_rows=('rating','size'),rating_sum=('rating','sum'),high_count=('high','sum'),low_count=('low','sum')))
s=pd.concat(parts).groupby(level=0).sum()
s['original_mean_rating']=s.rating_sum/s.original_rating_rows
s['original_high_rating_percent']=s.high_count/s.original_rating_rows*100
s['original_low_rating_percent']=s.low_count/s.original_rating_rows*100
v=catalog.merge(s.reset_index(),on='book_id',validate='one_to_one')
v['pooled_mean_if_500_ratings_of_5_added']=(v.rating_sum+2500)/(v.original_rating_rows+500)
mask=v.target
v.loc[mask,'original_rating_count_percentile_within_target']=v.loc[mask,'original_rating_rows'].rank(pct=True)*100
v.loc[mask,'original_mean_rating_percentile_within_target']=v.loc[mask,'original_mean_rating'].rank(pct=True)*100
v.to_csv(t/'nmf_original_ratings_and_after_scores.csv',index=False)
f=v[v.book_id.isin(lead.book_id)].sort_values('after_reach_percent',ascending=False)
f.to_csv(t/'nmf_leading_thriller_prior_ratings.csv',index=False)
lines=['# Original ratings versus recommendation exposure','',
'Observed rating counts, observed mean ratings, and recommendation frequency are different quantities. Zero baseline Top-20 exposure does not imply zero ratings or dislike. The pooled mean shown in the CSV is an arithmetic input-data diagnostic, not the NMF prediction.','']
for row in f.itertuples():
 lines += [f'## {row.title} (ID {row.book_id})',
 f'- Original rating rows: {row.original_rating_rows:,}; observed mean: {row.original_mean_rating:.3f}/5.',
 f'- Ratings 4–5: {row.original_high_rating_percent:.2f}%; ratings 1–2: {row.original_low_rating_percent:.2f}%.',
 f'- Relative to target books: count percentile {row.original_rating_count_percentile_within_target:.1f}; mean-rating percentile {row.original_mean_rating_percentile_within_target:.1f}.',
 f'- Real users receiving it in Top-20: {row.baseline_reach_percent:.4f}% before → {row.after_reach_percent:.2f}% after.',
 f'- Learned book bias: {row.bias_before:.4f} → {row.bias_after:.4f}; mean user–book dot score: {row.mean_dot_before:.4f} → {row.mean_dot_after:.4f}.','']
lines += ['These are descriptive measurements. A rating-data mean is not a recommendation score; item selection depends on the full learned ranking among unseen candidates. High recommendation reach after injection does not establish prior popularity or prior dislike.']
(h/'reports/prior_book_ratings.md').write_text('\n'.join(lines))
print('\n'.join(lines))
