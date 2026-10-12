from pathlib import Path
import sys
import numpy as np
import pandas as pd
H=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf/layer2/history')
sys.path.insert(0,str(H/'notebook'))
import analyze_personalization as m
base,seen,genres=m.load_data('nmf');ids=base['ids'];items=base['items'];a,b=m.pairs(len(ids))
history=pd.read_csv(H/'tables/nmf_fixed_user_pairs.csv.gz');assert np.array_equal(history.user_a,ids[a]);assert np.array_equal(history.user_b,ids[b])
cos=history.history_cosine.to_numpy();q1,q3=np.quantile(cos,[.25,.75]);rows=[]
allgenres=sorted({g for gs in genres.values() for g in gs})
membership=np.array([[g in genres[int(i)] for g in allgenres] for i in items],dtype=float)
profiles=np.asarray(seen@(membership/membership.sum(axis=1)[:,None]));profiles/=np.linalg.norm(profiles,axis=1)[:,None]
for genre in ['Drama','Thriller']:
 target=np.array([genre in genres[int(i)] for i in items]);common=np.ones(len(ids),bool);fixed={}
 for model in ['nmf','svd']:
  for n in [0]+m.COUNTS:
   f=H/'data'/model/('baseline_top20.npz' if n==0 else f'{genre.lower()}_{n}_top20.npz');z=np.load(f);assert np.array_equal(z['ids'],ids) and np.array_equal(z['items'],items)
   fixed[model,n],ok=m.fixed_target(z['top'],target);common &=ok
 eligible=np.flatnonzero(common);ca,cb=m.pairs(len(eligible));aa,bb=eligible[ca],eligible[cb]
 hc=(profiles[aa]*profiles[bb]).sum(axis=1)
 strata={'all_pairs':np.ones(len(aa),bool),'different_histories':hc<=q1,'similar_histories':hc>=q3}
 pd.DataFrame({'user_a':ids[aa],'user_b':ids[bb],'history_cosine':hc}).to_csv(H/'tables'/f'cross_model_{genre.lower()}_fixed_pairs.csv.gz',index=False)
 for model in ['nmf','svd']:
  for n in [0]+m.COUNTS:
   for label,keep in strata.items():
    mask=keep;assert mask.any();ss=m.intersection(fixed[model,n],aa[mask],bb[mask])
    rows.append(dict(model=model,genre=genre,n_fake=n,stratum=label,n_eligible_users=int(common.sum()),eligible_users_percent=float(common.mean()*100),n_pairs=int(mask.sum()),fixed_target_books_per_user=2,mean_shared_target_books=float(ss.mean()),mean_target_jaccard_percent=float((ss/(4-ss)).mean()*100)))
pd.DataFrame(rows).to_csv(H/'tables/cross_model_fixed_target_control.csv',index=False)
m.plot()
print(pd.DataFrame(rows).query('n_fake in [0,500] and stratum == "different_histories"').to_string(index=False))
