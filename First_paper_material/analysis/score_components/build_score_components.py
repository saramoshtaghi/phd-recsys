"""Exact score components from saved checkpoints; no retraining or clipping."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,importlib.util
import numpy as np,pandas as pd
R=Path('/home/moshtasa/Research/phd-recsys');L=R/'Book/current/PerturbRec/svd_nmf';OUT=R/'First_paper_material/analysis/score_components';OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('old',L/'layer1/notebook/analyze_reviewer_hypothesis.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
df=pd.read_csv(R/'Book/Archive/data/genre_clean.csv',usecols=['user_id','book_id','rating','genres'])
users=np.sort(df.user_id.unique());items=np.sort(df.book_id.unique());u=np.searchsorted(users,df.user_id.to_numpy());i=np.searchsorted(items,df.book_id.to_numpy());ratings=df.rating.to_numpy(float)
tags=dict(df[['book_id','genres']].drop_duplicates('book_id').itertuples(index=False,name=None));target=np.array(['Thriller' in {x.strip() for x in str(tags[b]).split(',')} for b in items]);rows=[]
for model in ['svd','nmf']:
 for n in [0,2,4,6,25,50,100,200,500]:
  path=L/'layer2/embeddings/pilot'/f'{model}_baseline_a.npz' if n==0 else L/'layer2/embeddings/controlled'/model/'thriller'/f'{model}_thriller_{n}.npz'
  with np.load(path) as z:m=old.align(z,users,items)
  gi,bu,bi=old.observed_components(m,u,i)
  for label,mask in [('target',target),('non_target',~target)]:
   obs=mask[i];cnt=np.bincount(u[obs],minlength=len(users));eligible=mask.sum()-cnt;ok=eligible>0
   # Sum all candidate terms and subtract previously observed books per user.
   assert len(df[['user_id','book_id']].drop_duplicates())==len(df)
   dots=m['p']@m['q'][mask].sum(0)-np.bincount(u[obs],weights=gi[obs],minlength=len(users))
   bs=m['bi'][mask].sum()-np.bincount(u[obs],weights=bi[obs],minlength=len(users))
   vals={'mean_interaction':float(np.mean(dots[ok]/eligible[ok])),'mean_bi':float(np.mean(bs[ok]/eligible[ok])),'mean_bu':float(m['bu'][ok].mean()),'mu':m['mu']}
   vals['mean_prediction']=sum(vals.values());rows.append(dict(model=model,n_fake=n,group=label,view='real_unseen_prediction',n_users=int(ok.sum()),n_pairs=int(eligible.sum()),weighting='equal user mean over all unrated books in group',**vals))
   vals={'mean_interaction':float(gi[obs].mean()),'mean_bi':float(bi[obs].mean()),'mean_bu':float(bu[obs].mean()),'mu':m['mu']};vals['mean_prediction']=sum(vals.values())
   rows.append(dict(model=model,n_fake=n,group=label,view='real_observed_training',n_users=int(np.count_nonzero(cnt)),n_pairs=int(obs.sum()),weighting='equal observed rating pair',mean_recorded_rating=float(ratings[obs].mean()),**vals))
   if n:
    vals={'mean_interaction':float(m['fp'].mean(0)@m['q'][mask].mean(0)),'mean_bi':float(m['bi'][mask].mean()),'mean_bu':float(m['fbu'].mean()),'mu':m['mu']};vals['mean_prediction']=sum(vals.values())
    rows.append(dict(model=model,n_fake=n,group=label,view='synthetic_training',n_users=n,n_pairs=int(n*mask.sum()),weighting='equal dense synthetic rating pair',mean_recorded_rating=5 if label=='target' else 1,**vals))
  print(model,n,flush=True)
pd.DataFrame(rows).to_csv(OUT/'components.csv',index=False)
(OUT/'validation.json').write_text(json.dumps({'no_retraining':True,'raw_scores':True,'genre':'Thriller','n_real_users':len(users),'n_books':len(items),'target_books':int(target.sum()),'source_data':str(R/'Book/Archive/data/genre_clean.csv'),'notes':'Prediction uses all unseen books, not just top-K; training views measure fitted endpoints, not epoch trajectories.'},indent=2))
