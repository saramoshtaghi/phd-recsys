#!/usr/bin/env python3
"""Independent, full-record audit of all data feeding the six displayed reviewer figures."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from PIL import Image
R=Path('/home/moshtasa/Research/phd-recsys');NB=R/'First_paper_material/notebook';ROOT=R/'Book/current/PerturbRec/svd_nmf';E=ROOT/'layer2/embeddings';COUNTS=[0,2,4,6,25,50,100,200,500];GENRES=['Thriller','Adventure','Drama'];KEYS=['mu','mean_bu','mean_bi','mean_interaction','mean_prediction'];OUTPUT=NB/'three_genre_panel_audit.json'

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def checkpoint(genre,model,n):
 if n==0:return E/'pilot'/f'{model}_baseline_a.npz'
 return E/'controlled'/model/f'{model}_adventure_{n}.npz' if genre=='Adventure' else E/'controlled'/model/genre.lower()/f'{model}_{genre.lower()}_{n}.npz'
def accumulator():return dict(count=0,bu=0.,bi=0.,dot=0.,pred=0.,actual=0.,abs_error=0.)
def observed(bu,bi,p,q,mu,target):
 out=[accumulator(),accumulator()]
 for start in range(0,len(RATINGS),120000):
  stop=min(start+120000,len(RATINGS));ui=USER_ROW[start:stop];ii=ITEM_ROW[start:stop];dots=np.sum(p[ui]*q[ii],axis=1);ub=bu[ui];ib=bi[ii];pr=mu+ub+ib+dots;rr=RATINGS[start:stop]
  for j,mask in enumerate([target[ii],~target[ii]]):
   a=out[j];a['count']+=int(mask.sum());a['bu']+=float(ub[mask].sum());a['bi']+=float(ib[mask].sum());a['dot']+=float(dots[mask].sum());a['pred']+=float(pr[mask].sum());a['actual']+=float(rr[mask].sum());a['abs_error']+=float(np.abs(rr[mask]-pr[mask]).sum())
 return out

def synthetic(fbu,bi,fp,q,mu,target):
 out=[accumulator(),accumulator()]
 for start in range(0,len(fp),32):
  stop=min(start+32,len(fp));scores=fp[start:stop]@q.T;pr=scores+mu+fbu[start:stop,None]+bi[None,:]
  for j,mask in enumerate([target,~target]):
   a=out[j];rr=5 if j==0 else 1;count=(stop-start)*int(mask.sum());a['count']+=count;a['bu']+=float(fbu[start:stop].sum()*mask.sum());a['bi']+=float(bi[mask].sum()*(stop-start));a['dot']+=float(scores[:,mask].sum());a['pred']+=float(pr[:,mask].sum());a['actual']+=rr*count;a['abs_error']+=float(np.abs(pr[:,mask]-rr).sum())
 return out

def row_from_sums(a,mu):
 c=a['count'];assert c>0
 return dict(n_pairs=c,mu=mu,mean_bu=a['bu']/c,mean_bi=a['bi']/c,mean_interaction=a['dot']/c,mean_prediction=a['pred']/c,mae=a['abs_error']/c,mean_recorded_rating=a['actual']/c)

def candidate_means(bu,bi,p,q,mu,target):
 out=[];checks=[]
 for group,mask in [('target',target),('non_target',~target)]:
  history=SEEN[:,mask];num=mask.sum()-np.asarray(history.sum(1)).ravel();ok=num>0
  # Independently sum candidate VECTORS before taking dot products, instead of
  # summing observed pair interaction scores as in the figure-building code.
  available_q=q[mask].sum(0)[None,:]-history.astype(float)@q[mask]
  available_bias=bi[mask].sum()-np.asarray(history.astype(float)@bi[mask]).ravel()
  per_bi=available_bias[ok]/num[ok];per_dot=np.sum(p[ok]*available_q[ok],axis=1)/num[ok];per_bu=bu[ok];per_pred=mu+per_bu+per_bi+per_dot
  out.append(dict(group=group,view='real_unseen_prediction',n_users=int(ok.sum()),n_pairs=int(num.sum()),mu=mu,mean_bu=float(per_bu.mean()),mean_bi=float(per_bi.mean()),mean_interaction=float(per_dot.mean()),mean_prediction=float(per_pred.mean())))
  for user in np.random.RandomState(20261026).choice(np.flatnonzero(ok),12,replace=False):
   unseen=mask&(~SEEN[user].toarray().ravel());raw=mu+bu[user]+bi[unseen]+q[unseen]@p[user];indirect=mu+bu[user]+available_bias[user]/num[user]+p[user]@available_q[user]/num[user];assert np.isclose(raw.mean(),indirect,atol=1e-10,rtol=1e-11)
  checks.append(dict(group=group,direct_all_unrated_books_checked_for_users=12))
 return out,checks

def audit(job):
 genre,model,n=job;path=checkpoint(genre,model,n)
 with np.load(path) as z:
  ui=pd.Index(z['ids']).get_indexer(USERS);ii=pd.Index(z['items']).get_indexer(ITEMS);assert (ui>=0).all() and (ii>=0).all();fake=~pd.Index(z['ids']).isin(USERS);assert int(fake.sum())==n and len(z['ids'])==len(USERS)+n
  p=z['pu'][ui];q=z['qi'][ii];bu=z['bu'][ui];bi=z['bi'][ii];fp=z['pu'][fake];fbu=z['bu'][fake];mu=float(z['mu']);fingerprint={k:digest(z[k]) for k in ['pu','qi','bu','bi','mu']}
 target=TARGETS[genre];expected_mu=(REAL_SUM+n*(5*int(target.sum())+int((~target).sum())))/(len(RATINGS)+n*len(ITEMS));assert np.isclose(mu,expected_mu,rtol=0,atol=1e-12)
 real=observed(bu,bi,p,q,mu,target);fake_sums=synthetic(fbu,bi,fp,q,mu,target);rows=[]
 for j,group in enumerate(['target','non_target']):
  rows.append(dict(group=group,view='real_observed_training',**row_from_sums(real[j],mu)))
  if n:rows.append(dict(group=group,view='synthetic_training',**row_from_sums(fake_sums[j],mu)))
  combined={k:real[j][k]+fake_sums[j][k] for k in real[j]};rows.append(dict(group=group,view='all_training',**row_from_sums(combined,mu)))
 unseen,checks=candidate_means(bu,bi,p,q,mu,target);rows+=unseen
 maximum=0.;mismatches=[];comparison=[]
 for row in rows:
  expected=STORED[genre];selected=expected[(expected.model==model)&(expected.n_fake==n)&(expected.group==row['group'])&(expected.view==row['view'])];assert len(selected)==1,(genre,model,n,row['group'],row['view'],len(selected));s=selected.iloc[0]
  assert np.isclose(row['mean_prediction'],sum(row[k] for k in ['mu','mean_bu','mean_bi','mean_interaction']),atol=1e-10)
  for key,value in row.items():
   if key in ['group','view'] or key not in s or pd.isna(s[key]):continue
   if key=='n_users' and pd.isna(s[key]):continue
   delta=abs(float(s[key])-float(value));maximum=max(maximum,delta)
   if not np.isclose(float(s[key]),float(value),rtol=2e-11,atol=1e-10):mismatches.append(dict(view=row['view'],group=row['group'],field=key,stored=float(s[key]),recomputed=float(value),delta=delta))
  comparison.append(dict(genre=genre,model=model,n_fake=n,**row))
 print('AUDITED',genre,model,n,'max abs difference',maximum,'mismatches',len(mismatches),flush=True)
 return dict(genre=genre,model=model,n_fake=n,checkpoint=str(path),array_sha256=fingerprint,global_mean_formula_matches=True,all_original_rating_pairs_checked=len(RATINGS),synthetic_pairs_checked=n*len(ITEMS),unseen_candidate_checks=checks,max_absolute_difference=maximum,mismatches=mismatches,rows=comparison)

def stored_tables():
 out={}
 out['Adventure']=pd.read_csv(ROOT/'layer1/tables/adventure_score_components.csv');out['Drama']=pd.read_csv(NB/'drama_reviewer_score_components.csv');thriller=pd.read_csv(NB/'score_components.csv');training=pd.read_csv(NB/'training_score_components.csv');thriller=pd.concat([thriller,training[training.view=='all_training']],ignore_index=True);err=pd.read_csv(NB/'training_all_users_mae.csv');thriller['mae']=np.nan
 for i,row in thriller.iterrows():
  if row['view']=='real_unseen_prediction':continue
  e=err[(err.model==row['model'])&(err.n_fake==row['n_fake'])&(err.group==row['group'])];assert len(e)==1;key={'all_training':'all_mae','real_observed_training':'real_mae','synthetic_training':'synthetic_mae'}[row['view']];thriller.at[i,'mae']=float(e.iloc[0][key])
 out['Thriller']=thriller
 for genre in out:out[genre]=out[genre][out[genre].n_fake.isin(COUNTS)]
 return out

if __name__=='__main__':
 tick=time.time();df=pd.read_csv(R/'Book/Archive/data/genre_clean.csv',usecols=['user_id','book_id','rating','genres']);USERS=np.sort(df.user_id.unique());ITEMS=np.sort(df.book_id.unique());USER_ROW=pd.Index(USERS).get_indexer(df.user_id);ITEM_ROW=pd.Index(ITEMS).get_indexer(df.book_id);RATINGS=df.rating.to_numpy(float);REAL_SUM=float(RATINGS.sum());SEEN=csr_matrix((np.ones(len(df),bool),(USER_ROW,ITEM_ROW)),shape=(len(USERS),len(ITEMS)));assert SEEN.nnz==len(df)==5815569;meta=df.drop_duplicates('book_id').set_index('book_id').loc[ITEMS];tags=[{g.strip() for g in str(v).split(',')} for v in meta.genres];TARGETS={g:np.array([g in t for t in tags]) for g in GENRES};STORED=stored_tables()
 with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(audit,[(g,m,n) for g in GENRES for m in ['svd','nmf'] for n in COUNTS]))
 mismatches=[dict(genre=r['genre'],model=r['model'],n_fake=r['n_fake'],**v) for r in results for v in r['mismatches']];allrows=[v for r in results for v in r['rows']];pd.DataFrame(allrows).to_csv(NB/'three_genre_panel_independent_components.csv',index=False)
 duplicates=[]
 for model in ['svd','nmf']:
  for n in COUNTS:
   cases=[r for r in results if r['model']==model and r['n_fake']==n]
   for j in range(3):
    for k in range(j):
     same=cases[j]['array_sha256']==cases[k]['array_sha256']
     if same:duplicates.append(dict(model=model,n_fake=n,genres=[cases[j]['genre'],cases[k]['genre']],expected_shared_baseline=n==0))
 images=[]
 for folder in ['thriller','advanture','drama']:
  paths=list((R/'First_paper_material/figures/first_reviewer'/folder).glob('*.png'));assert len(paths)==2
  for path in paths:
   with Image.open(path) as im:im.verify()
   images.append(dict(folder=folder,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 assert len(set(p['sha256'] for p in images))==6
 manifests=json.loads((NB/'genre_reviewer_figures_manifest.json').read_text());source_checks={v['genre']:hashlib.sha256(Path(v['source']).read_bytes()).hexdigest()==v['source_sha256'] for v in manifests['figures']};assert all(source_checks.values())
 out=dict(status='passed' if not mismatches and all(v['expected_shared_baseline'] for v in duplicates) else 'failed',counts=COUNTS,genres=GENRES,cases=len(results),target_book_counts={g:int(v.sum()) for g,v in TARGETS.items()},cases_detail=results,mismatches=mismatches,identical_checkpoint_pairs=duplicates,images=images,image_data_source_hash_checks=source_checks,all_images_distinct=True,independent_method='Observed ratings scored in original-row blocks; synthetic full-catalog ratings scored in separate dense blocks; unseen-candidate vectors summed using CSR histories BEFORE the dot product, followed by per-user averaging; 12 direct complete-unseen-list checks for each model/dose/genre/group.',no_retraining=True,raw_scores=True,elapsed_seconds=time.time()-tick,finished=time.time());OUTPUT.write_text(json.dumps(out,indent=2,allow_nan=False));print('AUDIT FINISHED',out['status'],'cases',len(results),'mismatches',len(mismatches),'seconds',round(time.time()-tick),flush=True)
 if out['status']!='passed':raise SystemExit(1)
