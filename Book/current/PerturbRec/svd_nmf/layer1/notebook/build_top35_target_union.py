import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import ast,json
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
R=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf');L=R/'layer1';E=R/'layer2/embeddings'
OUT=L/'tables/top35_target_union';OUT.mkdir(parents=True,exist_ok=True)
SRC=Path('/home/moshtasa/Research/phd-recsys/Book/Archive/data/genre_clean.csv');COUNTS=[0,2,4,6,25,50,100,200,500,1000,2000,2500,3000];K=35

def tags(v):
 try:
  x=ast.literal_eval(str(v))
  if isinstance(x,(list,tuple)):return [str(g).strip() for g in x if str(g).strip().lower()!='unknown']
 except (ValueError,SyntaxError):pass
 return [g.strip() for g in str(v).replace('|',',').split(',') if g.strip().lower()!='unknown']

def top35(score,seen):
 s=score.copy();assert np.isfinite(s).all();s[seen]=-np.inf;assert ((~seen).sum(axis=1)>=K).all()
 ix=np.argpartition(-s,K-1,axis=1)[:,:K];cut=np.take_along_axis(s,ix,axis=1).min(axis=1)
 for row in np.flatnonzero((s>=cut[:,None]).sum(axis=1)>K):
  candidates=np.flatnonzero(s[row]>=cut[row]);ix[row]=candidates[np.lexsort((candidates,-s[row,candidates]))[:K]]
 vals=np.take_along_axis(s,ix,axis=1);order=np.lexsort((ix,-vals),axis=1);ix=np.take_along_axis(ix,order,axis=1)
 assert not np.take_along_axis(seen,ix,axis=1).any();return ix

def worker(model):
 df=pd.read_csv(SRC,usecols=['user_id','book_id','genres','original_title']);base=np.load(E/'pilot'/f'{model}_baseline_a.npz');ids=np.sort(base['ids']);items=np.sort(base['items']);u=np.searchsorted(ids,df.user_id.to_numpy());i=np.searchsorted(items,df.book_id.to_numpy());assert np.array_equal(ids[u],df.user_id) and np.array_equal(items[i],df.book_id)
 seen=csr_matrix((np.ones(len(df),bool),(u,i)),shape=(len(ids),len(items)));meta=df.drop_duplicates('book_id').set_index('book_id').loc[items];N=len(ids);summary=[];validation=[];baseline_freq=None;baseline15={}
 references={g:pd.read_csv(L/'tables'/f'controlled_{g.lower()}_top15.csv') for g in ['Drama','Thriller']};ext=pd.read_csv(L/'tables/extension_2500_3000_recommendation_counts.csv')
 for genre in ['Drama','Thriller']:
  target=np.array([genre in tags(v) for v in meta.genres]);folder=OUT/model/genre;folder.mkdir(parents=True,exist_ok=True)
  for n in COUNTS:
   path=folder/f'{n}_book_frequencies.csv';recfile=folder/f'{n}_summary.json'
   if path.exists() and recfile.exists():summary.append(json.loads(recfile.read_text()));continue
   z=base if n==0 else np.load(E/'controlled'/model/genre.lower()/f'{model}_{genre.lower()}_{n}.npz')
   ui=np.argsort(z['ids']);ui=ui[np.searchsorted(z['ids'][ui],ids)];ii=np.argsort(z['items']);assert np.array_equal(z['ids'][ui],ids) and np.array_equal(z['items'][ii],items)
   pu,qi,bi=z['pu'][ui],z['qi'][ii],z['bi'][ii];pu_min=float(pu.min());qi_min=float(qi.min());valid_factors=(model!='nmf' or min(pu_min,qi_min)>=-1e-12)
   if n==0 and baseline_freq is not None:freq=baseline_freq.copy();total15=baseline15[genre]
   else:
    freq=np.zeros(len(items),dtype=np.int64);total15=0
    # Also collect both target counts for the shared baseline to avoid recomputing it.
    baseline_masks={g:np.array([g in tags(v) for v in meta.genres]) for g in ['Drama','Thriller']} if n==0 else {}
    for start in range(0,N,256):
     sl=slice(start,min(start+256,N));ix=top35(pu[sl]@qi.T+bi,seen[sl].toarray());freq+=np.bincount(ix.ravel(),minlength=len(items));total15+=int(target[ix[:,:15]].sum())
     for g,mask in baseline_masks.items():baseline15[g]=baseline15.get(g,0)+int(mask[ix[:,:15]].sum())
    if n==0:baseline_freq=freq.copy()
   assert freq.sum()==N*K and freq.max()<=N
   T=int(freq[target].sum());U=int((freq[target]>0).sum());most=int(np.flatnonzero(target)[np.argmax(freq[target])]);mean=T/N
   ref=references[genre];ref=ref[(ref.model==model)&(ref.n_fake==n)]
   if len(ref):assert np.isclose(total15/N,ref.mean_target_count.iloc[0],atol=1e-9)
   old35=ext[(ext.model==model)&(ext.genre==genre)&(ext.n_fake==n)&(ext.k==35)]
   if len(old35):assert np.isclose(mean,old35.mean_target_count.iloc[0],atol=1e-9)
   f=freq[target];rec=dict(model=model,genre=genre,n_fake=n,k=K,n_users=N,mean_target_count=mean,target_share_of_top35_percent=100*T/(N*K),total_target_recommendation_slots=T,union_distinct_target_titles=U,union_repeat_percent=100*(1-U/T) if T else None,most_frequent_book_id=int(items[most]),most_frequent_book_title=str(meta.original_title.iloc[most]),most_frequent_book_users=int(freq[most]),most_frequent_book_user_reach_percent=100*freq[most]/N,most_frequent_book_share_of_target_slots_percent=100*freq[most]/T if T else None,top10_share_of_target_slots_percent=100*np.sort(f)[-10:].sum()/T if T else None,exact_mean_target_titles_shared_by_two_users=float((f.astype(float)*(f-1)).sum()/(N*(N-1))),pu_min=pu_min,qi_min=qi_min,nmf_nonnegative_constraint_ok=valid_factors,top15_reference_matched=bool(len(ref)),top35_reference_matched=bool(len(old35)))
   pd.DataFrame({'book_id':items,'title':meta.original_title.to_numpy(),'is_target':target,'n_real_users_recommended':freq}).to_csv(path,index=False);recfile.write_text(json.dumps(rec,indent=2));summary.append(rec)
   print(model,genre,n,'mean35',round(mean,4),'union',U,'repeat',round(rec['union_repeat_percent'],4),'maxshare',round(rec['most_frequent_book_share_of_target_slots_percent'],3),'valid',valid_factors,flush=True)
  pd.DataFrame([x for x in summary if x['genre']==genre]).to_csv(folder/'summary.csv',index=False)
 pd.DataFrame(summary).to_csv(OUT/f'{model}_summary.csv',index=False)
 return model

if __name__=='__main__':
 rng=np.random.default_rng(42);s=rng.integers(-3,4,(10,70)).astype(float);mask=rng.random(s.shape)<.1;x=s.copy();x[mask]=-np.inf;assert np.array_equal(top35(s,mask),np.argsort(-x,axis=1,kind='stable')[:,:35])
 with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(worker,['nmf','svd']))
 data=pd.concat([pd.read_csv(OUT/f'{m}_summary.csv') for m in ['nmf','svd']]);assert len(data)==52;data.to_csv(OUT/'summary.csv',index=False)
 print('ALL TOP35 METRICS COMPLETE',flush=True)
