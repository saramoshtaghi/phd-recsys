"""Recompute target exposure and title unions at K=15,25,35 from the same rankings."""
from build_top35_target_union import *
MULTI=L/'tables/topk_target_union';MULTI.mkdir(parents=True,exist_ok=True)
KS=[15,25,35]

def run_model(job):
 model,partition=job
 df=pd.read_csv(SRC,usecols=['user_id','book_id','genres','original_title'])
 base=np.load(E/'pilot'/f'{model}_baseline_a.npz');ids=np.sort(base['ids']);items=np.sort(base['items']);N=len(ids)
 u=np.searchsorted(ids,df.user_id.to_numpy());i=np.searchsorted(items,df.book_id.to_numpy())
 assert np.array_equal(ids[u],df.user_id) and np.array_equal(items[i],df.book_id)
 seen=csr_matrix((np.ones(len(df),bool),(u,i)),shape=(N,len(items)))
 meta=df.drop_duplicates('book_id').set_index('book_id').loc[items]
 masks={g:np.array([g in tags(v) for v in meta.genres]) for g in ['Drama','Thriller']}
 old=pd.read_csv(OUT/'summary.csv');rows=[];baseline=None
 for genre in ['Drama','Thriller']:
  folder=MULTI/model/genre;folder.mkdir(parents=True,exist_ok=True)
  refs=pd.read_csv(L/'tables'/f'controlled_{genre.lower()}_top15.csv')
  for n in COUNTS[partition::4]:
   cache=folder/f'{n}_summary.json'
   if cache.exists():rows.extend(json.loads(cache.read_text()));continue
   z=base if n==0 else np.load(E/'controlled'/model/genre.lower()/f'{model}_{genre.lower()}_{n}.npz')
   ui=np.argsort(z['ids']);ui=ui[np.searchsorted(z['ids'][ui],ids)];ii=np.argsort(z['items'])
   assert np.array_equal(z['ids'][ui],ids) and np.array_equal(z['items'][ii],items)
   pu,qi,bi=z['pu'][ui],z['qi'][ii],z['bi'][ii]
   valid=model!='nmf' or min(pu.min(),qi.min())>=-1e-12
   if n==0 and baseline is not None:freqs=baseline
   else:
    freqs={k:np.zeros(len(items),np.int64) for k in KS}
    for start in range(0,N,256):
     sl=slice(start,min(N,start+256));ix=top35(pu[sl]@qi.T+bi,seen[sl].toarray())
     for k in KS:freqs[k]+=np.bincount(ix[:,:k].ravel(),minlength=len(items))
    if n==0:baseline=freqs
   records=[];target=masks[genre]
   for k in KS:
    freq=freqs[k];assert freq.sum()==N*k and freq.max()<=N
    f=freq[target];T=int(f.sum());U=int((f>0).sum());most=int(np.flatnonzero(target)[np.argmax(f)])
    rec=dict(model=model,genre=genre,n_fake=n,k=k,n_users=N,mean_target_count=T/N,target_share_percent=100*T/(N*k),total_target_recommendation_slots=T,union_distinct_target_titles=U,union_repeat_percent=100*(1-U/T) if T else None,most_frequent_book_id=int(items[most]),most_frequent_book_title=str(meta.original_title.iloc[most]),most_frequent_book_users=int(freq[most]),most_frequent_book_user_reach_percent=100*freq[most]/N,nmf_nonnegative_constraint_ok=bool(valid))
    if k==35:
     r=old[(old.model==model)&(old.genre==genre)&(old.n_fake==n)].iloc[0]
     assert T==r.total_target_recommendation_slots and U==r.union_distinct_target_titles
    if k==15:
     r=refs[(refs.model==model)&(refs.n_fake==n)]
     if len(r):assert np.isclose(T/N,r.mean_target_count.iloc[0],atol=1e-9)
    records.append(rec)
   cache.write_text(json.dumps(records,indent=2));rows.extend(records)
   print(model,genre,n,'K15/25/35',[round(x['mean_target_count'],3) for x in records],flush=True)
 pd.DataFrame(rows).to_csv(MULTI/f'{model}_{partition}_summary.csv',index=False)

if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=8) as pool:list(pool.map(run_model,[(m,p) for m in ['nmf','svd'] for p in range(4)]))
 d=pd.concat([pd.read_csv(MULTI/f'{m}_{p}_summary.csv') for m in ['nmf','svd'] for p in range(4)]);assert len(d)==156
 d.to_csv(MULTI/'summary.csv',index=False)
 print('ALL K15/25/35 METRICS COMPLETE',flush=True)
