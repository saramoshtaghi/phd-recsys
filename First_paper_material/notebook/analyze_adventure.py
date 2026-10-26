#!/usr/bin/env python3
"""Full-range Adventure evaluation on saved raw checkpoints, no retraining/clipping."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from scipy.linalg import orthogonal_procrustes
R=Path('/home/moshtasa/Research/phd-recsys');P=R/'Book/current/PerturbRec';ROOT=P/'svd_nmf';L=ROOT/'layer2';NB=R/'First_paper_material/notebook';T=ROOT/'layer1/tables';LT=L/'materials/tables';SRC=R/'Book/Archive/data/genre_clean.csv'
sys.path.insert(0,str(L/'materials/notebook'));from embedding_utils import neighbors
COUNTS=[0,2,4,6,25,50,100,200,500,1000,2000,2500,3000,3500];KS=[15,20,25,35]
def write_json(path,data):
 tmp=Path(str(path)+'.partial');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)
def checkpoint(model,n):return L/'embeddings/pilot'/f'{model}_baseline_a.npz' if not n else L/'embeddings/controlled'/model/f'{model}_adventure_{n}.npz'
def cachepath(model,n):return L/'embeddings/controlled'/model/f'{model}_adventure_{n}_analysis.npz'
def resultpath(model,n):return NB/f'adventure_{model}_{n}_analysis.json'
def align(z):
 ids=z['ids'];order=np.argsort(ids);ui=order[np.searchsorted(ids[order],USERS)];ii=np.argsort(z['items']);assert np.array_equal(ids[ui],USERS) and np.array_equal(z['items'][ii],ITEMS)
 fake=np.flatnonzero(~np.isin(ids,USERS))
 return dict(p=z['pu'][ui].astype(float),q=z['qi'][ii].astype(float),bu=z['bu'][ui].astype(float),bi=z['bi'][ii].astype(float),mu=float(z['mu']),fp=z['pu'][fake].astype(float),fbu=z['bu'][fake].astype(float))
def top(score,seen):
 score=score.copy();score[seen]=-np.inf;k=35
 part=np.argpartition(-score,k-1,axis=1)[:,:k];threshold=np.take_along_axis(score,part,axis=1).min(1)
 higher=score>threshold[:,None];tie=score==threshold[:,None];need=k-higher.sum(1);chosen=higher|(tie&(tie.cumsum(1)<=need[:,None]));assert np.all(chosen.sum(1)==k)
 ids=np.nonzero(chosen)[1].reshape(len(score),k);order=np.argsort(-np.take_along_axis(score,ids,axis=1),axis=1,kind='stable');return np.take_along_axis(ids,order,axis=1)
def components(m,n,model):
 out=[];gi=np.einsum('ij,ij->i',m['p'][UI],m['q'][II]);bu=m['bu'][UI];bi=m['bi'][II];pred=m['mu']+bu+bi+gi
 for group,mask in [('target',TARGET),('non_target',~TARGET)]:
  obs=mask[II];nobs=int(obs.sum());vals=dict(mu=m['mu'],mean_bu=float(bu[obs].mean()),mean_bi=float(bi[obs].mean()),mean_interaction=float(gi[obs].mean()),mean_prediction=float(pred[obs].mean()),mae=float(np.abs(pred[obs]-RATING[obs]).mean()),mean_recorded_rating=float(RATING[obs].mean()))
  out.append(dict(model=model,n_fake=n,group=group,view='real_observed_training',n_pairs=nobs,**vals))
  cnt=np.bincount(UI[obs],minlength=N);available=int(mask.sum())-cnt;ok=available>0
  dots=m['p']@m['q'][mask].sum(0)-np.bincount(UI[obs],weights=gi[obs],minlength=N)
  bs=m['bi'][mask].sum()-np.bincount(UI[obs],weights=bi[obs],minlength=N)
  vals=dict(mu=m['mu'],mean_bu=float(m['bu'][ok].mean()),mean_bi=float(np.mean(bs[ok]/available[ok])),mean_interaction=float(np.mean(dots[ok]/available[ok])))
  vals['mean_prediction']=sum(vals.values());out.append(dict(model=model,n_fake=n,group=group,view='real_unseen_prediction',n_pairs=int(available.sum()),n_users=int(ok.sum()),**vals))
  if n:
   total=0;sums=np.zeros(8);negative=0;minimum=np.inf
   for a in range(0,n,64):
    v=m['fp'][a:a+64]@m['q'][mask].T;offset=m['mu']+m['fbu'][a:a+64,None]+m['bi'][mask];pr=offset+v;rating=5 if group=='target' else 1;er=pr-rating
    sums+=np.array([v.sum(),offset.sum(),pr.sum(),np.abs(er).sum(),(er**2).sum(),np.broadcast_to(m['bi'][mask],v.shape).sum(),np.broadcast_to(m['fbu'][a:a+64,None],v.shape).sum(),er.sum()]);total+=v.size;negative+=np.count_nonzero(v<-1e-10);minimum=min(minimum,float(v.min()))
   z=sums/total;out.append(dict(model=model,n_fake=n,group=group,view='synthetic_training',n_pairs=int(total),mu=m['mu'],mean_bu=float(z[6]),mean_bi=float(z[5]),mean_interaction=float(z[0]),mean_offset=float(z[1]),mean_prediction=float(z[2]),mae=float(z[3]),rmse=float(np.sqrt(z[4])),mean_recorded_rating=rating,negative_interaction_percent=100*negative/total,min_interaction=minimum))
  selected=[r for r in out if r['group']==group and r['view'] in ['real_observed_training','synthetic_training']];weights=np.array([r['n_pairs'] for r in selected]);keys=['mu','mean_bu','mean_bi','mean_interaction','mean_prediction','mae','mean_recorded_rating'];means={k:float(np.average([r[k] for r in selected],weights=weights)) for k in keys};out.append(dict(model=model,n_fake=n,group=group,view='all_training',n_pairs=int(weights.sum()),**means))
 for row in out:assert np.isclose(row['mean_prediction'],row['mu']+row['mean_bu']+row['mean_bi']+row['mean_interaction'],atol=1e-9)
 return out

def run(job):
 model,n=job;dest=resultpath(model,n);cache=cachepath(model,n)
 if dest.exists() and cache.exists():return json.loads(dest.read_text())
 tick=time.time()
 with np.load(checkpoint(model,n)) as z:m=align(z)
 with np.load(checkpoint(model,0)) as z:b=align(z)
 valid=bool(model!='nmf' or min(m['p'].min(),m['q'].min(),m['fp'].min() if n else 0)>=-1e-12)
 print('ANALYZE',model,n,flush=True);comp=components(m,n,model)
 topids=np.empty((N,35),np.int32);swaps={k:{'baseline':0,'after':0,'embedding_only':0,'item_bias_only':0} for k in [15,25,35]}
 basecache=np.load(cachepath(model,0)) if n else None;base_top=basecache['top35'] if n else None
 for a in range(0,N,256):
  stop=min(a+256,N);seen=SEEN[a:stop].toarray();g=m['p'][a:stop]@m['q'].T;current=top(g+m['bi'],seen);topids[a:stop]=current
  if n:
   g0=b['p'][a:stop]@b['q'].T
   rankings={'baseline':base_top[a:stop],'after':current,'embedding_only':top(g+b['bi'],seen),'item_bias_only':top(g0+m['bi'],seen)}
   for name,ids in rankings.items():
    for k in swaps:swaps[k][name]+=int(TARGET[ids[:,:k]].sum())
 # Validate complete rankings directly for fixed users without partition shortcuts.
 for uid in np.random.RandomState(42).choice(N,5,replace=False):
  score=m['p'][uid]@m['q'].T+m['bi'];score[SEEN[uid].toarray()[0]]=-np.inf;ref=np.lexsort((ITEMS,-score))[:35];assert np.array_equal(ref,topids[uid])
 rec=[];frequent=[]
 for k in KS:
  freq=np.bincount(topids[:,:k].ravel(),minlength=len(ITEMS));tf=freq[TARGET];slots=int(tf.sum());most=np.flatnonzero(TARGET)[np.argmax(tf)]
  rec.append(dict(model=model,genre='Adventure',n_fake=n,k=k,n_users=N,mean_target_count=slots/N,target_share_percent=100*slots/(N*k),union_distinct_target_titles=int(np.count_nonzero(tf)),most_frequent_book_id=int(ITEMS[most]),most_frequent_book_title=str(TITLES[most]),most_frequent_book_user_reach_percent=100*int(freq[most])/N,nmf_nonnegative_constraint_ok=valid))
  if k in [15,35]:
   for idx in np.argsort(-freq,kind='stable')[:15]:frequent.append(dict(k=k,book_id=int(ITEMS[idx]),title=str(TITLES[idx]),genres=str(GENRE_TEXT[idx]),users=int(freq[idx]),reach_percent=100*int(freq[idx])/N,original_rating_count=int(REAL_POP[idx]),original_mean_rating=float(REAL_MEAN[idx])))
 swaprows=[]
 if n:
  for k,counts in swaps.items():
   means={name:value/N for name,value in counts.items()};emb=.5*((means['embedding_only']-means['baseline'])+(means['after']-means['item_bias_only']));bias=.5*((means['item_bias_only']-means['baseline'])+(means['after']-means['embedding_only']));assert np.isclose(emb+bias,means['after']-means['baseline'])
   swaprows.append(dict(k=k,**means,embedding_swap_allocation=emb,item_bias_swap_allocation=bias,delta_count=means['after']-means['baseline']))
 biases=[dict(model=model,n_fake=n,genre=g,mean_bi=float(m['bi'][GENRE_MASK[:,j]].mean()),mu=m['mu'],mu_plus_bi=float(m['mu']+m['bi'][GENRE_MASK[:,j]].mean())) for j,g in enumerate(GENRES)]
 # Static historical user labels, not inferred changes in actual human preferences.
 topgenre=np.stack([GENRE_MASK[topids[:,:15],j].sum(1) for j in range(len(GENRES))],axis=1)
 tcounts=TARGET[topids[:,:15]].sum(1);neighborrows=[];knncache={}
 for metric in ['cosine','euclidean']:
  try:
   nn,_=neighbors(m['p'],USERS,USERS,k=20,metric=metric,workers=2)
   before=basecache['neighbors_'+metric] if n else nn
   overlap=(nn[:,:,None]==before[:,None,:]).any(2);turnover=1-overlap.mean(1)
   for q in np.random.RandomState(42).choice(N,5,replace=False):
    coords=m['p'] if metric=='euclidean' else m['p']/np.linalg.norm(m['p'],axis=1)[:,None]
    distances=np.linalg.norm(coords-coords[q],axis=1);distances[q]=np.inf;ref=np.lexsort((USERS,distances))[:20];assert np.array_equal(ref,nn[q])
   bcounts=TARGET[base_top[:,:15]].sum(1) if n else tcounts
   shares=topgenre[nn].mean((0,1));ret_after=float((tcounts[nn]*overlap).sum(1).mean()/20);inc_after=float((tcounts[nn]*(~overlap)).sum(1).mean()/20)
   assert np.isclose(ret_after+inc_after,shares[GENRES.index('Adventure')])
   neighborrows.append(dict(metric=metric,mean_turnover=float(turnover.mean()),mean_replaced_count=float(20*turnover.mean()),retained_after_contribution=ret_after,incoming_after_contribution=inc_after,retained_before_contribution=float((bcounts[nn]*overlap).sum(1).mean()/20),incoming_before_contribution=float((bcounts[nn]*(~overlap)).sum(1).mean()/20),genre_mean_books_top15={g:float(v) for g,v in zip(GENRES,shares)},neighbor_historical_genre_percent={g:float(100*(USER_LABEL[nn]==j).mean()) for j,g in enumerate(GENRES)},validation_queries=5))
   knncache['neighbors_'+metric]=nn.astype(np.int32)
  except ValueError as exc:neighborrows.append(dict(metric=metric,status='undefined',reason=str(exc)))
 rotation,_=orthogonal_procrustes(m['p'],b['p']);aligned=m['p']@rotation;den=np.linalg.norm(aligned,axis=1)*np.linalg.norm(b['p'],axis=1);ok=den>0;cs=np.einsum('ij,ij->i',aligned[ok],b['p'][ok])/den[ok]
 # Same fixed pairs at all injection levels; recommendation set overlap.
 pairrng=np.random.RandomState(42);p1=pairrng.randint(0,N,10000);p2=pairrng.randint(0,N-1,10000);p2+=p2>=p1
 intersections=(topids[p1,:15,None]==topids[p2,None,:15]).any(2).sum(1);jaccard=float(np.mean(intersections/(30-intersections)))
 np.savez_compressed(cache,top35=topids,ids=USERS,items=ITEMS,rotation=rotation,**knncache)
 examples=[]
 for uid in [int(x) for x in np.random.RandomState(42).choice(N,3,replace=False)]:examples.append(dict(user_id=int(USERS[uid]),books=[dict(book_id=int(ITEMS[j]),title=str(TITLES[j]),genres=str(GENRE_TEXT[j])) for j in topids[uid,:15]]))
 data=dict(model=model,n_fake=n,genre='Adventure',components=comp,recommendations=rec,component_swaps=swaprows,genre_biases=biases,neighbors=neighborrows,frequent_books=frequent,examples=examples,mean_pairwise_recommendation_jaccard_top15=jaccard,procrustes_aligned_mean_user_cosine=float(cs.mean()),min_pu=float(m['p'].min()),min_qi=float(m['q'].min()),nmf_nonnegative_constraint_ok=valid,checkpoint=str(checkpoint(model,n)),factors_clipped=False,range_clipped=False,elapsed_seconds=time.time()-tick)
 write_json(dest,data);print('ANALYSIS DONE',model,n,'seconds',round(time.time()-tick),'Top15',rec[0]['mean_target_count'],flush=True)
 return data

def initialize():
 global USERS,ITEMS,N,UI,II,RATING,SEEN,TARGET,GENRES,GENRE_MASK,TITLES,GENRE_TEXT,REAL_POP,REAL_MEAN,USER_LABEL
 df=pd.read_csv(SRC,usecols=['user_id','book_id','rating','genres','original_title']);USERS=np.sort(df.user_id.unique());ITEMS=np.sort(df.book_id.unique());N=len(USERS);assert N==53424 and len(ITEMS)==9415 and len(df)==5815569
 UI=np.searchsorted(USERS,df.user_id.to_numpy());II=np.searchsorted(ITEMS,df.book_id.to_numpy());RATING=df.rating.to_numpy(float);SEEN=csr_matrix((np.ones(len(df),bool),(UI,II)),shape=(N,len(ITEMS)));assert SEEN.nnz==len(df)
 meta=df.drop_duplicates('book_id').set_index('book_id').loc[ITEMS];GENRE_TEXT=meta.genres.astype(str).to_numpy();tags=[{g.strip() for g in x.split(',') if g.strip() and g.strip().lower()!='unknown'} for x in GENRE_TEXT];GENRES=sorted(set.union(*tags));GENRE_MASK=np.array([[g in t for g in GENRES] for t in tags]);TARGET=GENRE_MASK[:,GENRES.index('Adventure')];canonical=pd.read_csv(R/'Book/Archive/data/goodbooks-10k/books.csv').set_index('book_id')['title'];TITLES=np.array([str(canonical.get(int(i),meta.loc[i,'original_title'])).strip() or f'Book {int(i)}' for i in ITEMS]);REAL_POP=np.bincount(II,minlength=len(ITEMS));REAL_MEAN=np.bincount(II,weights=RATING,minlength=len(ITEMS))/REAL_POP
 freq=SEEN.astype(np.int32)@GENRE_MASK.astype(np.int32);USER_LABEL=np.argmax(freq,axis=1);assert TARGET.any() and (~TARGET).any()
 write_json(NB/'adventure_data_audit.json',dict(users=N,books=len(ITEMS),real_ratings=len(df),target_books=int(TARGET.sum()),non_target_books=int((~TARGET).sum()),genres=GENRES,counts=COUNTS,source=str(SRC),original_data_unchanged=True,neighbors_real_users_only=True,topk=[15,20,25,35],no_new_directories=True))

def summarize():
 rows=[json.loads(resultpath(m,n).read_text()) for m in ['svd','nmf'] for n in COUNTS]
 for key,name in [('components','adventure_score_components.csv'),('recommendations','adventure_recommendation_summary.csv'),('genre_biases','adventure_genre_biases.csv')]:pd.DataFrame([v for r in rows for v in r[key]]).to_csv(T/name,index=False)
 pd.DataFrame([dict(model=r['model'],n_fake=r['n_fake'],**v) for r in rows for v in r['component_swaps']]).to_csv(T/'adventure_component_swaps.csv',index=False)
 pd.DataFrame([dict(model=r['model'],n_fake=r['n_fake'],**v) for r in rows for v in r['neighbors']]).to_csv(LT/'adventure_neighbors.csv',index=False)
 write_json(NB/'adventure_analysis_status.json',dict(status='complete',cases=len(rows),counts=COUNTS,finished=time.time(),nonnegative_constraint_failures=[dict(model=r['model'],n_fake=r['n_fake'],min_pu=r['min_pu'],min_qi=r['min_qi']) for r in rows if not r['nmf_nonnegative_constraint_ok']]))
if __name__=='__main__':
 initialize()
 with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(run,[(m,0) for m in ['svd','nmf']]))
 done=set();pending={};failures=[]
 with ProcessPoolExecutor(max_workers=6) as pool:
  while len(done)+len(failures)<26:
   for m in ['svd','nmf']:
    for n in COUNTS[1:]:
     job=(m,n)
     if job in done or job in pending or any(x['job']==list(job) for x in failures):continue
     if checkpoint(m,n).exists() and (NB/f'adventure_{m}_{n}_training.json').exists():pending[job]=pool.submit(run,job)
   for job,fut in list(pending.items()):
    if fut.done():
     try:fut.result();done.add(job)
     except Exception as exc:failures.append(dict(job=list(job),error=repr(exc)));print('ANALYSIS FAILED',job,repr(exc),flush=True)
     del pending[job]
   write_json(NB/'adventure_analysis_status.json',dict(status='analyzing',completed=len(done),waiting=len(pending),total=26,failures=failures,updated=time.time()))
   if len(done)+len(failures)<26:time.sleep(15)
 if failures:raise RuntimeError(failures)
 summarize();print('ALL ADVENTURE ANALYSES COMPLETE',flush=True)
