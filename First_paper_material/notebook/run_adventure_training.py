#!/usr/bin/env python3
"""Adventure controlled sweep through 3500; existing directories only; raw factors."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import time,json,hashlib,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import numpy as np,pandas as pd
from surprise import SVD,NMF,Dataset,Reader
R=Path('/home/moshtasa/Research/phd-recsys');P=R/'Book/current/PerturbRec';L=P/'svd_nmf/layer2';NB=R/'First_paper_material/notebook'
SRC=R/'Book/Archive/data/genre_clean.csv';COUNTS=[2,4,6,25,50,100,200,500,1000,2000,2500,3000,3500]
PARAMS=dict(biased=True,n_factors=8,n_epochs=180,reg_pu=0.0,reg_qi=0.002,random_state=42,verbose=False)
MANIFEST=NB/'adventure_training_status.json'
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dataset(df):return Dataset.load_from_df(df[['user_id','book_id','rating']],Reader(rating_scale=(0,5))).build_full_trainset()
class FixedInit(np.random.RandomState):
 def __init__(self,pu,qi,n):super().__init__(42);self.a=pu;self.b=qi;self.n=n;self.calls=0;self.hashes=[]
 def take(self,kind,args,kw):
  size=kw['size']
  if self.calls==0:
   assert size==(len(self.a)+self.n,self.a.shape[1]);rng=np.random.RandomState(20260921)
   extra=getattr(rng,kind)(*args,size=(self.n,self.a.shape[1]));out=np.vstack([self.a,extra]);self.hashes.append(digest(out[:len(self.a)]))
  elif self.calls==1:assert size==self.b.shape;out=self.b.copy();self.hashes.append(digest(out))
  else:raise AssertionError('Unexpected RNG call')
  self.calls+=1;return out
 def normal(self,*a,**kw):return self.take('normal',a,kw)
 def uniform(self,*a,**kw):return self.take('uniform',a,kw)
def run(job):
 model,n=job;out=L/'embeddings/controlled'/model/f'{model}_adventure_{n}.npz';meta_path=NB/f'adventure_{model}_{n}_training.json'
 assert out.parent.is_dir() and NB.is_dir()
 if out.exists() and meta_path.exists():return json.loads(meta_path.read_text())
 started=time.time();df=pd.read_csv(SRC,usecols=['user_id','book_id','rating','genres']);assert len(df)==5815569
 cls={'svd':SVD,'nmf':NMF}[model];ts0=dataset(df);assert (ts0.n_users,ts0.n_items)==(53424,9415)
 init=cls(**{**PARAMS,'n_epochs':0}).fit(ts0);hashes=[digest(init.pu),digest(init.qi)]
 books=df[['book_id','genres']].drop_duplicates('book_id');tags={int(b):str(g) for b,g in books.itertuples(index=False,name=None)}
 pos=sorted(b for b,g in tags.items() if 'Adventure' in [x.strip() for x in g.split(',')]);neg=sorted(set(tags)-set(pos));assert pos and neg
 items=np.array(pos+neg);ratings=np.r_[np.full(len(pos),5),np.ones(len(neg),dtype=int)]
 newids=np.arange(int(df.user_id.max())+1,int(df.user_id.max())+1+n)
 extra=pd.DataFrame({'user_id':np.repeat(newids,len(items)),'book_id':np.tile(items,n),'rating':np.tile(ratings,n)})
 ts=dataset(pd.concat([df[['user_id','book_id','rating']],extra],ignore_index=True))
 assert [ts.to_raw_uid(i) for i in range(ts0.n_users)]==[ts0.to_raw_uid(i) for i in range(ts0.n_users)]
 assert [ts.to_raw_iid(i) for i in range(ts.n_items)]==[ts0.to_raw_iid(i) for i in range(ts0.n_items)]
 del extra,df
 rng=FixedInit(init.pu,init.qi,n);fit=cls(**{**PARAMS,'random_state':rng});print('TRAIN',model,n,'ratings',ts.n_ratings,flush=True);fit.fit(ts)
 assert rng.calls==2 and rng.hashes==hashes
 arrays=dict(ids=np.array([ts.to_raw_uid(i) for i in range(ts.n_users)],dtype=np.int64),items=np.array([ts.to_raw_iid(i) for i in range(ts.n_items)],dtype=np.int64),pu=fit.pu,qi=fit.qi,bu=fit.bu,bi=fit.bi,mu=ts.global_mean)
 assert all(np.isfinite(x).all() for x in arrays.values());assert fit.pu.shape==(53424+n,8)
 with open(str(out)+'.partial','wb') as h:np.savez_compressed(h,**arrays)
 Path(str(out)+'.partial').replace(out)
 meta=dict(model=model,genre='Adventure',n_fake=n,source=str(SRC),n_real=53424,books=9415,target_books=len(pos),non_target_books=len(neg),ratings=ts.n_ratings,parameters=PARAMS,initial_common_hashes=hashes,shared_initialization_verified=True,raw_scores=True,factors_clipped=False,min_pu=float(fit.pu.min()),min_qi=float(fit.qi.min()),nonnegative_factors=bool(fit.pu.min()>=-1e-12 and fit.qi.min()>=-1e-12),elapsed_seconds=time.time()-started,checkpoint=str(out))
 meta_path.write_text(json.dumps(meta,indent=2));print('DONE',model,n,round(meta['elapsed_seconds']),flush=True);return meta
if __name__=='__main__':
 for m in ['svd','nmf']:
  with np.load(L/'embeddings/pilot'/f'{m}_baseline_a.npz') as a,np.load(L/'embeddings/pilot'/f'{m}_baseline_b.npz') as b:assert all(np.array_equal(a[k],b[k]) for k in a.files)
 state=dict(status='training',started=time.time(),counts=COUNTS,models=['svd','nmf'],genre='Adventure',parameters=PARAMS,completed=[],failures=[])
 MANIFEST.write_text(json.dumps(state,indent=2))
 jobs=[(m,n) for n in reversed(COUNTS) for m in ['svd','nmf']]
 with ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn')) as pool:
  pending={pool.submit(run,j):j for j in jobs}
  for future in as_completed(pending):
   try:state['completed'].append(future.result())
   except Exception as exc:state['failures'].append(dict(job=pending[future],error=repr(exc)));print('FAILED',pending[future],repr(exc),flush=True)
   MANIFEST.write_text(json.dumps(state,indent=2))
 state.update(status='failed' if state['failures'] else 'trained',finished=time.time());MANIFEST.write_text(json.dumps(state,indent=2));print('TRAINING FINISHED',state['status'],flush=True)
 if state['failures']:sys.exit(1)
