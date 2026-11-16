"""Rebuild Shopping diagnostic measurements using the original 0519 settings and injection order."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import json,time,traceback,gc,sys
from pathlib import Path
import numpy as np,pandas as pd
from surprise import Dataset,Reader,SVD,NMF
R=Path('/home/moshtasa/Research/phd-recsys'); NB=R/'First_paper_material/notebook'; OUT=R/'First_paper_material/figures/first_reviewer/shopping'
NS=[0,30,60,300,500,2000,7000]
PARAMS=dict(biased=True,n_factors=8,n_epochs=180,reg_pu=0.,reg_qi=.002,random_state=42,verbose=False)
STATUS=NB/'shopping_reviewer_status.json'; CSV=NB/'shopping_reviewer_components.csv'
def status(**kw):
 STATUS.write_text(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw),indent=2)); print(kw,flush=True)
def main():
 OUT.mkdir(exist_ok=True)
 status(stage='loading original data')
 df=pd.read_csv(R/'Mobile/archive/0316_Single/data/data/mobilerec_final.csv',usecols=['uid','app_package','rating','app_category'],dtype={'uid':str,'app_package':str})
 df=df.dropna(subset=['uid','app_package','rating']); df.rating=pd.to_numeric(df.rating,errors='coerce').fillna(0).clip(0,5)
 df.app_category=df.app_category.fillna('').astype(str).str.strip()
 cats=df.drop_duplicates('app_package').set_index('app_package').app_category
 pos=cats[cats=='Shopping'].index.to_numpy(); neg=cats[(cats!='Shopping')&(cats!='')].index.to_numpy()
 realusers=df.uid.unique(); userset=set(realusers); m=len(realusers)
 allrows=pd.read_csv(CSV).to_dict('records') if CSV.exists() else []
 for modelname,cls in [('svd',SVD),('nmf',NMF)]:
  for n in NS:
   if any(x['model']==modelname and x['n_fake']==n and x['view']=='real_unseen_prediction' for x in allrows):continue
   status(stage='preparing dataset',model=modelname,n_fake=n,n_real_users=m,n_target_items=len(pos),n_non_target_items=len(neg),parameters=PARAMS)
   if n:
    ids=np.array([f'shopping_reviewer_synthetic_{m+1+i}' for i in range(n)],dtype=object)
    assert not set(ids)&userset
    synth=pd.DataFrame({'uid':np.concatenate([np.repeat(ids,len(pos)),np.repeat(ids,len(neg))]),'app_package':np.concatenate([np.tile(pos,n),np.tile(neg,n)]),'rating':np.concatenate([np.full(n*len(pos),5.),np.ones(n*len(neg))])})
    train=pd.concat([df[['uid','app_package','rating']],synth],ignore_index=True);del synth
   else:train=df[['uid','app_package','rating']]
   ts=Dataset.load_from_df(train[['uid','app_package','rating']],Reader(rating_scale=(0,5))).build_full_trainset()
   checkpoint=NB/f'shopping_{modelname}_{n}_factors.npz'
   model=cls(**PARAMS)
   if checkpoint.exists():
    a=np.load(checkpoint);model.pu=a['pu'];model.qi=a['qi'];model.bu=a['bu'];model.bi=a['bi'];status(stage='using saved factors',model=modelname,n_fake=n)
   else:
    status(stage='training',model=modelname,n_fake=n,n_ratings=ts.n_ratings)
    model.fit(ts)
    np.savez_compressed(checkpoint,pu=model.pu,qi=model.qi,bu=model.bu,bi=model.bi,mu=ts.global_mean)
   status(stage='measuring training pairs',model=modelname,n_fake=n)
   u=train.uid.map(ts._raw2inner_id_users).to_numpy(dtype=np.int32); i=train.app_package.map(ts._raw2inner_id_items).to_numpy(dtype=np.int32);rating=train.rating.to_numpy(float)
   target=np.array([cats.get(ts.to_raw_iid(j),'')=='Shopping' for j in range(ts.n_items)])
   realmask=np.array([ts.to_raw_uid(j) in userset for j in range(ts.n_users)])
   acc={}
   for start in range(0,len(u),250000):
    uu=u[start:start+250000];ii=i[start:start+250000];rr=rating[start:start+250000]
    bu=model.bu[uu];bi=model.bi[ii];inter=np.einsum('ij,ij->i',model.pu[uu],model.qi[ii]);pred=ts.global_mean+bu+bi+inter
    for view,vm in [('all_training',np.ones(len(uu),bool)),('synthetic_training',~realmask[uu])]:
     for group,gm in [('target',target[ii]),('non_target',~target[ii])]:
      mask=vm&gm
      if not mask.any():continue
      key=(view,group); sums=acc.setdefault(key,np.zeros(8));sums+=np.array([mask.sum(),bu[mask].sum(),bi[mask].sum(),inter[mask].sum(),pred[mask].sum(),np.abs(rr[mask]-pred[mask]).sum(),rr[mask].sum(),0])
   for (view,group),s in acc.items():
    allrows.append(dict(genre='Shopping',model=modelname,n_fake=n,view=view,group=group,n_pairs=int(s[0]),mu=ts.global_mean,mean_bu=s[1]/s[0],mean_bi=s[2]/s[0],mean_interaction=s[3]/s[0],mean_prediction=s[4]/s[0],mae=s[5]/s[0],mean_recorded_rating=s[6]/s[0]))
   del train,u,i,rating;gc.collect()
   status(stage='measuring unrated apps for real users',model=modelname,n_fake=n)
   realids=np.flatnonzero(realmask);sums={g:np.zeros(5) for g in ['target','non_target']}
   for st in range(0,len(realids),128):
    batch=realids[st:st+128]; inter=model.pu[batch]@model.qi.T
    unseen=np.ones(inter.shape,bool)
    for row,uid in enumerate(batch):unseen[row,[j for j,r in ts.ur[int(uid)]]]=False
    for group,maskitems in [('target',target),('non_target',~target)]:
     mask=unseen&maskitems[None,:];counts=mask.sum(axis=1);valid=counts>0
     if not valid.any():continue
     bimean=(mask@model.bi)/np.maximum(counts,1);imean=(inter*mask).sum(axis=1)/np.maximum(counts,1);bumean=model.bu[batch];pred=ts.global_mean+bumean+bimean+imean
     sums[group]+=np.array([valid.sum(),bumean[valid].sum(),bimean[valid].sum(),imean[valid].sum(),pred[valid].sum()])
    if st%25600==0: print('prediction progress',modelname,n,st,len(realids),flush=True)
   for group,s in sums.items():
    allrows.append(dict(genre='Shopping',model=modelname,n_fake=n,view='real_unseen_prediction',group=group,n_users=int(s[0]),mu=ts.global_mean,mean_bu=s[1]/s[0],mean_bi=s[2]/s[0],mean_interaction=s[3]/s[0],mean_prediction=s[4]/s[0]))
   pd.DataFrame(allrows).to_csv(CSV,index=False)
   del model,ts;gc.collect();status(stage='completed condition',model=modelname,n_fake=n)
 status(stage='all measurements complete',output=str(CSV))
if __name__=='__main__':
 try:main()
 except Exception:
  status(stage='failed',error=traceback.format_exc());raise
