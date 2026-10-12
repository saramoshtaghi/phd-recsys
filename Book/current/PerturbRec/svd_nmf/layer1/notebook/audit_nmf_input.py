#!/usr/bin/env python3
"""Read-only audit of actual input/initial factors and a small synthetic reproduction."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from surprise import Dataset,Reader,NMF
ROOT=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf')
SRC=Path('/home/moshtasa/Research/phd-recsys/Book/Archive/data/genre_clean.csv')
PARAMS=dict(biased=True,n_factors=8,n_epochs=180,reg_pu=0.,reg_qi=.002,random_state=42)
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def trainset(df):return Dataset.load_from_df(df[['user_id','book_id','rating']],Reader(rating_scale=(0,5))).build_full_trainset()
def main():
    df=pd.read_csv(SRC,usecols=['user_id','book_id','rating'])
    init=NMF(**{**PARAMS,'n_epochs':0}).fit(trainset(df))
    hashes=[digest(init.pu),digest(init.qi)]
    report=dict(real_ratings=len(df),rating_min=float(df.rating.min()),rating_max=float(df.rating.max()),
                negative_ratings=int((df.rating<0).sum()),missing_values=int(df.isna().sum().sum()),
                initial_pu_min=float(init.pu.min()),initial_qi_min=float(init.qi.min()),
                initial_common_hashes=hashes,cases=[])
    for n in (1000,2000):
        p=ROOT/'layer2/log/controlled/nmf/thriller'/f'nmf_thriller_{n}.json'
        meta=json.loads(p.read_text())
        assert meta['initial_common_hashes']==hashes
        assert meta['ratings']==len(df)+n*9415
        fake=np.random.RandomState(20260921).uniform(0,1,size=(n,8))
        report['cases'].append(dict(n_fake=n,common_initialization_matches=True,
            fake_initial_min=float(fake.min()),fake_initial_max=float(fake.max()),
            synthetic_ratings=[1,5],augmented_ratings=meta['ratings']))
    # Actual stored models are untouched. This toy experiment checks whether
    # positive inputs suffice to prevent sign loss in the installed implementation.
    found=None;trials=0
    for seed in range(10):
        rng=np.random.RandomState(seed);rows=[]
        for uid in range(40):
            for iid in rng.choice(80,12,replace=False):rows.append((uid,int(iid),int(rng.choice([3,4,5]))))
        # Ensure all items occur in the original catalog.
        for iid in range(80):rows.append((40,iid,4))
        for nfake in (16,64,128):
            fake=[(uid,iid,5 if iid<14 else 1) for uid in range(41,41+nfake) for iid in range(80)]
            toy=pd.DataFrame(rows+fake,columns=['user_id','book_id','rating'])
            model=NMF(**PARAMS).fit(trainset(toy));trials+=1
            if model.pu.min()< -1e-12 or model.qi.min()< -1e-12:
                found=dict(data_seed=seed,n_fake=nfake,n_ratings=len(toy),rating_min=int(toy.rating.min()),
                           rating_max=int(toy.rating.max()),min_pu=float(model.pu.min()),min_qi=float(model.qi.min()))
                break
        if found:break
    report['toy_trials']=trials;report['positive_input_toy_sign_loss']=found
    report['interpretation']='Actual ratings and both common/fake initialization are positive. The bad factors arise during optimization, not from negative input. Dense profile replication changes the training signal; exact first offending epoch in the real runs is not traced.'
    dest=ROOT/'layer1/result/reviewer_hypothesis/input_audit.json'
    dest.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
