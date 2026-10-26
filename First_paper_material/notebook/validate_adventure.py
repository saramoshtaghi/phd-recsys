#!/usr/bin/env python3
"""Independent checks of persisted input and numerical analysis identities."""
from pathlib import Path
import json,time,hashlib
import numpy as np,pandas as pd
R=Path('/home/moshtasa/Research/phd-recsys');NB=R/'First_paper_material/notebook';P=R/'Book/current/PerturbRec';L=P/'svd_nmf/layer2';COUNTS=[2,4,6,25,50,100,200,500,1000,2000,2500,3000,3500]

def input_validation():
 source=R/'Book/Archive/data/genre_clean.csv';df=pd.read_csv(source,usecols=['user_id','book_id','rating','genres'])[['user_id','book_id','rating','genres']];df[['user_id','book_id','rating']]=df[['user_id','book_id','rating']].astype('int64');meta=df[['book_id','genres']].drop_duplicates('book_id');mapping=dict(meta.itertuples(index=False,name=None));target={i for i,g in mapping.items() if 'Adventure' in [v.strip() for v in g.split(',')]};allitems=set(mapping);start=int(df.user_id.max())+1;ordered=np.array(sorted(target)+sorted(allitems-target));perbook=np.zeros(9415,dtype=np.int64);items=np.array(sorted(allitems));perfake=np.zeros(3500,dtype=np.int64);observed=0;realpos=0
 for part in pd.read_csv(P/'Data/f_Adventure_3500u_pos5_neg1_books.csv',chunksize=300000):
  part=part[['user_id','book_id','rating','genres']];nreal=min(max(len(df)-observed,0),len(part))
  if nreal:
   a=part.iloc[:nreal].reset_index(drop=True);b=df.iloc[realpos:realpos+nreal].reset_index(drop=True);assert a.equals(b),'Original CSV prefix changed';realpos+=nreal
  fake=part.iloc[nreal:]
  if len(fake):
   uid=fake.user_id.to_numpy();bid=fake.book_id.to_numpy();rat=fake.rating.to_numpy();offset=observed+nreal-len(df);positions=np.arange(offset,offset+len(fake));assert np.array_equal(uid,start+positions//9415);assert np.array_equal(bid,ordered[positions%9415]);assert uid.min()>=start and uid.max()<start+3500;assert np.isin(bid,items).all();expected=np.where(np.isin(bid,list(target)),5,1);assert np.array_equal(rat,expected);perfake+=np.bincount(uid-start,minlength=3500);perbook+=np.bincount(np.searchsorted(items,bid),minlength=9415)
   assert all(str(g)==str(mapping[int(b)]) for b,g in fake[['book_id','genres']].drop_duplicates().itertuples(index=False,name=None))
  observed+=len(part)
 assert realpos==len(df) and observed==len(df)+3500*9415;assert np.all(perfake==9415) and np.all(perbook==3500)
 manifests=[]
 for n in COUNTS:
  m=json.loads((P/'Data/manifests'/f'adventure_{n}.json').read_text());p=P/'Data'/f'f_Adventure_{n}u_pos5_neg1_books.csv';assert p.stat().st_size==m['bytes'];assert m['rows']==len(df)+n*9415 and m['n_fake']==n and m['target_books']==1789;manifests.append(m)
 result=dict(status='passed',checked_largest_csv_rows=observed,original_rows_compared_exactly=realpos,all_3500_fake_users_rate_every_book_once=True,ratings_follow_policy=True,all_13_input_manifests_verified=True,source_dataframe_sha256=hashlib.sha256(pd.util.hash_pandas_object(df,index=False).values.tobytes()).hexdigest())
 (NB/'adventure_input_validation.json').write_text(json.dumps(result,indent=2));print('INPUT VALIDATION PASSED',flush=True)

def numerical_validation():
 while True:
  try:
   st=json.loads((NB/'adventure_analysis_status.json').read_text())
   if st.get('failures'):raise RuntimeError(st['failures'])
   if st.get('status')=='complete':break
  except FileNotFoundError:pass
  time.sleep(15)
 checked=0;constraint_failures=[]
 for model in ['svd','nmf']:
  for n in [0]+COUNTS:
   d=json.loads((NB/f'adventure_{model}_{n}_analysis.json').read_text());rs=d['recommendations'];assert {r['k'] for r in rs}=={15,20,25,35}
   for r in rs:assert 0<=r['mean_target_count']<=r['k'];assert np.isclose(r['target_share_percent'],100*r['mean_target_count']/r['k']);assert 0<r['union_distinct_target_titles']<=1789
   for r in d['components']:assert np.isclose(r['mean_prediction'],r['mu']+r['mean_bu']+r['mean_bi']+r['mean_interaction']);assert r.get('mae',0)>=0
   for r in d['component_swaps']:assert np.isclose(r['embedding_swap_allocation']+r['item_bias_swap_allocation'],r['delta_count'])
   if not d['nmf_nonnegative_constraint_ok']:constraint_failures.append(dict(model=model,n=n,min_pu=d['min_pu'],min_qi=d['min_qi']))
   if n:
    z=json.loads((NB/f'adventure_{model}_{n}_training.json').read_text());assert z['shared_initialization_verified'] and not z['factors_clipped'] and z['ratings']==5815569+n*9415
   for row in d['neighbors']:
    if row.get('status')=='undefined':continue
    assert 0<=row['mean_turnover']<=1 and np.isclose(row['retained_after_contribution']+row['incoming_after_contribution'],row['genre_mean_books_top15']['Adventure']);assert all(0<=v<=15 for v in row['genre_mean_books_top15'].values())
   checked+=1
 # Same observations imply exactly equal global means between model types.
 for n in [0]+COUNTS:
  a=json.loads((NB/f'adventure_svd_{n}_analysis.json').read_text());b=json.loads((NB/f'adventure_nmf_{n}_analysis.json').read_text());assert np.isclose(a['components'][0]['mu'],b['components'][0]['mu'],rtol=0,atol=1e-12)
 out=dict(status='passed',cases=checked,counts=[0]+COUNTS,shared_initialization_verified=True,score_component_identities_verified=True,topk_counts_verified=True,conditional_swap_identities_verified=True,neighborhood_contributions_verified=True,raw_factors_preserved=True,nmf_constraint_failures=constraint_failures,finished=time.time());(NB/'adventure_numerical_validation.json').write_text(json.dumps(out,indent=2));print('NUMERICAL VALIDATION PASSED',checked,flush=True)
if __name__=='__main__':input_validation();numerical_validation()
