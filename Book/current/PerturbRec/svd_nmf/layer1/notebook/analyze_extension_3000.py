#!/usr/bin/env python3
"""Compare saved controlled runs at 2000/2500/3000; no additional training."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor
ROOT=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf')
L=ROOT/'layer1';E=ROOT/'layer2/embeddings';COUNTS=[0,2000,2500,3000];KS=[15,20,25,35]
SRC=Path('/home/moshtasa/Research/phd-recsys/Book/Archive/data/genre_clean.csv')

def exposures(score,seen,target):
 score[seen]=-np.inf;assert np.all((~seen).sum(1)>=35)
 partitioned=np.partition(score,[-k for k in KS],axis=1)
 answer={}
 for k in KS:
  threshold=partitioned[:,-k];higher=score>threshold[:,None];ties=score==threshold[:,None]
  remaining=k-higher.sum(1)
  chosen=higher|(ties&(ties.cumsum(1)<=remaining[:,None]))
  assert np.all(chosen.sum(1)==k)
  answer[k]=(chosen&target).sum(1)
 return answer

def run(job):
 model,genre=job;df=pd.read_csv(SRC,usecols=['user_id','book_id','genres'])
 base=np.load(E/'pilot'/f'{model}_baseline_a.npz');ids=np.sort(base['ids']);items=np.sort(base['items'])
 ur=np.searchsorted(ids,df.user_id.to_numpy());ir=np.searchsorted(items,df.book_id.to_numpy())
 assert np.array_equal(ids[ur],df.user_id.to_numpy()) and np.array_equal(items[ir],df.book_id.to_numpy())
 seen=csr_matrix((np.ones(len(df),bool),(ur,ir)),shape=(len(ids),len(items)))
 genres=dict(df[['book_id','genres']].drop_duplicates('book_id').itertuples(index=False,name=None))
 target=np.array([genre in [g.strip() for g in str(genres[i]).split(',')] for i in items]);result=[]
 for n in COUNTS:
  z=base if n==0 else np.load(E/'controlled'/model/genre.lower()/f'{model}_{genre.lower()}_{n}.npz')
  ui=np.argsort(z['ids']);ui=ui[np.searchsorted(z['ids'][ui],ids)];ii=np.argsort(z['items'])
  assert np.array_equal(z['ids'][ui],ids) and np.array_equal(z['items'][ii],items)
  pu=z['pu'][ui];qi=z['qi'][ii];bi=z['bi'][ii];total={k:0 for k in KS}
  for a in range(0,len(ids),128):
   b=min(a+128,len(ids));score=pu[a:b]@qi.T+bi
   ans=exposures(score,seen[a:b].toarray(),target)
   for k in KS:total[k]+=int(ans[k].sum())
  for k in KS:result.append(dict(model=model,genre=genre,n_fake=n,k=k,n_users=len(ids),mean_target_count=total[k]/len(ids),target_share_percent=100*total[k]/(len(ids)*k)))
  print('EXPOSURE',model,genre,n,'Top15',round(total[15]/len(ids),5),flush=True)
  if n:z.close()
 return result

def plot(data):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15})
 navy='#142E49';muted='#546576';colors=['#34495E','#A0CBE8','#D6A300','#2A9DAB']
 for genre in ['Drama','Thriller']:
  for ks in [[15],[15,25,35]]:
   fig,axes=plt.subplots(1,2,figsize=(16,9),sharey=True);handles=[];maximum=0
   for ax,model in zip(axes,['nmf','svd']):
    x=np.arange(len(ks))*1.3;width=.22
    for j,(n,c) in enumerate(zip(COUNTS,colors)):
     vals=[]
     for k in ks:
      rows=data[(data.model==model)&(data.genre==genre)&(data.n_fake==n)&(data.k==k)];assert len(rows)==1;vals.append(rows.mean_target_count.iloc[0])
     maximum=max(maximum,max(vals));bars=ax.bar(x+(j-1.5)*width,vals,width=width*.94,color=c,zorder=3)
     if model=='nmf':handles.append(bars[0])
     for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+.15,f'{v:.2f}',ha='center',fontsize=10,color=navy)
    ax.set_title(model.upper(),fontsize=23,color=navy,weight='bold',pad=16)
    ax.set_xticks(x);ax.set_xticklabels([f'Top-{k}' for k in ks],fontsize=17,color=navy)
    ax.set_xlabel('Recommendation list size',fontsize=15,color=muted,labelpad=16)
    ax.tick_params(axis='y',labelsize=14,colors=muted,length=0)
    ax.grid(axis='y',color='#DCE3E9',linewidth=.7,zorder=0)
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#C0CBD5');ax.tick_params(axis='x',length=0,pad=10)
   for ax in axes:ax.set_ylim(0,max(3,maximum*1.15))
   axes[0].set_ylabel(f'Mean {genre} books per real user',fontsize=16,color=navy,labelpad=14)
   fig.subplots_adjust(left=.085,right=.975,bottom=.21,top=.72,wspace=.12)
   fig.text(.055,.943,f'{genre} recommendations beyond 2,000 injected users',fontsize=25,weight='bold',color=navy)
   fig.text(.055,.892,'Controlled model runs • Same clean catalog and shared initialization • 53,424 real users',fontsize=13,color=muted)
   fig.legend(handles,['Baseline','2,000 users','2,500 users','3,000 users'],loc='upper center',bbox_to_anchor=(.51,.84),ncol=4,frameon=False,fontsize=15)
   fig.text(.055,.075,'Previously rated books excluded; ranking uses learned user–book interactions and item biases.',fontsize=12.5,color=navy)
   fig.text(.055,.037,'Target-book ratings = 5; other-book ratings = 1. Shared linear y-axis; ascending book-ID tie resolution.',fontsize=11.5,color=muted)
   folder=L/'result/extension_2500_3000'/genre;folder.mkdir(parents=True,exist_ok=True)
   fig.savefig(folder/('comparison_top'+('_'.join(map(str,ks)))+'.png'),dpi=240,facecolor='white');plt.close(fig)

if __name__=='__main__':
 table=L/'tables/extension_2500_3000_recommendation_counts.csv'
 if table.exists():data=pd.read_csv(table)
 else:
  with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,[(m,g) for m in ['nmf','svd'] for g in ['Drama','Thriller']]))
  data=pd.DataFrame([r for a in results for r in a]);data.to_csv(table,index=False)
 assert len(data)==64
 # Check the unchanged Top-20 baseline and 2000 results against existing controlled outputs.
 old=pd.read_csv(L/'tables/controlled_recommendation_summary.csv')
 for row in data[data.k==20].itertuples():
  prev=old[(old.model==row.model)&(old.genre==row.genre)]
  if row.n_fake==0:assert np.isclose(row.mean_target_count,prev.target_count_baseline.iloc[0],atol=1e-9)
  elif row.n_fake==2000:assert np.isclose(row.mean_target_count,prev[prev.n_fake==2000].target_count_after.iloc[0],atol=1e-9)
 plot(data);print('RECOMMENDATION ANALYSIS COMPLETE',flush=True)
