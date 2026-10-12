import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor
ROOT=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf')
L=ROOT/'layer1';E=ROOT/'layer2/embeddings'
SRC=Path('/home/moshtasa/Research/phd-recsys/Book/Archive/data/genre_clean.csv')
COUNTS=[0,2,4,6,25,50,100,200,500,1000,2000,2500,3000]
TABLE=L/'tables/controlled_thriller_top15.csv'
PNG=L/'comparision - Thriller_top15_powerpoint.png'

def count_top(score,seen,target,k=15):
 score[seen]=-np.inf
 assert np.all((~seen).sum(1)>=k)
 threshold=np.partition(score,-k,axis=1)[:,-k]
 higher=score>threshold[:,None];ties=score==threshold[:,None]
 remaining=k-higher.sum(1)
 # Items are sorted by raw book ID, making tie resolution deterministic.
 chosen=higher|(ties&(ties.cumsum(1)<=remaining[:,None]))
 assert np.all(chosen.sum(1)==k)
 return (chosen&target).sum(1)

def run(model):
 df=pd.read_csv(SRC,usecols=['user_id','book_id','genres'])
 base=np.load(E/'pilot'/f'{model}_baseline_a.npz');ids=np.sort(base['ids']);items=np.sort(base['items'])
 ur=np.searchsorted(ids,df.user_id.to_numpy());ir=np.searchsorted(items,df.book_id.to_numpy())
 assert np.array_equal(ids[ur],df.user_id.to_numpy()) and np.array_equal(items[ir],df.book_id.to_numpy())
 seen=csr_matrix((np.ones(len(df),bool),(ur,ir)),shape=(len(ids),len(items)))
 genres=dict(df[['book_id','genres']].drop_duplicates('book_id').itertuples(index=False,name=None))
 target=np.array(['Thriller' in [g.strip() for g in str(genres[i]).split(',')] for i in items])
 rows=[]
 for n in COUNTS:
  z=base if n==0 else np.load(E/'controlled'/model/'thriller'/f'{model}_thriller_{n}.npz')
  ui=np.argsort(z['ids']);idx=np.searchsorted(z['ids'][ui],ids);ui=ui[idx]
  ii=np.argsort(z['items']);assert np.array_equal(z['ids'][ui],ids) and np.array_equal(z['items'][ii],items)
  pu=z['pu'][ui];qi=z['qi'][ii];bi=z['bi'][ii];total=0
  for a in range(0,len(ids),256):
   b=min(a+256,len(ids));score=pu[a:b]@qi.T+bi
   total+=int(count_top(score,seen[a:b].toarray(),target).sum())
  rec=dict(model=model,genre='Thriller',n_fake=n,k=15,n_users=len(ids),mean_target_count=total/len(ids),target_share_percent=100*total/(len(ids)*15))
  rows.append(rec);print('DONE',model,n,round(rec['mean_target_count'],5),flush=True)
  if n: z.close()
 return rows

def plot(data):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15})
 fig,axes=plt.subplots(1,2,figsize=(16,9),sharey=True)
 navy='#142E49';muted='#546576'
 colors=['#34495E','#4E79A7','#76B7B2','#59A14F','#B07AA1','#9C755F','#E15759','#F28E2B','#D6A300','#2A9DAB','#A0CBE8','#FF9DA7','#8064A2'];handles=[]
 ymax=max(data.mean_target_count.max()*1.15,3)
 for ax,m in zip(axes,['nmf','svd']):
  d=data[data.model==m].sort_values('n_fake');assert d.n_fake.tolist()==COUNTS
  x=(np.arange(len(COUNTS))-(len(COUNTS)-1)/2)*.084
  bars=ax.bar(x,d.mean_target_count,width=.079,color=colors,edgecolor='white',linewidth=.5,zorder=3)
  if m=='nmf':handles=list(bars)
  ax.set_title(m.upper(),fontsize=23,color=navy,weight='bold',pad=16)
  ax.set_xticks([0]);ax.set_xticklabels(['Top-15'],fontsize=17,color=navy)
  ax.set_xlabel('Recommendation list size',fontsize=15,color=muted,labelpad=16)
  ax.set_ylim(0,min(15,ymax));ax.tick_params(axis='y',labelsize=14,colors=muted,length=0)
  ax.grid(axis='y',color='#DCE3E9',linewidth=.7,zorder=0)
  for side in ['top','right','left']:ax.spines[side].set_visible(False)
  ax.spines['bottom'].set_color('#C0CBD5');ax.tick_params(axis='x',length=0,pad=10)
 axes[0].set_ylabel('Mean Thriller books per real user',fontsize=16,color=navy,labelpad=14)
 fig.subplots_adjust(left=.085,right=.975,bottom=.21,top=.72,wspace=.12)
 fig.text(.055,.943,'Thriller recommendations after synthetic-user injection',fontsize=26,weight='bold',color=navy)
 fig.text(.055,.892,'Controlled model runs • Baseline and 12 injection sizes • Top-15 recommendations',fontsize=13,color=muted)
 fig.legend(handles,['Baseline']+[f'{n} users' for n in COUNTS[1:]],loc='upper center',bbox_to_anchor=(.51,.856),ncol=7,frameon=False,fontsize=12,columnspacing=1.25,handlelength=1.3,handletextpad=.6)
 fig.text(.055,.075,'Mean over 53,424 real users; previously rated books excluded. Scores use learned factors and item biases.',fontsize=12.5,color=navy)
 fig.text(.055,.037,'Shared linear y-axis. Synthetic users: Thriller ratings = 5; other-book ratings = 1. Ties use ascending book ID.',fontsize=11.5,color=muted)
 fig.savefig(PNG,dpi=240,facecolor='white');plt.close(fig)

if __name__=='__main__':
 # Test boundary ties against a full stable sort before computing real data.
 rng=np.random.RandomState(42);s=rng.randint(-2,4,(7,30)).astype(float);mask=np.zeros_like(s,dtype=bool);mask[:,0]=True;t=rng.rand(30)>.5
 ref=s.copy();ref[mask]=-np.inf;expected=t[np.argsort(-ref,axis=1,kind='stable')[:,:15]].sum(1)
 assert np.array_equal(count_top(s.copy(),mask,t),expected)
 if TABLE.exists():
  data=pd.read_csv(TABLE)
  extension=pd.read_csv(L/'tables/extension_2500_3000_recommendation_counts.csv')
  new=extension[(extension.genre=='Thriller')&(extension.k==15)&(extension.n_fake.isin([2500,3000]))]
  assert len(new)==4
  old=data[~data.n_fake.isin([2500,3000])].copy()
  data=pd.concat([old,new],ignore_index=True).sort_values(['model','n_fake']).reset_index(drop=True)
  check=data[data.n_fake<=2000].sort_values(['model','n_fake']).reset_index(drop=True)
  pd.testing.assert_frame_equal(check,old.sort_values(['model','n_fake']).reset_index(drop=True))
  data.to_csv(TABLE,index=False)
 else:
  with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,['nmf','svd']))
  data=pd.DataFrame([r for a in results for r in a]);data.to_csv(TABLE,index=False)
 assert len(data)==26 and set(data.k)=={15}
 plot(data);print('SAVED',PNG,flush=True)
