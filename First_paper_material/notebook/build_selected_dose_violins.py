import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import json,gc
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R=Path('/home/moshtasa/Research/phd-recsys');M=R/'First_paper_material';NB=M/'notebook';OUT=M/'figures'
DOSES={'Books':500,'Movies':10,'Mobile apps':60}
CATEGORIES={'Books':['Thriller','Drama','Adventure'],'Movies':['1990s','1950s','1920s'],'Mobile apps':['Shopping','Casual','Beauty']}
COLORS={('svd','target'):'#0072B2',('svd','non_target'):'#8CC5E5',('nmf','target'):'#D55E00',('nmf','non_target'):'#F2AC79'}
LABELS={('svd','target'):'SVD · target',('svd','non_target'):'SVD · other',('nmf','target'):'NMF · target',('nmf','non_target'):'NMF · other'}
STATS=[];DISTS={}

def book_df():
 p=R/'Book/Archive/data/genre_clean.csv'
 return pd.read_csv(p,usecols=['user_id','book_id','rating','genres'],dtype={'user_id':'int64','book_id':'int64','rating':'float32','genres':'string'})
def movie_df():
 p=R/'Movie /archive/preprocessing/df_final.csv'
 d=pd.read_csv(p,usecols=['user_id','item_id','rating','decade'])
 d=d.dropna(subset=['user_id','item_id','rating']);d['decade']=pd.to_numeric(d.decade,errors='coerce').fillna(0).astype(int).astype(str)+'s'
 return d
def mobile_df():
 p=R/'Mobile/archive/0316_Single/data/data/mobilerec_final.csv'
 d=pd.read_csv(p,usecols=['uid','app_package','rating','app_category'],dtype={'uid':str,'app_package':str})
 d=d.dropna(subset=['uid','app_package','rating']);d['rating']=pd.to_numeric(d.rating,errors='coerce').fillna(0).clip(0,5).astype('float32');d['app_category']=d.app_category.fillna('').astype(str).str.strip()
 return d

def setup(domain):
 if domain=='Books':
  df=book_df();u_col='user_id';i_col='book_id';target_col='genres'
  metadata=df[[i_col,target_col]].drop_duplicates(i_col).set_index(i_col)[target_col]
 elif domain=='Movies':
  df=movie_df();u_col='user_id';i_col='item_id';target_col='decade'
  metadata=df[[i_col,target_col]].drop_duplicates(i_col).set_index(i_col)[target_col]
 else:
  df=mobile_df();u_col='uid';i_col='app_package';target_col='app_category'
  metadata=df[[i_col,target_col]].drop_duplicates(i_col).set_index(i_col)[target_col]
 raw_users=pd.unique(df[u_col]);raw_items=pd.unique(df[i_col])
 raw_u_idx={v:i for i,v in enumerate(raw_users)};raw_i_idx={v:i for i,v in enumerate(raw_items)}
 ui=np.fromiter((raw_u_idx[x] for x in df[u_col].to_numpy()),dtype=np.int32,count=len(df))
 ii=np.fromiter((raw_i_idx[x] for x in df[i_col].to_numpy()),dtype=np.int32,count=len(df))
 seen=csr_matrix((np.ones(len(df),dtype=np.uint8),(ui,ii)),shape=(len(raw_users),len(raw_items)))
 seen.sum_duplicates();seen.data[:]=1;seen.eliminate_zeros()
 item_labels=np.asarray([str(metadata.get(x,'')) for x in raw_items],dtype=object)
 if domain=='Books':
  label_sets=[set(s.split(', ')) if s!='<NA>' else set() for s in item_labels]
 else: label_sets=[{s} if s else set() for s in item_labels]
 del ui,ii
 return df,u_col,i_col,raw_users,raw_items,raw_u_idx,raw_i_idx,seen,label_sets

def factor_path(domain,category,model,n):
 if domain=='Books':
  folder='adventure' if category=='Adventure' else category.lower()
  if category=='Adventure': return R/f'Book/current/PerturbRec/svd_nmf/layer2/embeddings/controlled/{model}/{model}_adventure_{n}.npz'
  return R/f'Book/current/PerturbRec/svd_nmf/layer2/embeddings/controlled/{model}/{folder}/{model}_{folder}_{n}.npz'
 if domain=='Movies': return NB/f'movies_{category[:-1]}_{model}_{n}_factors.npz'
 prefix=category.lower()
 return NB/f'{prefix}_{model}_{n}_factors.npz'

def indices_for_checkpoint(domain,model,n,raw_users,raw_items):
 # Book checkpoints carry raw IDs, so align observations by ID. Movie/mobile checkpoints
 # use the same first-seen Surprise inner-ID order as the source training scripts.
 if domain=='Books': return None
 return np.arange(len(raw_users),dtype=np.int64),np.arange(len(raw_items),dtype=np.int64)

def calculate(domain,category,model,n,ctx):
 df,u_col,i_col,raw_users,raw_items,raw_u_idx,raw_i_idx,seen,label_sets=ctx
 path=factor_path(domain,category,model,n)
 with np.load(path) as z:
  pu=z['pu'];qi=z['qi'];bu=z['bu'];bi=z['bi'];mu=float(z['mu'])
  if domain=='Books':
   factor_users=z['ids'];factor_items=z['items']
   umap={int(v):i for i,v in enumerate(factor_users)};imap={int(v):i for i,v in enumerate(factor_items)}
   ur=np.asarray([umap[int(v)] for v in raw_users],dtype=np.int64)
   ir=np.asarray([imap[int(v)] for v in raw_items],dtype=np.int64)
   mat=csr_matrix((seen.data,(ur[seen.tocoo().row],ir[seen.tocoo().col])),shape=(len(pu),len(qi)))
   mat.sum_duplicates();mat.data[:]=1;mat.eliminate_zeros();mat=mat[ur];real_idx=np.arange(len(raw_users),dtype=np.int64)
  else:
   assert pu.shape[0]>=len(raw_users) and qi.shape[0]==len(raw_items),(domain,category,model,pu.shape,qi.shape,len(raw_users),len(raw_items))
   mat=seen;real_idx=np.arange(len(raw_users),dtype=np.int64)
  assert np.isfinite(pu).all() and np.isfinite(qi).all() and np.isfinite(bu).all() and np.isfinite(bi).all()
  target=np.asarray([category in s for s in label_sets],dtype=bool)
  masks={'target':target,'non_target':~target}
  for group,mask in masks.items():
   idx=np.flatnonzero(mask)
   obs=mat[:,idx]
   counts=np.asarray(obs.sum(axis=1)).ravel().astype(np.int64)
   avail=len(idx)-counts
   qsum=qi[idx].sum(axis=0)-obs@qi[idx]
   bisum=float(bi[idx].sum())-np.asarray(obs@bi[idx]).ravel()
   valid=avail>0
   pu_real=pu[real_idx]
   interaction=np.einsum('ij,ij->i',pu_real,qsum)/np.maximum(avail,1)
   item_bias=bisum/np.maximum(avail,1)
   scores=mu+bu[real_idx]+item_bias+interaction
   scores=scores[valid]
   key=(domain,category,model,group)
   DISTS[key]=scores.astype(np.float32)
   source=pd.read_csv(NB/('three_genre_panel_independent_components.csv' if domain=='Books' else f"movies_{category[:-1]}_reviewer_components.csv" if domain=='Movies' else f'{category.lower()}_reviewer_components.csv'))
   if domain=='Books': source=source[source.genre.astype(str).str.casefold()==category.casefold()]
   match=source[(source.model==model)&(source.n_fake==n)&(source.view=='real_unseen_prediction')&(source.group==group)]
   if len(match)==1:
    expected=float(match.iloc[0].mean_prediction);computed=float(scores.mean())
    if not np.isclose(computed,expected,rtol=1e-5,atol=1e-5):
     raise ValueError(f'mean mismatch {key}: computed {computed}, source {expected}; factor order/data mismatch')
   else: raise ValueError(f'Missing reference row: {key}')
   STATS.append(dict(domain=domain,category=category,n_fake=n,model=model,group=group,n_users=len(scores),mean=float(scores.mean()),median=float(np.median(scores)),q05=float(np.quantile(scores,.05)),q95=float(np.quantile(scores,.95)),min=float(scores.min()),max=float(scores.max()),n_below_0=int(np.sum(scores<0)),n_above_7=int(np.sum(scores>7)),factor_file=str(path)))
 del df

for domain,categories in CATEGORIES.items():
 print('LOADING',domain,flush=True);ctx=setup(domain)
 for category in categories:
  n=DOSES[domain]
  for model in ['svd','nmf']:
   print('CALCULATE',domain,category,n,model,flush=True);calculate(domain,category,model,n,ctx)
 del ctx;gc.collect()

pd.DataFrame(STATS).to_csv(NB/'selected_dose_prediction_user_distributions.csv',index=False)
# Draw one violin per model × category group; every distribution is derived from
# each real user's mean score over that user's unrated books/movies/apps in a group.
fig,axs=plt.subplots(3,1,figsize=(11.5,11.2),sharex=False)
order=[('svd','target'),('svd','non_target'),('nmf','target'),('nmf','non_target')]
for ax,(domain,categories) in zip(axs,CATEGORIES.items()):
 positions=[];values=[];colors=[]
 for ci,category in enumerate(categories):
  for offset,key in zip([-.30,-.10,.10,.30],order):
   dist=DISTS[(domain,category,key[0],key[1])]
   if len(dist)>20000:
    rng=np.random.default_rng(8675309+ci+len(domain))
    sample=dist[rng.choice(len(dist),20000,replace=False)]
   else: sample=dist
   positions.append(ci+offset);values.append(sample);colors.append(COLORS[key])
 vp=ax.violinplot(values,positions=positions,widths=.17,showmeans=False,showmedians=True,showextrema=False,bw_method='scott')
 for body,color in zip(vp['bodies'],colors): body.set_facecolor(color);body.set_edgecolor('#35424C');body.set_linewidth(.45);body.set_alpha(.72)
 vp['cmedians'].set_color('#263238');vp['cmedians'].set_linewidth(.75)
 ax.set_xticks(np.arange(3),categories)
 ax.set_ylim(0,7)
 ax.set_xlim(-.55,2.55);ax.grid(axis='y',color='#E1E7EB',lw=.55);ax.set_axisbelow(True)
 ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
 ax.set_title(f'{domain} · {DOSES[domain]} synthetic users',loc='left',fontweight='bold',fontsize=10)
 ax.set_ylabel('Per-user mean predicted rating')
handles=[Patch(facecolor=COLORS[k],edgecolor='#35424C',alpha=.72,label=LABELS[k]) for k in order]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),ncol=4,frameon=False,fontsize=8)
fig.suptitle('Per-User Prediction Distributions for Unrated Items',fontsize=14,fontweight='bold',y=1.025)
fig.text(.5,.012,'Per-user mean scores for unrated items; center line = median. Axes show 0–7; values outside are clipped in the figure and retained in the stats CSV.',ha='center',fontsize=7.6,color='#4D5B66')
fig.subplots_adjust(left=.09,right=.99,top=.95,bottom=.055,hspace=.42)
out=OUT/'prediction_real_users_selected_dose_violins.png';fig.savefig(out,dpi=500,bbox_inches='tight');plt.close(fig)
(R/'First_paper_material/notebook/selected_dose_violin_validation.json').write_text(json.dumps({'figure':str(out),'doses':DOSES,'categories':CATEGORIES,'models':['svd','nmf'],'views':'real_unseen_prediction','per_user_means_validated_against_existing_component_means':True,'user_sampling_for_density':20000,'stats':STATS},indent=2))
print('SAVED',out,flush=True)
for x in STATS: print(x['domain'],x['category'],x['model'],x['group'],'mean',round(x['mean'],4),'median',round(x['median'],4),'q05q95',round(x['q05'],3),round(x['q95'],3),'minmax',round(x['min'],3),round(x['max'],3),flush=True)
