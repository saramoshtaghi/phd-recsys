"""Five-panel prediction views: Top-15 target count plus fitted score components."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,time,json
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path('/home/moshtasa/Research/phd-recsys');NB=R/'First_paper_material/notebook'
category=sys.argv[1];movie=category in ['1990','1950'];prefix='movies_'+category if movie else category.lower();folder=R/'First_paper_material/figures/first_reviewer'/category.lower()
if not movie:
 while True:
  p=NB/f'{prefix}_reviewer_status.json'
  s=json.loads(p.read_text()) if p.exists() else {}
  if s.get('stage')=='failed':raise RuntimeError(s)
  if s.get('stage')=='all measurements complete':break
  time.sleep(30)
source=NB/f'{prefix}_reviewer_components.csv';f=pd.read_csv(source);f=f[f.view=='real_unseen_prediction'];ns=sorted(f.n_fake.unique())
if movie:
 df=pd.read_csv(R/'Movie /archive/preprocessing/df_final.csv',dtype={'user_id':str,'item_id':str});u='user_id';i='item_id';g='decade';df[g]=pd.to_numeric(df[g],errors='coerce').fillna(0).astype(int).astype(str)
else:
 df=pd.read_csv(R/'Mobile/archive/0316_Single/data/data/mobilerec_final.csv',usecols=['uid','app_package','rating','app_category'],dtype={'uid':str,'app_package':str});u='uid';i='app_package';g='app_category';df[g]=df[g].fillna('').astype(str).str.strip()
df=df.dropna(subset=[u,i,'rating']);users=pd.Index(df[u].unique());items=pd.Index(df[i].unique());meta=df.drop_duplicates(i).set_index(i)[g].reindex(items);target=meta.eq(category).to_numpy();seen=csr_matrix((np.ones(len(df),bool),(users.get_indexer(df[u]),items.get_indexer(df[i]))),shape=(len(users),len(items)))
countsfile=NB/f'{prefix}_top15_diagnostic_counts.csv';rows=pd.read_csv(countsfile).to_dict('records') if countsfile.exists() else []
for model in ['svd','nmf']:
 for n in ns:
  if any(r['model']==model and r['n_fake']==n for r in rows):continue
  a=np.load(NB/f'{prefix}_{model}_{n}_factors.npz');P=a['pu'][:len(users)];Q=a['qi'];bi=a['bi'];assert len(Q)==len(items)
  total=0
  for st in range(0,len(users),128):
   score=P[st:st+128]@Q.T+bi
   sm=seen[st:st+128].tocoo();score[sm.row,sm.col]=-np.inf
   ix=np.argpartition(-score,14,axis=1)[:,:15];total+=int(target[ix].sum())
  rows.append(dict(model=model,n_fake=n,n_users=len(users),mean_target_count=total/len(users)))
  pd.DataFrame(rows).to_csv(countsfile,index=False);print(category,model,n,total/len(users),flush=True)
rec=pd.DataFrame(rows);C={'svd':'#0072B2','nmf':'#D55E00'};x=np.arange(len(ns));label=category+'s' if movie else category;item='Movie' if movie else 'App'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(10,8));grid=fig.add_gridspec(3,4);axs=[fig.add_subplot(grid[0,:2]),fig.add_subplot(grid[0,2:]),fig.add_subplot(grid[1,:2]),fig.add_subplot(grid[1,2:]),fig.add_subplot(grid[2,1:3])]
for j,m in enumerate(C):axs[0].bar(x+(-.19 if j==0 else .19),rec[rec.model==m].set_index('n_fake').loc[ns].mean_target_count,width=.36,color=C[m],zorder=3)
axs[0].set_title(f'(A) {label} items in Top-15',loc='left');axs[0].set_ylabel('Mean target items');axs[0].set_ylim(0,15)
for ax,(key,title) in zip(axs[1:],[('mean_bi',f'(B) {item} bias $b_i$'),('mean_interaction',r'(C) Dot product $p_u^\top q_i$'),('mean_bu',r'(D) User bias $b_u$'),('mean_prediction',r'(E) Predicted rating $\hat r_{ui}$')]):
 for m in C:
  for group,ls,mk in [('target','-','o'),('non_target','--','s')]:
   d=f[(f.model==m)&(f.group==group)].set_index('n_fake').loc[ns];assert np.allclose(d.mean_prediction,d.mu+d.mean_bi+d.mean_bu+d.mean_interaction);ax.plot(x,d[key],color=C[m],ls=ls,marker=mk,ms=3,lw=2 if key=='mean_prediction' else 1.5)
 ax.set_title(title,loc='left');ax.set_ylabel('Mean predicted rating' if key=='mean_prediction' else 'Mean score component')
for ax in axs:ax.set_xticks(x,[str(n) for n in ns]);ax.set_xlabel('Injected users');ax.set_axisbelow(True);ax.grid(axis='y',color='#E2E8EC',lw=.6)
fig.suptitle(f'{label}: Recommendations and Score Components — Real Users ({len(users):,})',fontsize=12,fontweight='bold',y=.98)
handles=[Line2D([],[],color=C[m],label=m.upper()) for m in C]+[Line2D([],[],color='#536471',ls=ls,marker=mk,label=t) for t,ls,mk in [(label,'-','o'),('Non-'+label,'--','s')]]
fig.legend(handles=handles,ncol=4,loc='upper center',bbox_to_anchor=(.5,.95),frameon=False)
fig.subplots_adjust(left=.10,right=.98,bottom=.07,top=.87,hspace=.65,wspace=.7)
filename=f'prediction_real_users_{category.lower()}_non_{category.lower()}_4panel.png';folder.mkdir(exist_ok=True);fig.savefig(folder/filename,dpi=400);plt.close(fig)
print('SAVED',folder/filename,flush=True)
import subprocess
paths=[str((folder/filename).relative_to(R)),str(countsfile.relative_to(R))]
subprocess.run(['git','add','-f','--',*paths],cwd=R,check=True)
subprocess.run(['git','commit','-m',f'Add {category} five-panel prediction overview','--',*paths],cwd=R,check=True)
subprocess.run(['git','push'],cwd=R,check=True)
