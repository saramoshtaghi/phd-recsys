#!/usr/bin/env python3
"""End-point UMAP: all original users before/after, all 3500 injected users."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['NUMBA_NUM_THREADS']='2'
from pathlib import Path
import time,json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.linalg import orthogonal_procrustes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/moshtasa/Research/phd-recsys');L=R/'Book/current/PerturbRec/svd_nmf/layer2';NB=R/'First_paper_material/notebook';F=L/'final_results';N=3500

def run(model):
 while not (NB/f'adventure_{model}_{N}_training.json').exists():time.sleep(15)
 import umap
 tick=time.time();print('UMAP START',model,flush=True)
 base=np.load(L/'embeddings/pilot'/f'{model}_baseline_a.npz');z=np.load(L/'embeddings/controlled'/model/f'{model}_adventure_{N}.npz');ids=np.sort(base['ids']);sort=np.argsort(base['ids']);bp=base['pu'][sort];sort1=np.argsort(z['ids']);allids=z['ids'][sort1];pu=z['pu'][sort1];real=pu[np.searchsorted(allids,ids)];fake=pu[~np.isin(allids,ids)];assert len(fake)==3500 and len(real)==53424
 rotation,_=orthogonal_procrustes(real,bp);after=real@rotation;injected=fake@rotation;combined=np.vstack([bp,after,injected]);assert len(combined)==2*53424+3500
 # Cosine is undefined for zero vectors; do not alter factors to conceal that case.
 if np.any(np.linalg.norm(combined,axis=1)==0):raise ValueError('Cosine UMAP undefined: raw checkpoint contains zero vectors')
 coords=umap.UMAP(n_neighbors=20,min_dist=.1,metric='cosine',random_state=42,n_jobs=1,n_epochs=200).fit_transform(combined)
 n=len(ids);xy0=coords[:n];xy1=coords[n:2*n];xyfake=coords[2*n:];assert len(xyfake)==3500
 np.savez_compressed(L/'embeddings/controlled'/model/f'{model}_adventure_3500_umap.npz',ids=ids,before=xy0,after=xy1,synthetic=xyfake,rotation=rotation)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axs=plt.subplots(1,2,figsize=(7.1,3.1),sharex=True,sharey=True)
 for ax,xy,title in zip(axs,[xy0,xy1],['Baseline','3,500 injected users']):
  ax.scatter(xy[:,0],xy[:,1],s=.6,color='#0072B2',alpha=.24,linewidths=0,rasterized=True,label='Original users');ax.set_title(title,fontsize=9);ax.set_xlabel('UMAP 1')
 axs[0].set_ylabel('UMAP 2');axs[1].scatter(xyfake[:,0],xyfake[:,1],s=3,color='#D55E00',alpha=.65,linewidths=0,rasterized=True,label='Synthetic users');axs[1].legend(frameon=False,fontsize=7,markerscale=2);fig.suptitle(model.upper()+' | Adventure | jointly projected aligned user vectors',fontsize=10);fig.tight_layout()
 for extension in ['png','pdf']:
  path=F/f'{model.upper()}_Adventure_umap_before_after_3500.{extension}';fig.savefig(path,dpi=400,bbox_inches='tight',facecolor='white')
 plt.close(fig)
 meta=dict(model=model,n_fake=N,status='complete',n_real_users_before=n,n_real_users_after=n,n_synthetic_users=len(xyfake),full_user_population=True,alignment='orthogonal Procrustes',joint_projection=True,metric='cosine',n_neighbors=20,min_dist=.1,n_epochs=200,seed=42,elapsed_seconds=time.time()-tick,interpretation='Visualization only; different models use different projection coordinates; no causal movement claim.')
 (NB/f'adventure_{model}_umap_status.json').write_text(json.dumps(meta,indent=2));print('UMAP DONE',model,round(time.time()-tick),flush=True)
 return meta
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(run,['svd','nmf']))
