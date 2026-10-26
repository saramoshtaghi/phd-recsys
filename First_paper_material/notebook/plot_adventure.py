#!/usr/bin/env python3
"""Publish complete Adventure sweep figures, preserving every dose through 3500."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,textwrap
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
R=Path('/home/moshtasa/Research/phd-recsys');NB=R/'First_paper_material/notebook';F=R/'First_paper_material/figures';FR=F/'first_reviewer';ROOT=R/'Book/current/PerturbRec/svd_nmf';F1=ROOT/'layer1/result/final_results';F2=ROOT/'layer2/final_results'
COUNTS=[0,2,4,6,25,50,100,200,500,1000,2000,2500,3000,3500];X=np.arange(len(COUNTS));LABELS=['Base']+[str(n) for n in COUNTS[1:]];MODELS=['svd','nmf'];COLORS={'svd':'#D55E00','nmf':'#0072B2'}
C={'mu':'#D7AA00','mean_bu':'#61B8E8','mean_bi':'#D55E00','mean_interaction':'#0072B2','mean_prediction':'#009E73','mae':'#D62728'};NAMES={'mu':r'$\mu$: global mean','mean_bu':r'$b_u$: user bias','mean_bi':r'$b_i$: book bias','mean_interaction':r'$p_u^\top q_i$: interaction','mean_prediction':r'$\hat r_{ui}$: prediction','mae':'MAE: training error'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':6.5,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
OUTPUTS=[]
def save(fig,name,directory=F,copy=None):
 fig.savefig(directory/(name+'.png'),dpi=400,bbox_inches='tight',facecolor='white');fig.savefig(directory/(name+'.pdf'),bbox_inches='tight',facecolor='white');OUTPUTS.append(str(directory/(name+'.png')))
 if copy:
  import shutil
  for suffix in ['.png','.pdf']:shutil.copy2(directory/(name+suffix),copy/(name+suffix))
 plt.close(fig)
def decorate(ax,ylabel=None):
 ax.set_xticks(X,LABELS,rotation=55,ha='right');ax.set_xlim(-.5,len(X)-.5);ax.set_xlabel('Injected users');ax.set_ylabel(ylabel or '');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
def series(rows,key,model,**filters):
 selected=[r for r in rows if r.get('model',model)==model and all(r.get(k)==v for k,v in filters.items())];z={r['n_fake']:r[key] for r in selected};return np.array([z[n] for n in COUNTS])
def components(rows,view,name,title,copy=F2):
 fig,axs=plt.subplots(2,2,figsize=(7.1,4.6),sharex=True)
 for row,group in enumerate(['target','non_target']):
  for col,m in enumerate(MODELS):
   ax=axs[row,col];subset=[r for r in rows if r['view']==view and r['group']==group and r['model']==m];ns=[n for n in COUNTS if any(r['n_fake']==n for r in subset)];xx=[COUNTS.index(n) for n in ns]
   for key in C:
    if key=='mae' and view=='real_unseen_prediction':continue
    vals=[next(r[key] for r in subset if r['n_fake']==n) for n in ns];ax.plot(xx,vals,color=C[key],lw=2 if key=='mean_prediction' else 1.15,ls='--' if key=='mae' else '-',marker='o',ms=2,label=NAMES[key])
   ax.set_title(f'{m.upper()} | '+('Adventure' if group=='target' else 'Non-Adventure'));decorate(ax,'Mean score / MAE' if col==0 and view!='real_unseen_prediction' else 'Mean score' if col==0 else '')
 fig.suptitle(title+'\n'+r'$\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i$',fontsize=10,y=1.03);h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.09),frameon=False,fontsize=7);fig.tight_layout();save(fig,name,FR,copy)
def all_eight(rows):
 fig,axs=plt.subplots(4,2,figsize=(7.1,6.4),sharex=True)
 for row,(view,group) in enumerate([(v,g) for v in ['all_training','synthetic_training'] for g in ['target','non_target']]):
  for col,m in enumerate(MODELS):
   ax=axs[row,col];ss=[r for r in rows if r['view']==view and r['group']==group and r['model']==m]
   ns=COUNTS if view=='all_training' else COUNTS[1:]
   for key in C:ax.plot([COUNTS.index(n) for n in ns],[next(r[key] for r in ss if r['n_fake']==n) for n in ns],color=C[key],lw=1.9 if key=='mean_prediction' else 1,marker='o',ms=1.8,ls='--' if key=='mae' else '-',label=NAMES[key])
   ax.set_title(f'({chr(97+row*2+col)}) {m.upper()} | '+('All users' if view=='all_training' else 'Synthetic only')+' | '+('Adventure' if group=='target' else 'Non-Adventure'),fontsize=8);decorate(ax,'Mean score / MAE' if col==0 else '');ax.set_xlabel('Injected users' if row==3 else '')
 fig.suptitle(r'$\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i$   |   Adventure injection',fontsize=10,y=1.01);h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.06),frameon=False,fontsize=7);fig.tight_layout(h_pad=.8);save(fig,'Adventure_training_all_and_synthetic_8panel_paper',FR,F2)
def bars(rec):
 for k in [15,25,35]:
  fig,axs=plt.subplots(1,2,figsize=(7.1,2.7),sharey=True)
  for ax,m in zip(axs,MODELS):
   vals=series(rec,'mean_target_count',m,k=k);ax.bar(X,vals,color=['#AAB0B6']+[COLORS[m]]*(len(X)-1),width=.72);ax.set_title(m.upper());decorate(ax,f'Mean Adventure books in Top-{k}' if m=='svd' else '')
  fig.tight_layout();save(fig,f'Adventure_bars_top{k}_up_to_3500',F1,F)
  fig,axs=plt.subplots(3,2,figsize=(7.1,5.7),sharex=True)
  for col,m in enumerate(MODELS):
   for row,(key,label) in enumerate([('mean_target_count',f'Mean Adventure books\nin Top-{k}'),('union_distinct_target_titles','Distinct Adventure titles\nacross all users'),('most_frequent_book_user_reach_percent','Users receiving the most\ncommon Adventure book (%)')]):
    ax=axs[row,col];ax.bar(X,series(rec,key,m,k=k),color=['#AAB0B6']+[COLORS[m]]*(len(X)-1),width=.72);decorate(ax,label if col==0 else '');ax.set_xlabel('Injected users' if row==2 else '');ax.set_title(m.upper() if row==0 else '')
  fig.tight_layout();save(fig,f'comparision - Adventure_top{k}_union_powerpoint',F1,F2)
def bias_plots(bias):
 genres=sorted(set(r['genre'] for r in bias));others=plt.get_cmap('tab20')(np.linspace(0,1,len(genres)))
 for key,name,label in [('mean_bi','Adventure_book_bi','Mean book bias '+r'$b_i$'),('mu_plus_bi','Adventure_score_offsets',r'Mean $\mu+b_i$')]:
  fig,axs=plt.subplots(1,2,figsize=(7.1,3.3),sharey=True)
  for ax,m in zip(axs,MODELS):
   for j,g in enumerate(genres):ax.plot(X,series(bias,key,m,genre=g),color='#D55E00' if g=='Adventure' else others[j],alpha=1 if g=='Adventure' else .55,lw=2.2 if g=='Adventure' else .85,label=g,zorder=5 if g=='Adventure' else 2)
   ax.set_title(m.upper(),pad=12);decorate(ax,label if m=='svd' else '')
  h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,-.15),ncol=5,frameon=False,fontsize=6.5);fig.tight_layout();save(fig,name,F,F2)
def reviewer(data,comp,swaps):
 for m in MODELS:
  fig,axs=plt.subplots(1,4,figsize=(10.8,2.65));syn=[r for r in comp if r['model']==m and r['view']=='synthetic_training' and r['group']=='non_target'];xx=X[1:]
  axs[0].plot(xx,[r['negative_interaction_percent'] for r in sorted(syn,key=lambda r:r['n_fake'])],color=COLORS[m],marker='o',ms=2);axs[0].set_title('(a) Negative interactions');decorate(axs[0],'Synthetic × non-Adventure (%)')
  for key in ['mean_prediction','mean_interaction','mean_bi','mean_bu','mu']:
   axs[1].plot(xx,[r[key] for r in sorted(syn,key=lambda r:r['n_fake'])],color=C[key],lw=2 if key=='mean_prediction' else 1,label=NAMES[key])
  axs[1].axhline(1,color='black',ls=':',lw=.8);axs[1].set_title('(b) Fitting injected rating 1');decorate(axs[1],'Mean score components')
  ss=[r for r in comp if r['model']==m and r['view']=='real_unseen_prediction'];tar={r['n_fake']:r for r in ss if r['group']=='target'};non={r['n_fake']:r for r in ss if r['group']=='non_target'}
  for key,label in [('mean_bi','Book-bias gap'),('mean_interaction','Interaction gap'),('mean_prediction','Total score gap')]:axs[2].plot(X,[tar[n][key]-non[n][key] for n in COUNTS],color=C[key],lw=2 if key=='mean_prediction' else 1,label=label)
  axs[2].set_title('(c) Real unseen score gap');decorate(axs[2],'Adventure − non-Adventure')
  for key,color,label in [('item_bias_swap_allocation',C['mean_bi'],'Book-bias allocation'),('embedding_swap_allocation',C['mean_interaction'],'Vector allocation'),('delta_count',C['mean_prediction'],'Total Top-15 change')]:axs[3].plot(xx,series(swaps,key,m,k=15)[1:] if False else [next(r[key] for r in swaps if r['model']==m and r['n_fake']==n and r['k']==15) for n in COUNTS[1:]],color=color,label=label,lw=2 if key=='delta_count' else 1)
  axs[3].axhline(0,color='gray',lw=.6);axs[3].set_title('(d) Component swaps');decorate(axs[3],'Change in mean Adventure books')
  for ax in axs[1:]:ax.legend(fontsize=5.8,frameon=False)
  fig.suptitle(m.upper()+' | Adventure injection',y=1.04,fontsize=11);fig.tight_layout(w_pad=1);save(fig,f'{m.upper()}_Adventure_reviewer_hypothesis_4panel',FR,F2)
def neighborhood(data):
 for m in MODELS:
  ds=sorted([r for r in data if r['model']==m],key=lambda r:r['n_fake']);genres=list(ds[0]['neighbors'][0]['genre_mean_books_top15'])
  for metric in ['cosine','euclidean']:
   rows=[next((v for v in r['neighbors'] if v['metric']==metric and v.get('status')!='undefined'),None) for r in ds]
   if any(r is None for r in rows):continue
   fig,(a,b)=plt.subplots(2,1,figsize=(8.5,5.8),gridspec_kw={'height_ratios':[1,2]});replaced=np.array([r['mean_turnover']*100 for r in rows]);a.bar(X,100-replaced,color='#56B4E9',label='Retained');a.bar(X,replaced,bottom=100-replaced,color='#E69F00',label='Replaced')
   a.set_ylim(0,100);a.set_title(m.upper()+' | 20 closest real users | '+metric,pad=10);decorate(a,'Neighbors (%)');a.legend(ncol=2,frameon=False,fontsize=7)
   matrix=np.array([[r['genre_mean_books_top15'][g] for r in rows] for g in genres]);v=max(float(np.abs(matrix-matrix[:,[0]]).max()),.01);delta=matrix-matrix[:,[0]]
   rgba=plt.get_cmap('RdBu_r')(TwoSlopeNorm(vmin=-v,vcenter=0,vmax=v)(delta));targetrow=genres.index('Adventure')
   for j in range(len(genres)):
    if j!=targetrow:rgba[j,:,:3]=.35+.65*rgba[j,:,:3]
   b.imshow(rgba,aspect='auto');b.set_xticks(X,LABELS,rotation=55,ha='right');b.set_yticks(range(len(genres)),genres)
   for tick in b.get_yticklabels():
    if tick.get_text()=='Adventure':tick.set_weight('bold');tick.set_fontsize(9)
   for i in range(len(genres)):
    for j in range(len(COUNTS)):b.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=6.8 if i==targetrow else 5.8,fontweight='bold' if i==targetrow else 'normal',color='black')
   from matplotlib.patches import Rectangle
   b.add_patch(Rectangle((-.5,targetrow-.5),len(X),1,fill=False,edgecolor='black',lw=1.2));b.set_xlabel('Injected users');b.set_title('Mean genre books in each neighbor’s Top-15',pad=10);fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-v,vcenter=0,vmax=v),cmap='RdBu_r'),ax=b,label='Change from baseline (books)',pad=.02,fraction=.025);fig.tight_layout();save(fig,f'{m.upper()}_Adventure_neighborhood_recommendation_change_heatmap_up_to_3500_{metric}',F2)
 fig,axs=plt.subplots(1,2,figsize=(7.1,2.7))
 for m in MODELS:
  ds=sorted([r for r in data if r['model']==m],key=lambda r:r['n_fake']);axs[0].plot(X,[r['procrustes_aligned_mean_user_cosine'] for r in ds],color=COLORS[m],label=m.upper(),marker='o',ms=2)
  ns=[next(v for v in r['neighbors'] if v['metric']=='cosine') for r in ds]
  for key,ls in [('retained_after_contribution','-'),('incoming_after_contribution','--')]:axs[1].plot(X,[v.get(key,np.nan) for v in ns],color=COLORS[m],ls=ls,label=m.upper()+(' retained' if ls=='-' else ' incoming'))
 decorate(axs[0],'Mean aligned user cosine');decorate(axs[1],'Mean Adventure books in Top-15');axs[0].legend(frameon=False);axs[1].legend(frameon=False,fontsize=6);fig.tight_layout();save(fig,'Adventure_cosine_and_neighbor_contributions',F2)
def personalization(data,rec):
 fig,axs=plt.subplots(1,3,figsize=(9,2.8))
 for m in MODELS:
  ds=sorted([r for r in data if r['model']==m],key=lambda r:r['n_fake']);axs[0].plot(X,series(rec,'mean_target_count',m,k=15),label=m.upper(),color=COLORS[m],marker='o',ms=2);axs[1].plot(X,series(rec,'most_frequent_book_user_reach_percent',m,k=15),color=COLORS[m],marker='o',ms=2);axs[2].plot(X,[r['mean_pairwise_recommendation_jaccard_top15'] for r in ds],color=COLORS[m],marker='o',ms=2)
 for ax,title,label in zip(axs,['(a) Target exposure','(b) One-book concentration','(c) Recommendation overlap'],['Mean Adventure books in Top-15','Users receiving most common\nAdventure book (%)','Mean pairwise Top-15 Jaccard']):ax.set_title(title);decorate(ax,label)
 axs[0].legend(frameon=False);fig.tight_layout();save(fig,'Adventure_personalization_3panel',F2,F)
 rows=[dict(model=r['model'],n_fake=r['n_fake'],**book) for r in data for book in r['frequent_books']];pd.DataFrame(rows).to_csv(NB/'adventure_frequent_books_with_genres.csv',index=False)
 lines=['# Adventure: recommendation examples and original book popularity','All rankings use raw scores, exclude already-rated books, and concern original users.']
 for m in MODELS:
  for n in [0,500,2000,3500]:
   d=next(r for r in data if r['model']==m and r['n_fake']==n);lines+=['',f'## {m.upper()}, {n} injected users']
   for book in [v for v in d['frequent_books'] if v['k']==15][:8]:lines.append(f"- {book['title']} | {book['genres']} | recommended to {book['reach_percent']:.2f}% of users | original ratings: {book['original_rating_count']}; original mean: {book['original_mean_rating']:.2f}")
 (NB/'adventure_book_examples.md').write_text('\n'.join(lines))
def report(data,rec):
 lines=['# Adventure controlled experiment: completed sweep','', 'Counts: '+', '.join(map(str,COUNTS))+'.','53,424 original users; 9,415 books; 5,815,569 original ratings; 1,789 Adventure and 7,626 non-Adventure books.','Each synthetic user rates every Adventure book 5 and every other book 1.','Surprise biased SVD/NMF: 8 factors, 180 epochs, seed 42; identical common-entity initial factors within model.','No clipping of scores, factors, or injection range. Rankings use raw scores and unseen books.','']
 for m in MODELS:
  s=[r for r in rec if r['model']==m and r['k']==15];a=next(r for r in s if r['n_fake']==0);z=next(r for r in s if r['n_fake']==3500);peak=max(s,key=lambda r:r['mean_target_count']);lines+=[f"{m.upper()}: mean Adventure books in Top-15: {a['mean_target_count']:.4f} at baseline, {z['mean_target_count']:.4f} at 3500; maximum {peak['mean_target_count']:.4f} at {peak['n_fake']}. No monotonicity assumed."]
 invalid=[f"{r['model'].upper()} n={r['n_fake']} (min pu={r['min_pu']}, min qi={r['min_qi']})" for r in data if not r['nmf_nonnegative_constraint_ok']];lines+=['','Non-negative factor check: '+('; '.join(invalid) if invalid else 'all NMF checkpoints passed.'),'','Interpretation and limits:','- All-training means weight observed rating pairs; adding dense fake profiles changes that mixture. Synthetic-only views isolate which records are measured, not a separate model.','- Red MAE measures reconstruction of training ratings; it is not held-out recommendation accuracy.','- Real-unseen components average available unseen candidates per user, then users. They describe score components; actual Top-K counts are separately computed from complete rankings.','- Within-user item ranking depends on book bias plus user–book interaction. Global mean and user bias are identical offsets across a user’s candidates.','- Component swaps are conditional replacements in fitted models, with a symmetric split of the interaction between replacements. They do not identify a unique causal training mechanism.','- Neighbors are the 20 closest other real users, compared with baseline. Recommendation genre counts are measured on the neighbors’ own current Top-15 lists, not actual human taste changes. Overlapping genre labels can sum above 15.','- Heatmap text gives absolute mean books. Color gives increase/decrease from baseline; non-target rows are visually muted only.','- UMAP is a joint projection of aligned before/after vectors at 3500, not evidence of causal movement.','- One controlled training seed; no universal ranking of algorithms is claimed.']
 (NB/'adventure_results_README.md').write_text('\n'.join(lines))
if __name__=='__main__':
 print('Waiting for all Adventure analyses through 3500',flush=True)
 while True:
  try:
   status=json.loads((NB/'adventure_analysis_status.json').read_text())
   if status.get('failures'):raise RuntimeError(status['failures'])
   if status.get('status')=='complete':break
  except FileNotFoundError:pass
  time.sleep(15)
 data=[json.loads((NB/f'adventure_{m}_{n}_analysis.json').read_text()) for m in MODELS for n in COUNTS];comp=[r for d in data for r in d['components']];rec=[r for d in data for r in d['recommendations']];bias=[r for d in data for r in d['genre_biases']];swaps=[dict(model=d['model'],n_fake=d['n_fake'],**r) for d in data for r in d['component_swaps']]
 bars(rec);bias_plots(bias);components(comp,'all_training','Adventure_training_all_users_thriller_non_thriller_4panel'.replace('thriller_non_thriller','adventure_non_adventure'),'All observed training ratings');components(comp,'synthetic_training','Adventure_training_synthetic_users_adventure_non_adventure_4panel','Synthetic users’ observed training ratings');components(comp,'real_unseen_prediction','Adventure_prediction_real_users_adventure_non_adventure_4panel','Original users’ unseen candidate scores');all_eight(comp);reviewer(data,comp,swaps);neighborhood(data);personalization(data,rec);report(data,rec)
 (NB/'adventure_figure_manifest.json').write_text(json.dumps(dict(status='complete',counts=COUNTS,figures=OUTPUTS,completed=time.time()),indent=2));print('ALL ADVENTURE FIGURES COMPLETE',len(OUTPUTS),flush=True)
