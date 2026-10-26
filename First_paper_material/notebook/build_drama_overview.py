from pathlib import Path
import os
os.environ['MPLBACKEND']='Agg'
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path('/home/moshtasa/Research/phd-recsys');NB=R/'First_paper_material/notebook'
f=pd.read_csv(NB/'three_genre_panel_independent_components.csv')
f=f[(f.genre=='Drama')&(f.view=='real_unseen_prediction')]
rec=pd.read_csv(R/'Book/current/PerturbRec/svd_nmf/layer1/tables/topk_target_union/summary.csv')
rec=rec[(rec.genre=='Drama')&(rec.k==15)]
ns=[0,2,4,6,25,50,100,200,500];x=np.arange(len(ns));colors={'svd':'#0072B2','nmf':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#334755','text.color':'#163248'})
fig,axs=plt.subplots(2,2,figsize=(11,7.2))
ax=axs[0,0]
for j,model in enumerate(['svd','nmf']):
 d=rec[rec.model==model].set_index('n_fake').loc[ns]
 assert len(d)==9 and (d.n_users==53424).all()
 ax.bar(x+(-.19 if j==0 else .19),d.mean_target_count,width=.36,color=colors[model])
ax.set_title('(A) Drama books in Top-15',loc='left',pad=10);ax.set_ylabel('Mean number of Drama books');ax.set_ylim(0,15)
for ax,(key,title) in zip([axs[0,1],axs[1,0],axs[1,1]],[('mean_bi',r'(B) Book bias $b_i$'),('mean_interaction',r'(C) Dot product $p_u^\top q_i$'),('mean_prediction',r'(D) Predicted rating $\hat r_{ui}$')]):
 for model in ['svd','nmf']:
  for group,style in [('target','-'),('non_target','--')]:
   d=f[(f.model==model)&(f.group==group)].set_index('n_fake').loc[ns]
   assert len(d)==9
   assert np.allclose(d.mean_prediction,d.mu+d.mean_bu+d.mean_bi+d.mean_interaction,atol=1e-10)
   ax.plot(x,d[key],color=colors[model],ls=style,lw=2.3 if key=='mean_prediction' else 1.9,marker='o' if group=='target' else 's',ms=3.2)
 ax.set_title(title,loc='left',pad=10);ax.set_ylabel('Mean raw score component' if key!='mean_prediction' else 'Mean raw predicted rating')
 if key!='mean_prediction':ax.axhline(0,color='#96A4AE',lw=.7)
for ax in axs.flat:
 ax.set_xticks(x,[str(n) for n in ns]);ax.tick_params(axis='x',labelsize=9);ax.set_xlabel('Injected users');ax.grid(axis='y',color='#E2E8EC',lw=.65);ax.set_axisbelow(True);ax.margins(x=.04)
fig.suptitle('Drama: recommendation outcomes and score components',x=.08,ha='left',y=.98,fontsize=15,fontweight='bold')
handles=[Line2D([0],[0],color=colors['svd'],lw=2.5,label='SVD'),Line2D([0],[0],color=colors['nmf'],lw=2.5,label='NMF'),Line2D([0],[0],color='#536471',lw=1.8,marker='o',ms=4,label='Drama'),Line2D([0],[0],color='#536471',lw=1.8,ls='--',marker='s',ms=4,label='Non-Drama')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.935),ncol=4,frameon=False,columnspacing=2.5)
fig.text(.5,.018,'A: Top-15 recommendations. B–D: all unrated books, averaged within each group for each real user, then across users.',ha='center',fontsize=8,color='#536471')
fig.subplots_adjust(left=.08,right=.98,bottom=.11,top=.82,hspace=.48,wspace=.27)
out=R/'First_paper_material/figures/first_reviewer/drama/drama_recommendation_and_components_4panel.png'
fig.savefig(out,dpi=300,facecolor='white');print(out)

