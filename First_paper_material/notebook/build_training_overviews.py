from pathlib import Path
import os
os.environ['MPLBACKEND']='Agg'
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path('/home/moshtasa/Research/phd-recsys/First_paper_material')
source=pd.read_csv(R/'notebook/three_genre_panel_independent_components.csv')
ns=[0,2,4,6,25,50,100,200,500];x=np.arange(len(ns));colors={'svd':'#0072B2','nmf':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#334755','text.color':'#163248'})
for genre,folder in [('Thriller','thriller'),('Adventure','advanture'),('Drama','drama')]:
 f=source[(source.genre==genre)&(source.view=='all_training')]
 fig,axs=plt.subplots(2,2,figsize=(11,7.2))
 panels=[('mae','(A) Training error (MAE)','Mean absolute error'),('mean_bi',r'(B) Book bias $b_i$','Mean raw score component'),('mean_interaction',r'(C) Dot product $p_u^\top q_i$','Mean raw score component'),('mean_prediction',r'(D) Predicted rating $\hat r_{ui}$','Mean raw predicted rating')]
 for ax,(key,title,ylabel) in zip(axs.flat,panels):
  for model in ['svd','nmf']:
   for group,style in [('target','-'),('non_target','--')]:
    d=f[(f.model==model)&(f.group==group)].set_index('n_fake').loc[ns]
    assert len(d)==9 and np.isfinite(d[key]).all()
    assert np.allclose(d.mean_prediction,d.mu+d.mean_bu+d.mean_bi+d.mean_interaction,atol=1e-10)
    ax.plot(x,d[key],color=colors[model],ls=style,lw=2.3 if key=='mean_prediction' else 1.9,marker='o' if group=='target' else 's',ms=3.2)
  if key=='mean_prediction':
   for group,style in [('target','-'),('non_target','--')]:
    a=f[(f.model=='svd')&(f.group==group)].set_index('n_fake').loc[ns].mean_recorded_rating
    b=f[(f.model=='nmf')&(f.group==group)].set_index('n_fake').loc[ns].mean_recorded_rating
    assert np.allclose(a,b,atol=1e-12)
    ax.plot(x,a,color='#777777',lw=1.25,ls=style,alpha=.85,zorder=2)
  ax.set_title(title,loc='left',pad=10);ax.set_ylabel(ylabel)
  if key in ['mean_bi','mean_interaction']:ax.axhline(0,color='#96A4AE',lw=.7)
  if key=='mae':ax.set_ylim(bottom=0)
  ax.set_xticks(x,[str(n) for n in ns]);ax.tick_params(axis='x',labelsize=9);ax.set_xlabel('Injected users');ax.grid(axis='y',color='#E2E8EC',lw=.65);ax.set_axisbelow(True);ax.margins(x=.04)
 fig.suptitle(genre+': training fit and score components | All users',x=.08,ha='left',y=.98,fontsize=15,fontweight='bold')
 handles=[Line2D([0],[0],color=colors['svd'],lw=2.5,label='SVD'),Line2D([0],[0],color=colors['nmf'],lw=2.5,label='NMF'),Line2D([0],[0],color='#536471',lw=1.8,marker='o',ms=4,label=genre),Line2D([0],[0],color='#536471',lw=1.8,ls='--',marker='s',ms=4,label='Non-'+genre),Line2D([0],[0],color='#777777',lw=1.25,label='Recorded rating (D)')]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.935),ncol=5,frameon=False,columnspacing=1.7,fontsize=9)
 fig.text(.5,.018,'Real + synthetic training ratings; averages weighted by observed rating pairs. A: mean individual absolute error. D: grey = mean recorded rating.',ha='center',fontsize=7.8,color='#536471')
 fig.subplots_adjust(left=.08,right=.98,bottom=.11,top=.82,hspace=.48,wspace=.27)
 out=R/'figures/first_reviewer'/folder/('training_'+genre.lower()+'_all_users_overview_4panel.png')
 fig.savefig(out,dpi=300,facecolor='white');plt.close(fig);print(out)

