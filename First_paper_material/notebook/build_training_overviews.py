from pathlib import Path
import os
os.environ['MPLBACKEND']='Agg'
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path('/home/moshtasa/Research/phd-recsys/First_paper_material')
source=pd.read_csv(R/'notebook/three_genre_panel_independent_components.csv')
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--synthetic-only',action='store_true');args=ap.parse_args()
ns=([2,4,6,25,50,100,200,500] if args.synthetic_only else [0,2,4,6,25,50,100,200,500]);x=np.arange(len(ns));colors={'svd':'#0072B2','nmf':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#334755','text.color':'#163248'})
for genre,folder in [('Thriller','thriller'),('Adventure','advanture'),('Drama','drama')]:
 f=source[(source.genre==genre)&(source.view==('synthetic_training' if args.synthetic_only else 'all_training'))]
 fig=plt.figure(figsize=(11,9.6))
 grid=fig.add_gridspec(3,4)
 axs=[fig.add_subplot(grid[0,:2]),fig.add_subplot(grid[0,2:]),fig.add_subplot(grid[1,:2]),fig.add_subplot(grid[1,2:]),fig.add_subplot(grid[2,1:3])]
 panels=[('mae','(A) Training error (MAE)','Mean absolute error'),('mean_bi',r'(B) Book bias $b_i$','Mean raw score component'),('mean_interaction',r'(C) Dot product $p_u^\top q_i$','Mean raw score component'),('mean_bu',r'(D) User bias $b_u$','Mean raw score component'),('mean_prediction',r'(E) Predicted rating $\hat r_{ui}$','Mean raw predicted rating')]
 for ax,(key,title,ylabel) in zip(axs,panels):
  for model in ['svd','nmf']:
   for group,style in [('target','-'),('non_target','--')]:
    d=f[(f.model==model)&(f.group==group)].set_index('n_fake').loc[ns]
    assert len(d)==len(ns) and np.isfinite(d[key]).all()
    assert np.allclose(d.mean_prediction,d.mu+d.mean_bu+d.mean_bi+d.mean_interaction,atol=1e-10)
    ax.plot(x,d[key],color=colors[model],ls=style,lw=2.3 if key=='mean_prediction' else 1.9,marker='o' if group=='target' else 's',ms=3.2)
  if key=='mean_prediction':
   for group,style in [('target','-'),('non_target','--')]:
    a=f[(f.model=='svd')&(f.group==group)].set_index('n_fake').loc[ns].mean_recorded_rating
    b=f[(f.model=='nmf')&(f.group==group)].set_index('n_fake').loc[ns].mean_recorded_rating
    assert np.allclose(a,b,atol=1e-12)
    ax.plot(x,a,color='#777777',lw=1.25,ls=style,alpha=.85,zorder=2)
  ax.set_title(title,loc='left',pad=10);ax.set_ylabel(ylabel)
  if key in ['mean_bi','mean_interaction','mean_bu']:ax.axhline(0,color='#96A4AE',lw=.7)
  if key=='mae':ax.set_ylim(bottom=0)
  ax.set_xticks(x,[str(n) for n in ns]);ax.tick_params(axis='x',labelsize=9);ax.set_xlabel('Injected users');ax.grid(axis='y',color='#E2E8EC',lw=.65);ax.set_axisbelow(True);ax.margins(x=.04)
 fig.suptitle(genre+': training fit and score components | '+('Synthetic users only' if args.synthetic_only else 'All users (53,424 real + n synthetic)'),x=.08,ha='left',y=.98,fontsize=12.5 if not args.synthetic_only else 15,fontweight='bold')
 handles=[Line2D([0],[0],color=colors['svd'],lw=2.5,label='SVD'),Line2D([0],[0],color=colors['nmf'],lw=2.5,label='NMF'),Line2D([0],[0],color='#536471',lw=1.8,marker='o',ms=4,label=genre),Line2D([0],[0],color='#536471',lw=1.8,ls='--',marker='s',ms=4,label='Non-'+genre),Line2D([0],[0],color='#777777',lw=1.25,label='Recorded rating (E)')]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.945),ncol=5,frameon=False,columnspacing=1.7,fontsize=9)
 fig.text(.5,.018,('Synthetic training ratings only. A: mean individual absolute error. E: grey = recorded rating (target 5; non-target 1).' if args.synthetic_only else 'Real + synthetic training ratings; averages weighted by observed rating pairs. A: mean individual absolute error. E: grey = mean recorded rating.'),ha='center',fontsize=7.8,color='#536471')
 fig.subplots_adjust(left=.08,right=.98,bottom=.09,top=.86,hspace=.70,wspace=.65)
 out=R/'figures/first_reviewer'/folder/('training_'+genre.lower()+('_synthetic_only_overview_5panel.png' if args.synthetic_only else '_all_users_overview_4panel.png'))
 fig.savefig(out,dpi=300,facecolor='white');plt.close(fig);print(out)

