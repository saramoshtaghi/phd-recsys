#!/usr/bin/env python3
"""Compact eight-panel comparison redrawn from the existing diagnostic tables."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path('/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/svd_nmf/layer1')
COUNTS=[2,4,6,25,50,100,200,500]
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';NAVY='#142E49'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
data={}
for model in ('NMF','SVD'):
 p=ROOT/'result/reviewer_hypothesis'/model/'Thriller'
 fake=pd.read_csv(p/'synthetic_score_components.csv');fake=fake[(fake.pair_group=='non_target')&fake.n_fake.isin(COUNTS)].sort_values('n_fake')
 margin=pd.read_csv(p/'candidate_margins.csv');margin=margin[margin.n_fake.isin(COUNTS)].sort_values('n_fake')
 rec=pd.read_csv(p/'recommendation_component_swaps.csv');rec=rec[(rec.k==15)&rec.n_fake.isin(COUNTS)].sort_values('n_fake')
 assert all(d.n_fake.tolist()==COUNTS for d in (fake,margin,rec))
 data[model]=(fake,margin,rec)
def limit(part,cols,extras):
 v=np.r_[np.concatenate([data[m][part][cols].to_numpy().ravel() for m in data]),extras]
 lo,hi=float(v.min()),float(v.max());pad=.14*(hi-lo)
 return lo-pad,hi+pad
limits=[(-3,103),limit(0,['mean_interaction','mean_offset','mean_prediction'],[1]),limit(1,['delta_interaction_margin','delta_bi_margin'],[0]),limit(2,['embedding_swap_allocation','item_bias_swap_allocation','delta_count'],[0])]
fig,axes=plt.subplots(2,4,figsize=(16,9),sharex=True,sharey='col')
x=np.arange(len(COUNTS))
titles=['Negative interactions','Fitting non-target rating 1','Target score advantage','Top-15 exposure change']
ylabels=['Pairs (%)','Mean raw score contribution','Change in score margin','Change in mean Thriller count']
for row,model in enumerate(('NMF','SVD')):
 fake,margin,rec=data[model]
 axes[row,0].plot(x,fake.negative_interaction_percent,'o-',color=BLUE,lw=2.5,ms=5)
 for col,color in [('mean_interaction',BLUE),('mean_offset',ORANGE),('mean_prediction',GREEN)]:axes[row,1].plot(x,fake[col],'o-',color=color,lw=2.5,ms=5)
 axes[row,1].axhline(1,color='#555',ls=':',lw=1.5)
 for col,color in [('delta_interaction_margin',BLUE),('delta_bi_margin',ORANGE)]:axes[row,2].plot(x,margin[col],'o-',color=color,lw=2.5,ms=5)
 for col,color in [('embedding_swap_allocation',BLUE),('item_bias_swap_allocation',ORANGE),('delta_count',GREEN)]:axes[row,3].plot(x,rec[col],'o-',color=color,lw=2.5,ms=5)
 for column,ax in enumerate(axes[row]):
  ax.set_ylim(*limits[column]);ax.set_xticks(x,[str(n) for n in COUNTS],rotation=40,ha='right',fontsize=11)
  ax.tick_params(axis='y',labelsize=11);ax.tick_params(axis='x',labelbottom=True)
  ax.set_ylabel(ylabels[column],fontsize=11,labelpad=7)
  ax.grid(axis='y',color='#E0E5EA',lw=.7);ax.set_axisbelow(True)
  if column in (2,3):ax.axhline(0,color='#8A949E',lw=.8)
  if row==0:ax.set_title(titles[column],fontsize=14,weight='bold',color=NAVY,pad=14)
  else:ax.set_xlabel('Injected users',fontsize=12,labelpad=8)
fig.subplots_adjust(left=.075,right=.98,bottom=.22,top=.78,wspace=.43,hspace=.42)
fig.text(.045,.945,'Thriller injection: how NMF and SVD respond',fontsize=26,weight='bold',color=NAVY)
fig.text(.045,.889,'2–500 injected users · Same measures and linear scales in each column',fontsize=15,color='#50657A')
fig.text(.019,.66,'NMF',rotation=90,fontsize=20,weight='bold',color=NAVY,va='center')
fig.text(.019,.36,'SVD',rotation=90,fontsize=20,weight='bold',color=NAVY,va='center')
handles=[Line2D([0],[0],color=BLUE,lw=3,marker='o',label='Interaction / embedding'),Line2D([0],[0],color=ORANGE,lw=3,marker='o',label='Bias component'),Line2D([0],[0],color=GREEN,lw=3,marker='o',label='Prediction / total exposure change'),Line2D([0],[0],color='#555',lw=1.5,ls=':',label='Observed rating = 1')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.105),ncol=4,frameon=False,fontsize=12,columnspacing=1.4)
fig.text(.045,.067,r'Rating fit: bias offset = $\mu+b_u+b_i$. Columns 3–4 show changes from baseline for real users.',fontsize=11.5,color='#50657A')
fig.text(.045,.025,'Component swaps describe fitted-score contributions; they do not establish a causal training mechanism.',fontsize=11.5,color='#50657A')
path=ROOT.parent/'layer2/final_results/Thriller_NMF_SVD_components_powerpoint.png'
fig.savefig(path,dpi=240,facecolor='white');plt.close(fig);print('SAVED',path)
